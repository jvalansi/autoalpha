"""Fetch full earnings press releases (8-K Item 2.02, exhibit 99.x) from SEC EDGAR.

Second run of the LLM press-release test (PLAN.md Phase 5). For every reaction bar
(days_since_earnings == 0, prior close >= $5) in [--start, --end] across the loop and
vault datasets, find the company's 8-K with Item 2.02 accepted between the previous
bar's 16:00 ET close and the reaction bar's 09:30 ET open, and take the text of its
first EX-99 exhibit (the press release). Same timing rule as score_press_releases.py;
EDGAR's acceptance timestamp is UTC.

Tickers map to CIKs through SEC's current company_tickers.json, so names that have
since been delisted or renamed are missed.

Caches (all write temp + rename, resumable): data/llm_news_sec/submissions/{cik}.json,
data/llm_news_sec/docs/{accession}.txt. Output: data/llm_news_sec/releases.parquet.

Usage:
    python scripts/fetch_sec_releases.py [--start 2023-11-01 --end 2024-06-28]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DIR = Path("data/llm_news_sec")
UA = {"User-Agent": "autoalpha research jvalansi1@gmail.com"}
CALLS_PER_SEC = 8  # SEC fair-access limit is 10/s
MAX_CHARS = 12_000
DATASETS = ["data/loop_data.parquet", "data/vault_data.parquet"]

_lock = threading.Lock()
_next_slot = [0.0]


def _throttle() -> None:
    with _lock:
        now = time.monotonic()
        wait = _next_slot[0] - now
        _next_slot[0] = max(now, _next_slot[0]) + 1.0 / CALLS_PER_SEC
    if wait > 0:
        time.sleep(wait)


def sec_get(url: str) -> requests.Response:
    for attempt in range(4):
        _throttle()
        try:
            r = requests.get(url, headers=UA, timeout=30)
        except (requests.ConnectionError, requests.Timeout):
            err = "network"
        else:
            if r.status_code == 200:
                return r
            err = f"HTTP {r.status_code}"
            if r.status_code < 500 and r.status_code != 429:
                break
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{url}: {err}")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.rename(path)


def load_events(start: str, end: str) -> pd.DataFrame:
    frames = []
    lo = (pd.Timestamp(start) - pd.Timedelta(days=10)).to_pydatetime()
    for path in DATASETS:
        t = ds.dataset(path).to_table(
            columns=["date", "ticker", "Close", "days_since_earnings"],
            filter=(ds.field("date") >= lo) & (ds.field("date") <= pd.Timestamp(end).to_pydatetime())).to_pandas()
        frames.append(t.reset_index() if "ticker" not in t.columns else t)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["date", "ticker"], keep="last").sort_values(["ticker", "date"])
    g = df.groupby("ticker")
    df["prev_close"], df["prev_date"] = g["Close"].shift(1), g["date"].shift(1)
    ev = df[(df["days_since_earnings"] == 0) & (df["date"] >= start) & (df["prev_close"] >= 5)]
    return ev[["ticker", "date", "prev_date"]].reset_index(drop=True)


def submissions(cik: int, since: str) -> pd.DataFrame:
    """8-K filings with Item 2.02 for a CIK: accession, acceptance (ET, naive)."""
    path = DIR / "submissions" / f"{cik}.json"
    if path.exists():
        rows = json.loads(path.read_text())
    else:
        d = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
        blocks = [d["filings"]["recent"]]
        for f in d["filings"].get("files", []):
            if f.get("filingTo", "") >= since:
                blocks.append(sec_get(f"https://data.sec.gov/submissions/{f['name']}").json())
        rows = []
        for b in blocks:
            for form, items, acc, accepted in zip(b["form"], b["items"], b["accessionNumber"], b["acceptanceDateTime"]):
                if form == "8-K" and "2.02" in (items or ""):
                    rows.append({"accession": acc, "accepted": accepted})
        _atomic_write(path, json.dumps(rows))
    df = pd.DataFrame(rows, columns=["accession", "accepted"])
    df["accepted"] = (pd.to_datetime(df["accepted"], utc=True, errors="coerce")
                      .dt.tz_convert("America/New_York").dt.tz_localize(None))
    return df.dropna()


def release_text(cik: int, accession: str) -> str:
    path = DIR / "docs" / f"{accession}.txt"
    if path.exists():
        return path.read_text()
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}"
    headers = sec_get(f"{base}/{accession}-index-headers.html").text.replace("&lt;", "<").replace("&gt;", ">")
    docs = re.findall(r"<TYPE>([^\s<]+)\s*.*?<FILENAME>([^\s<]+)", headers, re.S)
    exhibit = next((name for typ, name in docs if typ.upper().startswith("EX-99")), None)
    text = ""
    if exhibit:
        raw = sec_get(f"{base}/{exhibit}").content
        text = BeautifulSoup(raw, "lxml").get_text(" ")
        text = re.sub(r"[\s\u200b]+", " ", text).strip()[:MAX_CHARS]
    _atomic_write(path, text)
    return text


def fetch_ticker(ticker: str, cik: int, evs: pd.DataFrame, since: str) -> list[dict]:
    subs = submissions(cik, since)
    out = []
    for ev in evs.itertuples():
        lo, hi = ev.prev_date + pd.Timedelta(hours=16), ev.date + pd.Timedelta(hours=9, minutes=30)
        hits = subs[(subs["accepted"] >= lo) & (subs["accepted"] < hi)].sort_values("accepted")
        if hits.empty:
            continue
        h = hits.iloc[-1]
        text = release_text(cik, h["accession"])
        if text:
            out.append({"ticker": ticker, "date": ev.date, "accession": h["accession"],
                        "accepted": h["accepted"], "text": text})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-11-01")
    parser.add_argument("--end", default="2024-06-28")
    args = parser.parse_args()
    for sub in ("submissions", "docs"):
        (DIR / sub).mkdir(parents=True, exist_ok=True)

    events = load_events(args.start, args.end)
    ciks = {v["ticker"].upper(): int(v["cik_str"])
            for v in sec_get("https://www.sec.gov/files/company_tickers.json").json().values()}
    # vault tickers use '-' for share classes (BRK-B); SEC uses the same
    events["cik"] = events["ticker"].str.upper().map(ciks)
    log.info("%d events, %d tickers, %d with a CIK", len(events), events["ticker"].nunique(),
             events["cik"].notna().sum())
    events = events.dropna(subset=["cik"])

    rows, failed = [], 0
    groups = list(events.groupby("ticker"))
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch_ticker, tk, int(g["cik"].iloc[0]), g, args.start): tk for tk, g in groups}
        for i, fut in enumerate(futures, 1):
            try:
                rows.extend(fut.result())
            except Exception as exc:  # rerun resumes from the caches
                failed += 1
                log.warning("%s: %s", futures[fut], exc)
            if i % 200 == 0:
                log.info("%d/%d tickers, %d releases, %d failed", i, len(groups), len(rows), failed)
    df = pd.DataFrame(rows)
    tmp = DIR / "releases.parquet.tmp"
    df.to_parquet(tmp)
    tmp.rename(DIR / "releases.parquet")
    log.info("done: %d releases for %d events (%d tickers failed)", len(df), len(events), failed)


if __name__ == "__main__":
    main()
