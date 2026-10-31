"""Score earnings-window press releases with an LLM (OpenAI Batch API).

Event = vault reaction bar (days_since_earnings == 0) on or after --since, prior close
>= $5. Text = the ticker's FMP press releases timestamped in the 4 calendar days before
the window from the previous bar's 16:00 ET close to the reaction bar's 09:30 ET open
(FMP times are ET) whose title reads as a results release (EARNINGS_TITLE and not
SCHEDULING_TITLE), newest first, at most 3. A release issued during or before the
previous session was already priced on that bar, so the event is mis-dated and dropped.
The title filter drops dividend notices and earnings-date / conference-call
announcements; it does not catch another issuer's results in the ticker's feed.
Everything the model sees was public before the bar we enter at.

The model (gpt-4.1-mini, knowledge cutoff 2024-06-01) must have finished training
before every event, so its training data cannot contain the price reaction.

Steps:
    python scripts/score_press_releases.py build     # data/llm_news/requests.jsonl + events.parquet
    python scripts/score_press_releases.py submit    # upload + create batch, id -> batch.json
    python scripts/score_press_releases.py collect   # poll; on completion -> scores.parquet
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

DIR = Path("data/llm_news")
MODEL = "gpt-4.1-mini-2025-04-14"
MAX_RELEASES = 3
MAX_CHARS = 1500
EARNINGS_TITLE = re.compile(
    r"RESULTS|EARNINGS|\bREPORTS?\b.*\b(QUARTER|Q[1-4]|FISCAL|FULL[- ]YEAR|REVENUE|SALES)\b", re.I)
SCHEDULING_TITLE = re.compile(
    r"\b(TO|WILL) (REPORT|ANNOUNCE|RELEASE|HOST|ISSUE|PUBLISH)\b|\bR?E?SCHEDULE|\bDATES?\b|\bSETS\b"
    r"|CONFERENCE CALL|WEBCAST|\bINVITATION\b|\bNOTICE OF\b", re.I)


def is_results_title(title: str) -> bool:
    return bool(EARNINGS_TITLE.search(title)) and not SCHEDULING_TITLE.search(title)

SYSTEM = "You are a financial analyst. Reply with JSON only."
PROMPT = """Company ticker: {ticker}
Below are the company's press releases published in the days before the US market opened on {date}, newest first.

{releases}

Judge how this news will move the stock over the next two weeks relative to the overall market. Weigh the results against what investors most likely expected, any guidance, and the tone.
Return JSON: {{"score": integer from -5 (very negative) to 5 (very positive), "guidance": "raised" | "maintained" | "lowered" | "none"}}"""


def load_events(since: str) -> pd.DataFrame:
    t = ds.dataset("data/vault_data.parquet").to_table(
        columns=["date", "ticker", "Close", "days_since_earnings"],
        filter=ds.field("date") >= pd.Timestamp(since) - pd.Timedelta(days=10)).to_pandas()
    df = t.reset_index() if "ticker" not in t.columns else t
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["date", "ticker"], keep="last").sort_values(["ticker", "date"])
    g = df.groupby("ticker")
    df["prev_close"], df["prev_date"] = g["Close"].shift(1), g["date"].shift(1)
    ev = df[(df["days_since_earnings"] == 0) & (df["date"] >= since) & (df["prev_close"] >= 5)]
    return ev[["ticker", "date", "prev_date"]].reset_index(drop=True)


def releases_for(ticker: str, date: pd.Timestamp, prev_date: pd.Timestamp, cache: dict) -> list[dict]:
    if ticker not in cache:
        path = DIR / "press" / f"{ticker}.json"
        rows = json.loads(path.read_text()) if path.exists() else []
        cache.clear()  # events are grouped by ticker; keep one ticker in memory
        stamps = pd.to_datetime([r.get("date") for r in rows], errors="coerce")
        cache[ticker] = [(ts, r) for ts, r in zip(stamps, rows) if pd.notna(ts)]
    cutoff = date + pd.Timedelta(hours=9, minutes=30)
    lo = prev_date + pd.Timedelta(hours=16)
    hits = sorted((x for x in cache[ticker]
                   if lo <= x[0] < cutoff and is_results_title(x[1].get("title") or "")),
                  key=lambda x: x[0], reverse=True)
    return [r for _, r in hits[:MAX_RELEASES]]


def _atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    tmp.rename(path)


def build(since: str) -> None:
    events = load_events(since)
    cache: dict = {}
    lines, kept = [], []
    for ev in events.itertuples():
        rel = releases_for(ev.ticker, ev.date, ev.prev_date, cache)
        if not rel:
            continue
        body = "\n\n".join(f"[{r['date']} ET] {r.get('title', '')}\n{(r.get('text') or '')[:MAX_CHARS]}" for r in rel)
        cid = f"{ev.ticker}|{ev.date:%Y-%m-%d}"
        lines.append(json.dumps({
            "custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": MODEL, "temperature": 0, "max_tokens": 40,
                     "response_format": {"type": "json_object"},
                     "messages": [{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": PROMPT.format(
                                      ticker=ev.ticker, date=f"{ev.date:%Y-%m-%d}", releases=body)}]}}))
        kept.append({"ticker": ev.ticker, "date": ev.date, "n_releases": len(rel),
                     "first_release": rel[-1]["date"], "titles": " || ".join(r.get("title", "") for r in rel)})
    _atomic(DIR / "requests.jsonl", lambda t: t.write_text("\n".join(lines) + "\n"))
    _atomic(DIR / "events.parquet", lambda t: pd.DataFrame(kept).to_parquet(t))
    chars = sum(len(l) for l in lines)
    print(f"{len(events)} events, {len(kept)} with releases ({len(kept) / max(len(events), 1):.0%}); "
          f"~{chars / 4 / 1e6:.1f}M input tokens")


def submit() -> None:
    from openai import OpenAI
    if (DIR / "batch.json").exists():
        sys.exit(f"{DIR / 'batch.json'} exists: a batch was already submitted (delete it to resubmit)")
    client = OpenAI()
    f = client.files.create(file=open(DIR / "requests.jsonl", "rb"), purpose="batch")
    b = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")
    (DIR / "batch.json").write_text(json.dumps({"batch_id": b.id, "input_file_id": f.id}))
    print(b.id, b.status)


def collect(wait: bool) -> None:
    from openai import OpenAI
    client = OpenAI()
    batch_id = json.loads((DIR / "batch.json").read_text())["batch_id"]
    while True:
        b = client.batches.retrieve(batch_id)
        print(b.status, b.request_counts, flush=True)
        if b.status in ("completed", "failed", "expired", "cancelled") or not wait:
            break
        time.sleep(60)
    if b.status != "completed":
        sys.exit(0 if b.status in ("validating", "in_progress", "finalizing") else 1)
    failed = len(client.files.content(b.error_file_id).text.splitlines()) if b.error_file_id else 0
    if not b.output_file_id:
        sys.exit(f"batch completed with no output ({failed} failed requests)")
    rows = []
    for line in client.files.content(b.output_file_id).text.splitlines():
        r = json.loads(line)
        ticker, date = r["custom_id"].split("|")
        out = {"ticker": ticker, "date": pd.Timestamp(date), "score": None, "guidance": None}
        try:
            content = json.loads(r["response"]["body"]["choices"][0]["message"]["content"])
            out["score"] = max(-5, min(5, int(content["score"])))
            g = str(content.get("guidance", "none")).lower()
            out["guidance"] = g if g in ("raised", "maintained", "lowered", "none") else "none"
        except (KeyError, TypeError, ValueError, IndexError):
            pass
        rows.append(out)
    df = pd.DataFrame(rows)
    _atomic(DIR / "scores.parquet", lambda t: df.to_parquet(t))
    print(f"{len(df)} returned, {df['score'].isna().sum()} unparseable, {failed} failed in batch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["build", "submit", "collect"])
    parser.add_argument("--since", default="2024-07-01")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.step == "build":
        build(args.since)
    elif args.step == "submit":
        submit()
    else:
        collect(args.wait)


if __name__ == "__main__":
    main()
