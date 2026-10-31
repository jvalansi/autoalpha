"""Cache FMP press releases for every ticker with an earnings reaction bar in the vault.

Feeds the LLM news test (scripts/score_press_releases.py, scripts/evaluate_llm_news.py).
FMP v3 /press-releases/{ticker} pages newest-first, 100 per page; we page until the
oldest release is before --since. Recent releases carry only the first ~500
characters of text (headline + lede), older ones the full body.

One JSON file per ticker under data/llm_news/press/, written via temp file + rename
so a kill never leaves a half-written cache. Existing files are skipped (resumable).

Usage:
    FMP_API_KEY=... python scripts/fetch_press_releases.py [--since 2024-06-01]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_live_pead import fmp_get, redact

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

URL = "https://financialmodelingprep.com/api/v3/press-releases/{ticker}"
OUT = Path("data/llm_news/press")
MAX_PAGES = 30
CALLS_PER_SEC = 10  # Premium plan allows 750/min

_lock = threading.Lock()
_next_slot = [0.0]


def _throttle() -> None:
    with _lock:
        now = time.monotonic()
        wait = _next_slot[0] - now
        _next_slot[0] = max(now, _next_slot[0]) + 1.0 / CALLS_PER_SEC
    if wait > 0:
        time.sleep(wait)


def _get(ticker: str, page: int, key: str) -> list:
    _throttle()
    return fmp_get(URL.format(ticker=ticker), {"page": page, "apikey": key}, timeout=20)


def fetch_ticker(ticker: str, since: str, key: str) -> int:
    path = OUT / f"{ticker}.json"
    if path.exists():
        return -1
    releases = []
    for page in range(MAX_PAGES):
        rows = _get(ticker, page, key)
        if not rows:
            break
        releases.extend(rows)
        dated = [r["date"] for r in rows if r.get("date")]
        if dated and min(dated) < since:
            break
    else:
        log.warning("%s: hit %d pages before %s; older releases missing", ticker, MAX_PAGES, since)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([r for r in releases if r.get("date", "") >= since]))
    tmp.rename(path)
    return len(releases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2024-06-01")
    args = parser.parse_args()
    key = os.environ["FMP_API_KEY"]
    OUT.mkdir(parents=True, exist_ok=True)

    t = ds.dataset("data/vault_data.parquet").to_table(
        columns=["ticker"], filter=ds.field("days_since_earnings") == 0).to_pandas()
    tickers = sorted(t.index.unique() if "ticker" not in t.columns else t["ticker"].unique())
    log.info("%d tickers", len(tickers))

    done = failed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_ticker, tk, args.since, key): tk for tk in tickers}
        for fut in futures:
            try:
                fut.result()
            except Exception as exc:  # keep going; rerun picks up missing files
                failed += 1
                log.warning("%s: %s", futures[fut], redact(exc))
            done += 1
            if done % 200 == 0:
                log.info("%d/%d tickers (%d failed)", done, len(tickers), failed)
    log.info("done: %d tickers, %d failed", done, failed)


if __name__ == "__main__":
    main()
