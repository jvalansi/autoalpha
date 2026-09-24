"""Refetch quarterly fundamentals and rewrite them point-in-time (keyed on filing date).

Fetches 4 FMP endpoints per ticker for every ticker in the vault (and loop data with
--loop), caches the result to data/cache/fundamentals.parquet, and rewrites the
FUND_COLS columns of the parquet(s) in place. Idempotent; run weekly for the vault
so new filings land (update_vault.py forward-fills between runs).

Usage:
    FMP_API_KEY=... python scripts/refresh_fundamentals.py [--loop] [--use-cache]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_event_features import LOOP_PATH, VAULT_PATH, rewrite_columns
from autoalpha.data.fundamentals import FUND_COLS, fetch_fundamentals, pit_fundamentals

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CACHE_PATH = Path("data/cache/fundamentals.parquet")
CALLS_PER_TICKER = 4
MAX_CALLS_PER_MIN = 500   # plan allows 750/min; leave room for earnings-trader on the same key
MAX_FAILED_SHARE = 0.02   # refuse to rewrite if more tickers than this failed to fetch


class _RateLimiter:
    def __init__(self, per_min: float):
        self.interval = 60.0 / per_min
        self.lock = threading.Lock()
        self.next_t = time.monotonic()

    def wait(self, n: int) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_t)
            self.next_t = start + n * self.interval
        time.sleep(max(0.0, start - now))


def fetch_all(tickers: list[str], api_key: str) -> tuple[pd.DataFrame, list[str]]:
    limiter = _RateLimiter(MAX_CALLS_PER_MIN)
    local = threading.local()

    def one(ticker: str) -> pd.DataFrame | None:
        if not hasattr(local, "session"):
            local.session = requests.Session()
        for attempt in range(4):
            limiter.wait(CALLS_PER_TICKER)
            try:
                df = fetch_fundamentals(ticker, api_key, local.session)
                df.insert(0, "ticker", ticker)
                return df
            except requests.RequestException as exc:
                if attempt == 3:
                    log.warning("%s failed: %s", ticker, exc)
                    return None
                time.sleep(5 * (attempt + 1))

    frames, failed = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (ticker, df) in enumerate(zip(tickers, pool.map(one, tickers)), 1):
            if df is None:
                failed.append(ticker)
            elif not df.empty:
                frames.append(df)
            if i % 250 == 0:
                log.info("Fetched %d/%d (failed %d)", i, len(tickers), len(failed))
    return pd.concat(frames, ignore_index=True), failed


def rewrite(path: Path, table: pd.DataFrame) -> None:
    bars = pq.read_table(path, columns=["date", "ticker"]).to_pandas().reset_index()[["date", "ticker"]]
    new = pit_fundamentals(bars, table)
    rewrite_columns(path, new)
    log.info("%s: rewrote %d rows, non-null share %s", path, len(new),
             {c: round(float(new[c].notna().mean()), 3) for c in FUND_COLS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="also rewrite loop_data.parquet")
    parser.add_argument("--use-cache", action="store_true", help="skip fetching; use the cached table")
    args = parser.parse_args()
    paths = [VAULT_PATH, LOOP_PATH] if args.loop else [VAULT_PATH]

    if args.use_cache:
        table = pd.read_parquet(CACHE_PATH)
    else:
        tickers = sorted(set().union(*(
            set(pq.read_table(p, columns=["ticker"]).column("ticker").to_pylist()) for p in paths)))
        log.info("Fetching fundamentals for %d tickers", len(tickers))
        table, failed = fetch_all(tickers, os.environ["FMP_API_KEY"])
        if len(failed) > MAX_FAILED_SHARE * len(tickers):
            raise SystemExit(f"{len(failed)} tickers failed to fetch — not rewriting: {failed[:20]}")
        if failed and CACHE_PATH.exists():  # keep last week's rows rather than blanking a ticker
            prev = pd.read_parquet(CACHE_PATH)
            table = pd.concat([table, prev[prev["ticker"].isin(failed)]], ignore_index=True)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(CACHE_PATH, index=False)
        log.info("Cached %d quarters for %d tickers (%d failed)", len(table), table["ticker"].nunique(), len(failed))

    for path in paths:
        rewrite(path, table)


if __name__ == "__main__":
    main()
