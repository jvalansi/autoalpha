"""Incrementally refresh OHLCV cache for the current year.

Fetches any trading days newer than the latest cached bar for each ticker,
using yfinance bulk download (one request per batch of tickers).

Usage:
    python scripts/refresh_cache_recent.py [--since YYYY-MM-DD] [--batch-size N]
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CACHE_ROOT = Path("data/cache")
YEAR = 2026


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default=None,
                        help="Fetch from this date (default: day after latest cached bar)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Tickers per yfinance download batch (default 200)")
    args = parser.parse_args()

    today = pd.Timestamp.today().normalize()
    tickers = sorted(d.name for d in CACHE_ROOT.iterdir() if d.is_dir())
    log.info("Checking %d tickers for year %d cache updates...", len(tickers), YEAR)

    # Find the global minimum "needs data from" date
    if args.since:
        fetch_start = pd.Timestamp(args.since)
    else:
        latest_dates: list[pd.Timestamp] = []
        for t in tickers:
            fp = CACHE_ROOT / t / f"{YEAR}.parquet"
            if fp.exists():
                try:
                    df = pd.read_parquet(fp)
                    if not df.empty:
                        latest_dates.append(df.index.max())
                except Exception:
                    pass
        if not latest_dates:
            fetch_start = pd.Timestamp(f"{YEAR}-01-01")
        else:
            # Start from the day after the oldest latest date (catch any stragglers)
            fetch_start = min(latest_dates) + pd.Timedelta(days=1)

    if fetch_start >= today:
        log.info("Cache is up to date (fetch_start=%s >= today=%s)", fetch_start.date(), today.date())
        return

    fetch_end = today + pd.Timedelta(days=1)  # yfinance end is exclusive
    log.info("Fetching %s → %s for %d tickers in batches of %d",
             fetch_start.date(), today.date(), len(tickers), args.batch_size)

    n_updated = 0
    for i in range(0, len(tickers), args.batch_size):
        batch = tickers[i:i + args.batch_size]
        batch_str = " ".join(batch)
        try:
            raw = yf.download(
                batch_str,
                start=fetch_start.strftime("%Y-%m-%d"),
                end=fetch_end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            log.warning("Batch %d-%d failed: %s", i, i + len(batch), exc)
            continue

        if raw.empty:
            log.info("  Batch %d-%d: no new data", i, i + len(batch))
            continue

        # yfinance returns MultiIndex columns (field, ticker) for multi-ticker downloads
        if isinstance(raw.columns, pd.MultiIndex):
            fields = raw.columns.get_level_values(0).unique()
            for ticker in batch:
                try:
                    ticker_df = raw.xs(ticker, axis=1, level=1)[list(fields)]
                    ticker_df = ticker_df.dropna(how="all")
                    if ticker_df.empty:
                        continue
                    ticker_df.index = pd.to_datetime(ticker_df.index).tz_localize(None).normalize()
                    ticker_df.index.name = "date"
                    _merge_and_save(ticker, ticker_df)
                    n_updated += 1
                except (KeyError, Exception):
                    pass
        else:
            # Single ticker — shouldn't happen in batch mode but handle it
            ticker = batch[0]
            raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
            raw.index.name = "date"
            _merge_and_save(ticker, raw.dropna(how="all"))
            n_updated += 1

        log.info("  Batch %d-%d processed", i, i + len(batch))
        time.sleep(0.2)  # be gentle with yfinance

    log.info("Done — updated %d ticker cache files", n_updated)


def _merge_and_save(ticker: str, new_df: pd.DataFrame) -> None:
    fp = CACHE_ROOT / ticker / f"{YEAR}.parquet"
    if fp.exists():
        try:
            existing = pd.read_parquet(fp)
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        except Exception:
            combined = new_df
    else:
        fp.parent.mkdir(parents=True, exist_ok=True)
        combined = new_df

    # Keep only OHLCV columns that exist
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in combined.columns]
    combined[keep].to_parquet(fp)


if __name__ == "__main__":
    main()
