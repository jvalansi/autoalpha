"""Fetch 2025 and 2026 OHLCV for tickers missing those year files."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_ROOT = Path("data/cache")


def fetch_year(ticker: str, year: int) -> None:
    out = CACHE_ROOT / ticker / f"{year}.parquet"
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01" if year < 2026 else "2026-12-31"
    try:
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if df.empty:
            return
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        df.to_parquet(out)
    except Exception as exc:
        print(f"  FAIL {ticker} {year}: {exc}")


def main() -> None:
    missing: list[tuple[str, int]] = []
    for ticker_dir in sorted(CACHE_ROOT.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        files = [f.name for f in ticker_dir.glob("*.parquet")]
        for year in (2025, 2026):
            if f"{year}.parquet" not in files:
                missing.append((ticker, year))

    print(f"Fetching {len(missing)} ticker-year pairs...")
    for i, (ticker, year) in enumerate(missing, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(missing)} done...")
        fetch_year(ticker, year)
        time.sleep(0.1)

    print("Done.")


if __name__ == "__main__":
    main()
