"""Fetch key-metrics (dividend_yield, fcf_yield) and sector for all tickers.

Caches to:
  data/cache/{ticker}/fmp_key_metrics.parquet
  data/cache/{ticker}/fmp_sector.json

Usage:
    python scripts/fetch_new_features.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

CACHE_ROOT = Path("data/cache")
FMP_BASE = "https://financialmodelingprep.com/stable"
TICKERS = sorted(d.name for d in CACHE_ROOT.iterdir() if d.is_dir())


def fetch_key_metrics(ticker: str, api_key: str) -> None:
    path = CACHE_ROOT / ticker / "fmp_key_metrics.parquet"
    if path.exists():
        print(f"  {ticker} key_metrics: cached")
        return
    params = {"symbol": ticker, "period": "quarter", "limit": 40, "apikey": api_key}
    resp = requests.get(f"{FMP_BASE}/key-metrics", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json() or []
    if not data:
        print(f"  {ticker} key_metrics: no data")
        return
    df = pd.DataFrame(data)
    col_map = {"date": "date", "dividendYield": "dividend_yield", "freeCashFlowYield": "fcf_yield"}
    available = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(available)].rename(columns=available)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df.to_parquet(path, index=False)
    print(f"  {ticker} key_metrics: {len(df)} rows")


def fetch_sector(ticker: str, api_key: str) -> None:
    path = CACHE_ROOT / ticker / "fmp_sector.json"
    if path.exists():
        print(f"  {ticker} sector: cached")
        return
    params = {"symbol": ticker, "apikey": api_key}
    resp = requests.get(f"{FMP_BASE}/profile", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json() or []
    if not data:
        print(f"  {ticker} sector: no data")
        return
    sector = data[0].get("sector", "Unknown") if isinstance(data, list) else data.get("sector", "Unknown")
    path.write_text(json.dumps({"sector": sector}))
    print(f"  {ticker} sector: {sector}")


def main() -> None:
    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        raise EnvironmentError("FMP_API_KEY not set")

    print(f"Fetching new features for {len(TICKERS)} tickers...")
    for ticker in TICKERS:
        print(f"{ticker}:")
        try:
            fetch_key_metrics(ticker, api_key)
        except Exception as exc:
            print(f"  {ticker} key_metrics ERROR: {exc}")
        time.sleep(0.25)
        try:
            fetch_sector(ticker, api_key)
        except Exception as exc:
            print(f"  {ticker} sector ERROR: {exc}")
        time.sleep(0.25)

    print("Done.")


if __name__ == "__main__":
    main()
