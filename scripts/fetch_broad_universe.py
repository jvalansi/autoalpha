"""Fetch OHLCV + FMP data for all US stocks >= $300M market cap.

Idempotent — skips tickers that already have a complete cache.
Estimated runtime: 2-3 hours for ~2,400 new tickers.

Usage:
    FMP_API_KEY=<key> python scripts/fetch_broad_universe.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CACHE_ROOT = Path("data/cache")
FMP_BASE = "https://financialmodelingprep.com/stable"
MIN_MARKET_CAP = 300_000_000
OHLCV_START = "2018-01-01"
SLEEP = 0.25


def get_universe(api_key: str) -> list[str]:
    """Return sorted list of US stocks >= MIN_MARKET_CAP from FMP screener."""
    r = requests.get(
        "https://financialmodelingprep.com/api/v3/stock-screener",
        params={
            "exchange": "NYSE,NASDAQ,AMEX",
            "country": "US",
            "isEtf": "false",
            "isFund": "false",
            "isActivelyTrading": "true",
            "marketCapMoreThan": MIN_MARKET_CAP,
            "limit": 10000,
            "apikey": api_key,
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    tickers = sorted(s["symbol"] for s in data if s.get("symbol"))
    log.info("Screener returned %d tickers >= $%dM", len(tickers), MIN_MARKET_CAP // 1_000_000)
    return tickers


def is_complete(ticker: str) -> bool:
    d = CACHE_ROOT / ticker
    if not d.exists():
        return False
    files = {f.name for f in d.iterdir()}
    required = {"fmp_fundamentals.parquet", "fmp_earnings.parquet", "fmp_estimates.parquet",
                "fmp_valuation.parquet", "fmp_key_metrics.parquet", "fmp_sector.json",
                "2025.parquet", "2026.parquet"}
    return required.issubset(files) and bool(list(d.glob("[0-9][0-9][0-9][0-9].parquet")))


def fetch_ohlcv(ticker: str) -> bool:
    ticker_dir = CACHE_ROOT / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    try:
        hist = yf.Ticker(ticker).history(start=OHLCV_START, end="2027-01-01", auto_adjust=True)
        if hist.empty:
            return False
        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
        hist.index.name = "date"
        for year, grp in hist.groupby(hist.index.year):
            out = ticker_dir / f"{year}.parquet"
            if not out.exists():
                grp.to_parquet(out)
        return True
    except Exception as exc:
        log.warning("%-8s  OHLCV failed: %s", ticker, exc)
        return False


def fmp_get(endpoint: str, params: dict) -> list:
    try:
        r = requests.get(f"{FMP_BASE}/{endpoint}", params=params, timeout=15)
        r.raise_for_status()
        return r.json() or []
    except Exception as exc:
        log.warning("FMP %s %s failed: %s", endpoint, params.get("symbol", ""), exc)
        return []


def fetch_fundamentals(ticker: str, api_key: str) -> None:
    path = CACHE_ROOT / ticker / "fmp_fundamentals.parquet"
    if path.exists():
        return
    data = fmp_get("income-statement", {"symbol": ticker, "period": "quarter", "limit": 40, "apikey": api_key})
    time.sleep(SLEEP)
    if not data:
        return
    bal = fmp_get("balance-sheet-statement", {"symbol": ticker, "period": "quarter", "limit": 40, "apikey": api_key})
    time.sleep(SLEEP)
    if not bal:
        return
    df_inc = pd.DataFrame(data)
    df_bal = pd.DataFrame(bal)
    keep_inc = [c for c in ["date", "netIncome", "revenue"] if c in df_inc.columns]
    keep_bal = [c for c in ["date", "totalStockholdersEquity", "netDebt"] if c in df_bal.columns]
    if "date" not in df_inc.columns or "date" not in df_bal.columns:
        return
    merged = df_inc[keep_inc].merge(df_bal[keep_bal], on="date", how="inner")
    merged.to_parquet(path, index=False)


def fetch_earnings(ticker: str, api_key: str) -> None:
    path = CACHE_ROOT / ticker / "fmp_earnings.parquet"
    if path.exists():
        return
    data = fmp_get("earnings", {"symbol": ticker, "limit": 40, "apikey": api_key})
    time.sleep(SLEEP)
    if not data:
        return
    pd.DataFrame(data).to_parquet(path, index=False)


def fetch_estimates(ticker: str, api_key: str) -> None:
    path = CACHE_ROOT / ticker / "fmp_estimates.parquet"
    if path.exists():
        return
    data = fmp_get("analyst-estimates", {"symbol": ticker, "period": "quarter", "limit": 40, "apikey": api_key})
    time.sleep(SLEEP)
    if not data:
        return
    pd.DataFrame(data).to_parquet(path, index=False)


def fetch_valuation(ticker: str, api_key: str) -> None:
    path = CACHE_ROOT / ticker / "fmp_valuation.parquet"
    if path.exists():
        return
    data = fmp_get("ratios", {"symbol": ticker, "period": "quarter", "limit": 40, "apikey": api_key})
    time.sleep(SLEEP)
    if not data:
        return
    df = pd.DataFrame(data)
    col_map = {"date": "date", "priceEarningsRatio": "pe_ratio", "priceToBookRatio": "pb_ratio",
               "priceToSalesRatio": "ps_ratio", "enterpriseValueMultiple": "ev_ebitda"}
    available = {k: v for k, v in col_map.items() if k in df.columns}
    df[list(available)].rename(columns=available).to_parquet(path, index=False)


def fetch_key_metrics(ticker: str, api_key: str) -> None:
    path = CACHE_ROOT / ticker / "fmp_key_metrics.parquet"
    if path.exists():
        return
    data = fmp_get("key-metrics", {"symbol": ticker, "period": "quarter", "limit": 40, "apikey": api_key})
    time.sleep(SLEEP)
    if not data:
        pd.DataFrame(columns=["date", "fcf_yield"]).to_parquet(path, index=False)
        return
    df = pd.DataFrame(data)
    col_map = {"date": "date", "dividendYield": "dividend_yield", "freeCashFlowYield": "fcf_yield"}
    available = {k: v for k, v in col_map.items() if k in df.columns}
    df[list(available)].rename(columns=available).to_parquet(path, index=False)


def fetch_sector(ticker: str, api_key: str) -> None:
    path = CACHE_ROOT / ticker / "fmp_sector.json"
    if path.exists():
        return
    data = fmp_get("profile", {"symbol": ticker, "apikey": api_key})
    time.sleep(SLEEP)
    sector = data[0].get("sector", "Unknown") if data else "Unknown"
    path.write_text(json.dumps({"sector": sector}))


def fetch_ticker(ticker: str, api_key: str) -> bool:
    if is_complete(ticker):
        return True
    if not fetch_ohlcv(ticker):
        return False
    fetch_fundamentals(ticker, api_key)
    fetch_earnings(ticker, api_key)
    fetch_estimates(ticker, api_key)
    fetch_valuation(ticker, api_key)
    fetch_key_metrics(ticker, api_key)
    fetch_sector(ticker, api_key)
    return True


def main() -> None:
    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        log.error("FMP_API_KEY not set")
        sys.exit(1)

    universe = get_universe(api_key)

    already_done = sum(1 for t in universe if is_complete(t))
    log.info("Already complete: %d/%d  Need to fetch: %d", already_done, len(universe), len(universe) - already_done)

    ok, failed = 0, []
    for i, ticker in enumerate(universe, 1):
        if is_complete(ticker):
            continue
        if i % 100 == 0:
            log.info("Progress: %d/%d  ok=%d  failed=%d", i, len(universe), ok, len(failed))
        success = fetch_ticker(ticker, api_key)
        if success:
            ok += 1
        else:
            failed.append(ticker)

    log.info("Done: %d fetched OK, %d failed", ok, len(failed))
    if failed:
        log.warning("Failed: %s", failed)
        (Path("data") / "fetch_failed.json").write_text(json.dumps(failed))


if __name__ == "__main__":
    main()
