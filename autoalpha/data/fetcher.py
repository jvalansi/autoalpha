"""Unified data fetcher with vault holdout enforcement.

Wraps yfinance (prices) and FMP (earnings/fundamentals).
Raises VaultLeakError if any requested date range overlaps the holdout window.
Caches all fetched data as Parquet at data/cache/{ticker}/{year}.parquet.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

_VAULT_PATH = Path(__file__).resolve().parents[2] / "vault_holdout.json"
_CACHE_ROOT = Path("data/cache")

FMP_BASE = "https://financialmodelingprep.com/stable"


class VaultLeakError(Exception):
    """Raised when a data request overlaps the held-out vault window."""


def _load_vault() -> tuple[date, date]:
    with open(_VAULT_PATH) as f:
        v = json.load(f)
    return date.fromisoformat(v["holdout_start"]), date.fromisoformat(v["holdout_end"])


def _check_vault(start: date, end: date) -> None:
    vault_start, vault_end = _load_vault()
    if start <= vault_end and end >= vault_start:
        raise VaultLeakError(
            f"Requested [{start}, {end}] overlaps vault holdout [{vault_start}, {vault_end}]"
        )


def get_ohlcv(
    ticker: str,
    start: date,
    end: date,
    cache_dir: Path = _CACHE_ROOT,
) -> pd.DataFrame:
    """Return OHLCV DataFrame indexed by date. Raises VaultLeakError on overlap.

    Cache layout: data/cache/{ticker}/{year}.parquet — one file per calendar year.
    """
    _check_vault(start, end)

    ticker_dir = cache_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(start.year, end.year + 1):
        cache_path = ticker_dir / f"{year}.parquet"
        if cache_path.exists():
            df_year = pd.read_parquet(cache_path)
        else:
            year_start = date(year, 1, 1)
            year_end = date(year, 12, 31)
            tk = yf.Ticker(ticker)
            df_year = tk.history(
                start=year_start.isoformat(),
                end=(year_end + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=True,
            )
            if df_year.empty:
                frames.append(pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]))
                continue
            df_year = df_year[["Open", "High", "Low", "Close", "Volume"]]
            df_year.index = pd.to_datetime(df_year.index).tz_localize(None).normalize()
            df_year.index.name = "date"
            df_year.to_parquet(cache_path)
        frames.append(df_year)

    if not frames:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.concat(frames).sort_index()
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df[mask]


def get_earnings(
    ticker: str,
    start: date,
    end: date,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Return earnings surprise data from FMP for the given ticker and date range.

    Columns: date, eps_actual, eps_estimate, rev_actual, rev_estimate.
    """
    _check_vault(start, end)
    key = api_key or os.environ.get("FMP_API_KEY", "")
    if not key:
        raise EnvironmentError("FMP_API_KEY not set")

    url = f"{FMP_BASE}/earnings"
    params = {"symbol": ticker.upper(), "apikey": key, "limit": 40}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return df.reset_index(drop=True)


def get_transcripts(
    ticker: str,
    year: int,
    quarter: int,
    api_key: Optional[str] = None,
) -> str:
    """Return earnings call transcript text from FMP."""
    # Approximate the transcript date as the last day of the quarter for vault check.
    _quarter_end = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    month, day = _quarter_end[quarter]
    approx_date = date(year, month, day)
    _check_vault(approx_date, approx_date)

    key = api_key or os.environ.get("FMP_API_KEY", "")
    if not key:
        raise EnvironmentError("FMP_API_KEY not set")

    url = f"{FMP_BASE}/earning_call_transcript"
    params = {"symbol": ticker.upper(), "year": year, "quarter": quarter, "apikey": key}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return ""
    return data[0].get("content", "")


def get_estimates(
    ticker: str,
    start: date,
    end: date,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Return quarterly analyst consensus EPS and revenue estimates from FMP.

    Columns: date, estimatedEpsAvg, estimatedRevenueAvg.
    """
    _check_vault(start, end)
    key = api_key or os.environ.get("FMP_API_KEY", "")
    if not key:
        raise EnvironmentError("FMP_API_KEY not set")

    url = f"{FMP_BASE}/analyst-estimates"
    params = {"symbol": ticker.upper(), "period": "quarterly", "limit": 20, "apikey": key}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    keep_cols = [c for c in ["date", "estimatedEpsAvg", "estimatedRevenueAvg"] if c in df.columns]
    df = df[keep_cols]
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return df.reset_index(drop=True)


def get_fundamentals(
    ticker: str,
    start: date,
    end: date,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Return quarterly fundamental data (ROE, leverage, net_margin) from FMP."""
    _check_vault(start, end)
    key = api_key or os.environ.get("FMP_API_KEY", "")
    if not key:
        raise EnvironmentError("FMP_API_KEY not set")

    url = f"{FMP_BASE}/income-statement"
    params = {"symbol": ticker.upper(), "period": "quarter", "limit": 20, "apikey": key}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    income = resp.json()

    url2 = f"{FMP_BASE}/balance-sheet-statement"
    resp2 = requests.get(url2, params={**params, "limit": 20}, timeout=15)
    resp2.raise_for_status()
    balance = resp2.json()

    if not income or not balance:
        return pd.DataFrame()

    inc_df = pd.DataFrame(income)[["date", "netIncome", "revenue", "eps"]]
    bal_df = pd.DataFrame(balance)[["date", "totalStockholdersEquity", "netDebt"]]
    df = inc_df.merge(bal_df, on="date", how="inner")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()
    # Derived ratios used by the quality strategy (quality.py)
    equity = df["totalStockholdersEquity"].replace(0, float("nan"))
    df["roe"] = df["netIncome"] / equity
    revenue = df["revenue"].replace(0, float("nan"))
    df["net_margin"] = df["netIncome"] / revenue
    return df.reset_index(drop=True)
