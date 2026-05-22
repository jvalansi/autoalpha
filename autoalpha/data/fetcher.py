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
    """Return OHLCV DataFrame indexed by date. Raises VaultLeakError on overlap."""
    _check_vault(start, end)

    cache_path = cache_dir / ticker / f"{start.year}_{end.year}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        cached = df[mask]
        if not cached.empty:
            return cached

    tk = yf.Ticker(ticker)
    df = tk.history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=True,
    )
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


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
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return df.reset_index(drop=True)
