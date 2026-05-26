"""Unified data fetcher with vault holdout enforcement and FMP caching.

Wraps yfinance (prices) and FMP (earnings/fundamentals/transcripts).
Raises VaultLeakError if any requested date range overlaps the holdout window.

Cache layout (data/cache/{ticker}/):
  {year}.parquet              — OHLCV (one file per calendar year)
  fmp_earnings.parquet        — full earnings history (date-filtered at read time)
  fmp_estimates.parquet       — full analyst estimates history
  fmp_fundamentals.parquet    — full income+balance fundamentals history
  fmp_transcript_{year}q{q}.txt  — individual quarter transcripts
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


def _ticker_cache_dir(ticker: str, cache_dir: Path = _CACHE_ROOT) -> Path:
    d = cache_dir / ticker.upper()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------

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

    ticker_dir = _ticker_cache_dir(ticker, cache_dir)
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


# ---------------------------------------------------------------------------
# FMP helpers
# ---------------------------------------------------------------------------

def _fmp_get(url: str, params: dict) -> list:
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json() or []


def _load_parquet_cache(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            path.unlink(missing_ok=True)
    return None


# ---------------------------------------------------------------------------
# Earnings
# ---------------------------------------------------------------------------

def get_earnings(
    ticker: str,
    start: date,
    end: date,
    api_key: Optional[str] = None,
    cache_dir: Path = _CACHE_ROOT,
) -> pd.DataFrame:
    """Return earnings surprise data from FMP for the given ticker and date range."""
    _check_vault(start, end)
    key = api_key or os.environ.get("FMP_API_KEY", "")
    if not key:
        raise EnvironmentError("FMP_API_KEY not set")

    ticker_dir = _ticker_cache_dir(ticker, cache_dir)
    cache_path = ticker_dir / "fmp_earnings.parquet"

    df = _load_parquet_cache(cache_path)
    if df is None:
        data = _fmp_get(f"{FMP_BASE}/earnings", {"symbol": ticker.upper(), "apikey": key, "limit": 40})
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df.to_parquet(cache_path, index=False)
        logger.debug("Cached earnings for %s (%d rows)", ticker, len(df))

    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analyst estimates
# ---------------------------------------------------------------------------

def get_estimates(
    ticker: str,
    start: date,
    end: date,
    api_key: Optional[str] = None,
    cache_dir: Path = _CACHE_ROOT,
) -> pd.DataFrame:
    """Return quarterly analyst consensus EPS and revenue estimates from FMP."""
    _check_vault(start, end)
    key = api_key or os.environ.get("FMP_API_KEY", "")
    if not key:
        raise EnvironmentError("FMP_API_KEY not set")

    ticker_dir = _ticker_cache_dir(ticker, cache_dir)
    cache_path = ticker_dir / "fmp_estimates.parquet"

    df = _load_parquet_cache(cache_path)
    if df is None or "epsAvg" not in df.columns:
        # Re-fetch if cache is missing or predates the column-name fix
        if cache_path.exists():
            cache_path.unlink()
        data = _fmp_get(
            f"{FMP_BASE}/analyst-estimates",
            {"symbol": ticker.upper(), "period": "quarterly", "limit": 40, "apikey": key},
        )
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "date" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        keep_cols = [c for c in ["date", "epsAvg", "revenueAvg"] if c in df.columns]
        df = df[keep_cols]
        df.to_parquet(cache_path, index=False)
        logger.debug("Cached estimates for %s (%d rows)", ticker, len(df))

    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Valuation ratios (PE, PB, PS, EV/EBITDA)
# ---------------------------------------------------------------------------

def get_valuation_ratios(
    ticker: str,
    start: date,
    end: date,
    api_key: Optional[str] = None,
    cache_dir: Path = _CACHE_ROOT,
) -> pd.DataFrame:
    """Return quarterly valuation ratios (pe_ratio, pb_ratio, ps_ratio, ev_ebitda) from FMP."""
    _check_vault(start, end)
    key = api_key or os.environ.get("FMP_API_KEY", "")
    if not key:
        raise EnvironmentError("FMP_API_KEY not set")

    ticker_dir = _ticker_cache_dir(ticker, cache_dir)
    cache_path = ticker_dir / "fmp_valuation.parquet"

    df = _load_parquet_cache(cache_path)
    if df is None:
        # /ratios has pe, pb, ps; /key-metrics has evToEBITDA
        params = {"symbol": ticker.upper(), "period": "quarter", "limit": 40, "apikey": key}
        ratios_data = _fmp_get(f"{FMP_BASE}/ratios", params)
        km_data = _fmp_get(f"{FMP_BASE}/key-metrics", params)
        if not ratios_data:
            return pd.DataFrame()
        ratios_df = pd.DataFrame(ratios_data)[["date", "priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio"]]
        ratios_df.columns = ["date", "pe_ratio", "pb_ratio", "ps_ratio"]
        if km_data:
            km_df = pd.DataFrame(km_data)[["date", "evToEBITDA"]].rename(columns={"evToEBITDA": "ev_ebitda"})
            df = ratios_df.merge(km_df, on="date", how="left")
        else:
            df = ratios_df
            df["ev_ebitda"] = float("nan")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df.to_parquet(cache_path, index=False)
        logger.debug("Cached valuation ratios for %s (%d rows)", ticker, len(df))

    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def get_fundamentals(
    ticker: str,
    start: date,
    end: date,
    api_key: Optional[str] = None,
    cache_dir: Path = _CACHE_ROOT,
) -> pd.DataFrame:
    """Return quarterly fundamental data (ROE, leverage, net_margin) from FMP."""
    _check_vault(start, end)
    key = api_key or os.environ.get("FMP_API_KEY", "")
    if not key:
        raise EnvironmentError("FMP_API_KEY not set")

    ticker_dir = _ticker_cache_dir(ticker, cache_dir)
    cache_path = ticker_dir / "fmp_fundamentals.parquet"

    df = _load_parquet_cache(cache_path)
    if df is None:
        params = {"symbol": ticker.upper(), "period": "quarter", "limit": 40, "apikey": key}
        income = _fmp_get(f"{FMP_BASE}/income-statement", params)
        balance = _fmp_get(f"{FMP_BASE}/balance-sheet-statement", params)
        if not income or not balance:
            return pd.DataFrame()

        inc_df = pd.DataFrame(income)[["date", "netIncome", "revenue", "eps"]]
        bal_df = pd.DataFrame(balance)[["date", "totalStockholdersEquity", "netDebt"]]
        df = inc_df.merge(bal_df, on="date", how="inner")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df.to_parquet(cache_path, index=False)
        logger.debug("Cached fundamentals for %s (%d rows)", ticker, len(df))

    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()
    equity = df["totalStockholdersEquity"].replace(0, float("nan"))
    df["roe"] = df["netIncome"] / equity
    revenue = df["revenue"].replace(0, float("nan"))
    df["net_margin"] = df["netIncome"] / revenue
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------

def get_transcripts(
    ticker: str,
    year: int,
    quarter: int,
    api_key: Optional[str] = None,
    cache_dir: Path = _CACHE_ROOT,
) -> str:
    """Return earnings call transcript text from FMP."""
    _quarter_end = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    month, day = _quarter_end[quarter]
    _check_vault(date(year, month, day), date(year, month, day))

    key = api_key or os.environ.get("FMP_API_KEY", "")
    if not key:
        raise EnvironmentError("FMP_API_KEY not set")

    ticker_dir = _ticker_cache_dir(ticker, cache_dir)
    cache_path = ticker_dir / f"fmp_transcript_{year}q{quarter}.txt"

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    data = _fmp_get(
        f"{FMP_BASE}/earning_call_transcript",
        {"symbol": ticker.upper(), "year": year, "quarter": quarter, "apikey": key},
    )
    text = data[0].get("content", "") if data else ""
    cache_path.write_text(text, encoding="utf-8")
    logger.debug("Cached transcript for %s %dQ%d (%d chars)", ticker, year, quarter, len(text))
    return text
