"""Dollar bar constructor.

Dollar bar threshold = ADTV / 50 where ADTV = 63-day trailing average daily dollar volume,
recomputed monthly per ticker. Targets ~50 bars per trading day.

Requires Polygon.io historical minute data for real dollar bars.
Falls back to daily bars if Polygon is unavailable, logging a DataQualityWarning
with ADF p-value and lag-1 autocorrelation so degradation is visible.
"""
from __future__ import annotations

import logging
import os
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


class DataQualityWarning(UserWarning):
    pass


POLYGON_BASE = "https://api.polygon.io/v2"


def _adf_stats(series: pd.Series) -> tuple[float, float]:
    """Return (adf_pvalue, lag1_autocorr)."""
    try:
        result = adfuller(series.dropna(), autolag="AIC")
        pval = float(result[1])
    except Exception:
        pval = 1.0
    autocorr = float(series.autocorr(lag=1)) if len(series) > 2 else float("nan")
    return pval, autocorr


def _fetch_minute_polygon(
    ticker: str, start: date, end: date, api_key: str
) -> pd.DataFrame:
    """Fetch 1-minute OHLCV bars from Polygon.io."""
    url = f"{POLYGON_BASE}/aggs/ticker/{ticker}/range/1/minute/{start.isoformat()}/{end.isoformat()}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apikey": api_key}
    frames = []
    while url:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if results:
            df = pd.DataFrame(results)
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
            df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
            df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
            frames.append(df)
        url = data.get("next_url")
        params = {"apikey": api_key} if url else {}
    if not frames:
        return pd.DataFrame(columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    return pd.concat(frames, ignore_index=True)


def _compute_adtv(daily_df: pd.DataFrame, lookback: int = 63) -> float:
    """Compute average daily trading volume (dollar) over the last lookback days."""
    df = daily_df.tail(lookback)
    dollar_vol = df["Close"] * df["Volume"]
    return float(dollar_vol.mean()) if not dollar_vol.empty else 0.0


def build_dollar_bars(
    ticker: str,
    start: date,
    end: date,
    daily_df: pd.DataFrame,
    polygon_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Build dollar bars for a ticker between start and end.

    daily_df: pre-fetched daily OHLCV used to compute ADTV threshold.
    Returns DataFrame with columns: timestamp, Open, High, Low, Close, Volume, dollar_volume.
    """
    key = polygon_api_key or os.environ.get("POLYGON_API_KEY", "")

    if not key:
        warnings.warn(
            f"POLYGON_API_KEY not set for {ticker} — falling back to daily bars",
            DataQualityWarning,
            stacklevel=2,
        )
        return _daily_fallback(ticker, daily_df, start, end)

    try:
        minute_df = _fetch_minute_polygon(ticker, start, end, key)
    except Exception as exc:
        warnings.warn(
            f"Polygon fetch failed for {ticker} ({exc}) — falling back to daily bars",
            DataQualityWarning,
            stacklevel=2,
        )
        return _daily_fallback(ticker, daily_df, start, end)

    if minute_df.empty:
        warnings.warn(
            f"No Polygon data for {ticker} — falling back to daily bars",
            DataQualityWarning,
            stacklevel=2,
        )
        return _daily_fallback(ticker, daily_df, start, end)

    adtv = _compute_adtv(daily_df)
    if adtv <= 0:
        warnings.warn(
            f"Zero ADTV for {ticker} — cannot compute dollar bar threshold",
            DataQualityWarning,
            stacklevel=2,
        )
        return _daily_fallback(ticker, daily_df, start, end)

    threshold = adtv / 50.0
    return _aggregate_dollar_bars(minute_df, threshold)


def _aggregate_dollar_bars(minute_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Aggregate 1-minute bars into dollar bars given a dollar threshold."""
    records = []
    cum_dollar = 0.0
    bar_open = None
    bar_high = -np.inf
    bar_low = np.inf
    bar_close = None
    bar_volume = 0.0
    bar_start_ts = None

    for _, row in minute_df.iterrows():
        price = float(row["Close"])
        vol = float(row["Volume"])
        dollar = price * vol

        if bar_open is None:
            bar_open = float(row["Open"])
            bar_start_ts = row["timestamp"]

        bar_high = max(bar_high, float(row["High"]))
        bar_low = min(bar_low, float(row["Low"]))
        bar_close = price
        bar_volume += vol
        cum_dollar += dollar

        if cum_dollar >= threshold:
            records.append({
                "timestamp": bar_start_ts,
                "Open": bar_open,
                "High": bar_high,
                "Low": bar_low,
                "Close": bar_close,
                "Volume": bar_volume,
                "dollar_volume": cum_dollar,
            })
            cum_dollar = 0.0
            bar_open = None
            bar_high = -np.inf
            bar_low = np.inf
            bar_close = None
            bar_volume = 0.0
            bar_start_ts = None

    return pd.DataFrame(records)


def _daily_fallback(
    ticker: str, daily_df: pd.DataFrame, start: date, end: date
) -> pd.DataFrame:
    """Return daily bars as pseudo dollar bars, with IID quality stats logged."""
    mask = (daily_df.index >= pd.Timestamp(start)) & (daily_df.index <= pd.Timestamp(end))
    df = daily_df[mask].copy()
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "dollar_volume"])

    df["dollar_volume"] = df["Close"] * df["Volume"]
    df = df.rename_axis("timestamp").reset_index()

    returns = df["Close"].pct_change().dropna()
    adf_pval, lag1_autocorr = _adf_stats(returns)
    warnings.warn(
        f"[{ticker}] Daily bar fallback: ADF p={adf_pval:.4f} (threshold 0.05), "
        f"lag-1 autocorr={lag1_autocorr:.4f}",
        DataQualityWarning,
        stacklevel=3,
    )
    return df[["timestamp", "Open", "High", "Low", "Close", "Volume", "dollar_volume"]]
