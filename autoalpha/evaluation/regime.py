"""Regime-conditional performance breakdown.

Trend regime (SPY 63-day return):
  bull     > +5%
  bear     < -5%
  sideways otherwise

Vol regime (SPY 21-day realized annualized vol):
  high  > 20%
  low   otherwise
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from autoalpha.evaluation.sharpe import annualized_sharpe

logger = logging.getLogger(__name__)

_TREND_WINDOW = 63
_BULL_THRESH = 0.05
_BEAR_THRESH = -0.05
_VOL_WINDOW = 21
_HIGH_VOL_THRESH = 0.20


def _fetch_spy(start: date, end: date) -> pd.Series:
    try:
        tk = yf.Ticker("SPY")
        df = tk.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,
        )
        if df.empty:
            return pd.Series(dtype=float)
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        s.index.name = "date"
        return s
    except Exception as exc:
        logger.warning("Failed to fetch SPY: %s", exc)
        return pd.Series(dtype=float)


def classify_trend(spy_close: pd.Series) -> pd.Series:
    """Return daily trend regime: 'bull', 'bear', or 'sideways'. NaN during warm-up."""
    ret = spy_close.pct_change(_TREND_WINDOW)
    regime = pd.Series(None, index=ret.index, dtype=object)
    valid = ~ret.isna()
    regime[valid & (ret > _BULL_THRESH)] = "bull"
    regime[valid & (ret < _BEAR_THRESH)] = "bear"
    regime[valid & (ret >= _BEAR_THRESH) & (ret <= _BULL_THRESH)] = "sideways"
    return regime


def classify_vol(spy_close: pd.Series) -> pd.Series:
    """Return daily vol regime: 'high' or 'low'. NaN during warm-up."""
    daily_ret = spy_close.pct_change()
    realized_vol = daily_ret.rolling(_VOL_WINDOW).std() * (252**0.5)
    regime = pd.Series(None, index=realized_vol.index, dtype=object)
    valid = ~realized_vol.isna()
    regime[valid & (realized_vol > _HIGH_VOL_THRESH)] = "high"
    regime[valid & (realized_vol <= _HIGH_VOL_THRESH)] = "low"
    return regime


def regime_breakdown(
    returns: pd.Series,
    spy_close: Optional[pd.Series] = None,
) -> dict[str, dict[str, dict]]:
    """Compute performance breakdown by trend and vol regime.

    Returns:
        {
          "trend": {"bull": {"sharpe": ..., "n_days": ..., "mean_return": ...}, ...},
          "vol":   {"high": {...}, "low": {...}},
        }
    """
    if spy_close is None:
        # Fetch extra lookback for the rolling windows
        start = returns.index.min().date() - timedelta(days=_TREND_WINDOW * 2)
        end = returns.index.max().date()
        spy_close = _fetch_spy(start, end)

    if spy_close.empty:
        logger.warning("SPY data unavailable — skipping regime breakdown")
        return {}

    trend_regime = classify_trend(spy_close)
    vol_regime = classify_vol(spy_close)

    result: dict[str, dict] = {"trend": {}, "vol": {}}

    for label, regime_series in [("trend", trend_regime), ("vol", vol_regime)]:
        for regime_name in ["bull", "bear", "sideways"] if label == "trend" else ["high", "low"]:
            mask_dates = regime_series[regime_series == regime_name].index
            common = returns.index.intersection(mask_dates)
            sub = returns.loc[common]
            if sub.empty:
                continue
            result[label][regime_name] = {
                "sharpe": annualized_sharpe(sub),
                "n_days": len(sub),
                "mean_return": float(sub.mean()),
            }

    return result
