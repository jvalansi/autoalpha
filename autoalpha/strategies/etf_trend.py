"""Long-only multi-asset trend following on ETFs (time-series momentum).

Moskowitz, Ooi & Pedersen (2012, JFE); Hurst, Ooi & Pedersen, "A Century of
Evidence on Trend-Following Investing". Long-only because SimExecutor and the
paper account drop negative weights: an asset with a negative trend goes to cash.

  Signal:  12-month (252-bar) total return of each ETF, known at the bar's close.
  Weights: inverse 63-bar volatility, normalized over *all* assets with data, then
           zeroed for assets whose trend is negative — so the book is fully
           invested only when every trend is up, and the rest sits in cash.
  Trading: first bar of each month; the same targets repeat in between, so run
           with SimExecutor(hold_unchanged=True).

Parameters are the published defaults, fixed before any backtest was run.
"""
from __future__ import annotations

import pandas as pd

from autoalpha.core.strategy import Strategy

LOOKBACK = 252
VOL_WINDOW = 63


def add_trend_features(closes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """closes: dates × tickers (adjusted). Features at each date use closes up to it."""
    rets = closes.pct_change(fill_method=None)
    return {
        "trend": closes / closes.shift(LOOKBACK) - 1,
        "vol": rets.rolling(VOL_WINDOW).std() * (252 ** 0.5),
    }


class ETFTrend(Strategy):
    def __init__(self) -> None:
        self._targets: dict[str, float] = {}
        self._month: tuple[int, int] | None = None

    def fit(self, data: pd.DataFrame) -> None:
        self._targets, self._month = {}, None

    def predict(self, bar_data: pd.DataFrame, bar_date: pd.Timestamp | None = None) -> dict[str, float]:
        month = (bar_date.year, bar_date.month) if bar_date is not None else None
        if month is None or month != self._month:
            ok = bar_data.dropna(subset=["trend", "vol"])
            ok = ok[ok["vol"] > 0]
            inv_vol = 1.0 / ok["vol"]
            w = inv_vol / inv_vol.sum()
            self._targets = {t: float(x) for t, x in w[ok["trend"] > 0].items()}
            self._month = month
        return dict(self._targets)
