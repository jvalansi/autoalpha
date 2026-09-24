"""earnings-trader's PEAD rules, expressed on autoalpha's dataset columns.

Mirrors earnings-trader/src/decision.py + config.py so the strategy can be run
through autoalpha's holdout/forward promotion gate:

  Entry (on the earnings reaction bar, days_since_earnings == 0):
    EPS surprise >= 5%, revenue surprise > 0, overnight gap >= 3%,
    9-day run-up before the reaction day <= 10%, sector ETF move > -1.5% that day
    (earnings-trader's live definition), fewer than 10 open positions.
  Exit: 2.5 x ATR(14, Wilder) trailing stop on the close, or after 10 bars.
  Sizing: 10% of the book per position (10 slots), the rest in cash.

Differences that remain: autoalpha fills at the next bar's open, so entries
land one bar after the reaction day (earnings-trader buys near its close), and
stops/exits also fill at the next open. The guidance filter is not ported —
FMP no longer returns guidanceEps, so it always passes in earnings-trader too.
"""
from __future__ import annotations

import pandas as pd

from autoalpha.core.strategy import Strategy

MIN_EPS_BEAT_PCT = 0.05
MIN_GAP_PCT = 0.03
MAX_PRIOR_RUNUP_PCT = 0.10
SECTOR_ETF_MIN = -0.015
ATR_STOP_MULTIPLIER = 2.5
ATR_PERIOD = 14
HOLD_DAYS = 10
MAX_POSITIONS = 10


class EarningsTraderPEAD(Strategy):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._positions: dict[str, dict] = {}   # ticker -> {"stop": float, "days": int}
        self._atr = pd.Series(dtype=float)
        self._prev_close = pd.Series(dtype=float)
        self._last_date: pd.Timestamp | None = None

    def fit(self, data: pd.DataFrame) -> None:
        self.reset()

    def _update_atr(self, bars: pd.DataFrame) -> None:
        pc = self._prev_close.reindex(bars.index)
        tr = pd.concat([bars["High"] - bars["Low"], (bars["High"] - pc).abs(), (bars["Low"] - pc).abs()],
                       axis=1).max(axis=1)
        prev = self._atr.reindex(bars.index)
        # Wilder smoothing (ewm alpha=1/period, seeded with the first true range)
        self._atr = (prev + (tr - prev) / ATR_PERIOD).fillna(tr).combine_first(self._atr)
        self._prev_close = bars["Close"].combine_first(self._prev_close)

    def predict(self, bar_data: pd.DataFrame, bar_date: pd.Timestamp | None = None) -> dict[str, float]:
        if bar_date is not None and self._last_date is not None and bar_date <= self._last_date:
            self.reset()  # a new backtest fold started earlier in time
        self._last_date = bar_date
        bars = bar_data.dropna(subset=["High", "Low", "Close"])
        self._update_atr(bars)

        # Manage open positions (earnings-trader's evaluate_positions)
        for ticker in list(self._positions):
            pos = self._positions[ticker]
            pos["days"] += 1
            if ticker not in bars.index:
                continue
            price = float(bars.at[ticker, "Close"])
            if price <= pos["stop"] or pos["days"] >= HOLD_DAYS:
                del self._positions[ticker]
                continue
            pos["stop"] = max(pos["stop"], price - ATR_STOP_MULTIPLIER * float(self._atr[ticker]))

        # Scan the reaction bar for entries (earnings-trader's evaluate_entry)
        if len(self._positions) < MAX_POSITIONS and "days_since_earnings" in bars.columns:
            ev = bars[bars["days_since_earnings"] == 0]
            runup = (1 + ev["ret_10d"]) / (1 + ev["ret_1d"]) - 1
            ok = ev[(ev["earnings_surprise"] >= MIN_EPS_BEAT_PCT) & (ev["revenue_surprise"] > 0)
                    & (ev["gap_1d"] >= MIN_GAP_PCT) & (runup <= MAX_PRIOR_RUNUP_PCT)
                    & (ev["sector_ret_1d"] > SECTOR_ETF_MIN)]
            for ticker in ok.index:
                if len(self._positions) >= MAX_POSITIONS:
                    break
                if ticker in self._positions:
                    continue
                price = float(ok.at[ticker, "Close"])
                self._positions[ticker] = {"stop": price - ATR_STOP_MULTIPLIER * float(self._atr[ticker]),
                                           "days": 0}

        return {t: 1.0 / MAX_POSITIONS for t in self._positions}
