"""12-1 month cross-sectional momentum strategy.

Signal: cumulative return from t-252 to t-21 trading bars.
  - Skipping the most recent 21 bars avoids the 1-month short-term reversal effect.
  - Documented in Jegadeesh & Titman (1993) and confirmed across 150yr / 46 countries.

Universe: long top quintile by momentum signal, equal-weight.
Rebalance: first trading day of each calendar month.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from autoalpha.core.strategy import Strategy

logger = logging.getLogger(__name__)

_LOOKBACK = 252   # ~12 months in trading days
_SKIP = 21        # ~1 month skip to avoid reversal
_QUANTILE = 0.80  # top 20% = top quintile


class MomentumStrategy(Strategy):
    """12-1 month cross-sectional momentum, monthly rebalance."""

    def __init__(
        self,
        lookback: int = _LOOKBACK,
        skip: int = _SKIP,
        quantile: float = _QUANTILE,
    ):
        self._lookback = lookback
        self._skip = skip
        self._quantile = quantile
        self._price_history: pd.DataFrame = pd.DataFrame()
        self._prev_month: Optional[int] = None
        self._last_targets: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> None:
        """Seed the rolling price buffer from in-sample data.

        Keeps the last `lookback + skip` bars so momentum is computable
        from the very first OOS bar.
        """
        self._prev_month = None
        self._last_targets = {}

        if data.empty or "Close" not in data.columns:
            self._price_history = pd.DataFrame()
            return

        close = data["Close"].unstack(level="ticker")  # date × ticker
        keep = self._lookback + self._skip
        self._price_history = close.iloc[-keep:].copy() if len(close) >= keep else close.copy()

    def predict(
        self,
        bar_data: pd.DataFrame,
        bar_date: Optional[pd.Timestamp] = None,
    ) -> dict[str, float]:
        """Return long positions for top-quintile tickers on rebalance days.

        Non-rebalance bars return the previous targets unchanged (hold positions).
        """
        if "Close" not in bar_data.columns or bar_date is None:
            return self._last_targets

        # Append current bar to rolling price buffer
        close_row = bar_data["Close"]
        new_row = pd.DataFrame([close_row.to_dict()], index=[bar_date])
        new_row.index.name = "date"

        if self._price_history.empty:
            self._price_history = new_row
        else:
            self._price_history = pd.concat([self._price_history, new_row])
            # Trim to avoid unbounded growth
            max_rows = self._lookback + self._skip + 10
            if len(self._price_history) > max_rows:
                self._price_history = self._price_history.iloc[-max_rows:]

        # Rebalance on first trading day of each month (month change detected by bar_date)
        current_month = bar_date.month
        if current_month == self._prev_month:
            return self._last_targets

        self._prev_month = current_month
        self._last_targets = self._compute_targets()
        logger.debug(
            "Momentum rebalance on %s: %d positions", bar_date.date(), len(self._last_targets)
        )
        return self._last_targets

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_targets(self) -> dict[str, float]:
        """Compute cross-sectional momentum signal and return top-quintile targets."""
        prices = self._price_history

        min_rows = self._lookback + self._skip
        if len(prices) < min_rows:
            logger.debug(
                "Insufficient history for momentum: %d rows, need %d", len(prices), min_rows
            )
            return {}

        # Signal: (price at t-skip) / (price at t-lookback) - 1
        # iloc[-skip] = t-21 (most recent included); iloc[-lookback-skip] uses all available
        end_prices = prices.iloc[-self._skip]           # t-21
        start_prices = prices.iloc[-(self._lookback + self._skip)]  # t-252 from end_price

        signal = (end_prices - start_prices) / start_prices.replace(0, np.nan)
        signal = signal.dropna()

        if len(signal) < 5:
            return {}

        threshold = signal.quantile(self._quantile)
        top = signal[signal >= threshold].index.tolist()
        if not top:
            return {}

        weight = 1.0 / len(top)
        return {ticker: weight for ticker in top}
