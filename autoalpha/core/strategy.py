"""Abstract Strategy base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class Strategy(ABC):
    """A pure-function trading strategy.

    fit() is called once per CPCV fold on in-sample data.
    predict() is called on every bar and returns target portfolio fractions.
    """

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Train on in-sample MultiIndex(date, ticker) bar data.

        data columns: Open, High, Low, Close, Volume, plus any features.
        Event-driven strategies may implement this as a no-op.
        """

    @abstractmethod
    def predict(self, bar_data: pd.DataFrame, bar_date: Optional[pd.Timestamp] = None) -> dict[str, float]:
        """Return target position fractions for the current bar.

        bar_data: DataFrame with index=ticker, columns=OHLCV+features (same columns as fit()).
                  One row per ticker in the universe.
        bar_date: timestamp of the current bar (passed by Runner). Strategies that need
                  to detect calendar events (month-end, earnings date) should use this.
        Returns: {ticker: fraction} where 0.02 = 2% long, -0.01 = 1% short.
                 Absent tickers → flat (no position change required by caller).
        Non-event bars must return {}.
        """
