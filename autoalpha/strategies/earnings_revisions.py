"""Earnings acceleration strategy (EPS consensus YoY comparison).

Signal: year-over-year change in analyst consensus EPS estimate for the same
fiscal quarter (e.g., Q2 2024 estimate vs Q2 2023 estimate). A meaningful
upward shift signals accelerating earnings expectations for that quarter.

Using same-quarter YoY rather than sequential quarterly shift(1) avoids
comparing different fiscal periods (Q1 vs Q2) and avoids seasonal distortions
(e.g., retail Q4 is always higher than Q1 regardless of revision activity).

Entry: next bar's open (next-bar fill model).
Hold:  HOLD_BARS trading days.

Returns {} on non-revision bars.
"""
from __future__ import annotations

import logging
import os
from datetime import date
from typing import Optional

import pandas as pd

from autoalpha.core.strategy import Strategy
from autoalpha.data.fetcher import get_estimates, VaultLeakError

logger = logging.getLogger(__name__)

POSITION_SIZE = 0.02
REVISION_THRESHOLD = 0.05   # 5% upward revision required
HOLD_BARS = 21              # ~1 month hold


class EarningsRevisionsStrategy(Strategy):
    """Long when analyst EPS estimates are revised upward significantly."""

    def __init__(
        self,
        position_size: float = POSITION_SIZE,
        revision_threshold: float = REVISION_THRESHOLD,
        hold_bars: int = HOLD_BARS,
        fmp_api_key: Optional[str] = None,
    ):
        self._position_size = position_size
        self._revision_threshold = revision_threshold
        self._hold_bars = hold_bars
        self._api_key = fmp_api_key or os.environ.get("FMP_API_KEY", "")

        # {ticker: set of pd.Timestamp} — dates of significant upward revisions
        self._revision_dates: dict[str, set] = {}
        # {ticker: int} — bars held since entry
        self._holdings: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> None:
        """Identify upward EPS revision dates for all in-sample tickers."""
        self._revision_dates = {}
        self._holdings = {}

        if data.empty:
            return

        tickers = data.index.get_level_values("ticker").unique().tolist()
        dates = data.index.get_level_values("date")
        start = dates.min().date()
        end = dates.max().date()

        if not self._api_key:
            logger.warning("FMP_API_KEY not set — EarningsRevisions will produce no signals")
            return

        for ticker in tickers:
            try:
                rev_dates = self._find_revision_dates(ticker, start, end)
                if rev_dates:
                    self._revision_dates[ticker] = rev_dates
            except VaultLeakError:
                raise
            except Exception as exc:
                logger.warning("Failed to fetch estimates for %s: %s", ticker, exc)

    def predict(
        self,
        bar_data: pd.DataFrame,
        bar_date: Optional[pd.Timestamp] = None,
    ) -> dict[str, float]:
        """Signal entry on revision dates; exit after hold_bars."""
        result: dict[str, float] = {}

        if bar_date is None or bar_data.empty:
            return result

        # Manage existing holdings
        to_close: list[str] = []
        for ticker, bars_held in self._holdings.items():
            self._holdings[ticker] = bars_held + 1
            if self._holdings[ticker] >= self._hold_bars:
                result[ticker] = 0.0
                to_close.append(ticker)
        for t in to_close:
            del self._holdings[t]

        # Detect new revision entries
        for ticker in bar_data.index:
            if ticker in self._holdings:
                continue
            rev_dates = self._revision_dates.get(ticker, set())
            if bar_date in rev_dates:
                result[ticker] = self._position_size
                self._holdings[ticker] = 0

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_revision_dates(self, ticker: str, start: date, end: date) -> set:
        """Return dates where EPS estimate was revised upward by > threshold."""
        df = get_estimates(ticker, start, end, api_key=self._api_key)
        if df.empty or "estimatedEpsAvg" not in df.columns:
            return set()

        df = df.sort_values("date").reset_index(drop=True)

        # Compare same fiscal quarter year-over-year (Q2 2024 vs Q2 2023) to
        # avoid mixing different seasonal periods with a simple shift(1).
        df["fiscal_quarter"] = df["date"].dt.quarter
        df["eps_prev_yoy"] = df.groupby("fiscal_quarter")["estimatedEpsAvg"].shift(1)

        revision_dates: set = set()
        for _, row in df.iterrows():
            curr = row.get("estimatedEpsAvg")
            prev = row.get("eps_prev_yoy")
            if pd.isna(curr) or pd.isna(prev) or prev == 0:
                continue
            change = (float(curr) - float(prev)) / abs(float(prev))
            if change >= self._revision_threshold:
                revision_dates.add(pd.Timestamp(row["date"]))

        return revision_dates
