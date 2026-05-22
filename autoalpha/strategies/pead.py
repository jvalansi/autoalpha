"""Post-Earnings Announcement Drift (PEAD) strategy.

Entry conditions (all required):
  1. EPS actual > EPS estimate (earnings beat)
  2. Revenue actual > Revenue estimate (revenue beat)
  3. After-hours proxy: bar close ≥ 1% above prior bar close

Entry: next bar's open (via Runner's next-bar fill model).
Hold:  10 trading days, then exit.

Returns {} on non-event bars; hold is maintained implicitly by the
executor's "absent ticker = no change" behavior, with explicit exit
targets on day 10.
"""
from __future__ import annotations

import logging
import os
from datetime import date
from typing import Optional

import pandas as pd

from autoalpha.core.strategy import Strategy
from autoalpha.data.fetcher import get_earnings, VaultLeakError

logger = logging.getLogger(__name__)

POSITION_SIZE = 0.02       # 2% of portfolio per event
AH_THRESHOLD = 0.01        # 1% close-over-prev-close for AH proxy
HOLD_BARS = 10             # trading days to hold after entry


class PEADStrategy(Strategy):
    """Post-Earnings Announcement Drift."""

    def __init__(
        self,
        position_size: float = POSITION_SIZE,
        ah_threshold: float = AH_THRESHOLD,
        hold_bars: int = HOLD_BARS,
        fmp_api_key: Optional[str] = None,
    ):
        self._position_size = position_size
        self._ah_threshold = ah_threshold
        self._hold_bars = hold_bars
        self._api_key = fmp_api_key or os.environ.get("FMP_API_KEY", "")

        # {ticker: set of pd.Timestamp} — in-sample earnings beat dates
        self._beat_dates: dict[str, set] = {}
        # {ticker: int} — bars held since entry
        self._holdings: dict[str, int] = {}
        # {ticker: float} — previous bar's close price
        self._prev_closes: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> None:
        """Pre-load earnings beat dates for all tickers in the in-sample window."""
        self._beat_dates = {}
        self._holdings = {}
        self._prev_closes = {}

        if data.empty:
            return

        tickers = data.index.get_level_values("ticker").unique().tolist()
        dates = data.index.get_level_values("date")
        start = dates.min().date()
        end = dates.max().date()

        if not self._api_key:
            logger.warning("FMP_API_KEY not set — PEAD strategy will produce no signals")
            return

        for ticker in tickers:
            try:
                beats = self._fetch_beats(ticker, start, end)
                if beats:
                    self._beat_dates[ticker] = beats
            except VaultLeakError:
                raise
            except Exception as exc:
                logger.warning("Failed to fetch earnings for %s: %s", ticker, exc)

    def predict(
        self,
        bar_data: pd.DataFrame,
        bar_date: Optional[pd.Timestamp] = None,
    ) -> dict[str, float]:
        """Signal PEAD entry on earnings beat bars; exit after hold_bars."""
        result: dict[str, float] = {}

        if bar_date is None or bar_data.empty:
            return result

        # Step 1: manage existing holdings (increment counter; exit on expiry)
        to_close: list[str] = []
        for ticker, bars_held in self._holdings.items():
            self._holdings[ticker] = bars_held + 1
            if self._holdings[ticker] >= self._hold_bars:
                result[ticker] = 0.0
                to_close.append(ticker)
        for t in to_close:
            del self._holdings[t]

        # Step 2: detect new entries
        if "Close" in bar_data.columns:
            for ticker in bar_data.index:
                if ticker in self._holdings:
                    continue  # already holding
                beats = self._beat_dates.get(ticker, set())
                if bar_date not in beats:
                    continue

                # AH proxy: current close ≥ (1 + threshold) × prev close
                close = float(bar_data.loc[ticker, "Close"])
                prev_close = self._prev_closes.get(ticker, 0.0)
                if prev_close > 0 and close / prev_close >= (1 + self._ah_threshold):
                    result[ticker] = self._position_size
                    self._holdings[ticker] = 0

        # Step 3: update prev closes
        if "Close" in bar_data.columns:
            for ticker in bar_data.index:
                self._prev_closes[ticker] = float(bar_data.loc[ticker, "Close"])

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_beats(self, ticker: str, start: date, end: date) -> set:
        """Return set of timestamps where ticker beat both EPS and revenue."""
        df = get_earnings(ticker, start, end, api_key=self._api_key)
        if df.empty:
            return set()

        beats: set = set()
        for _, row in df.iterrows():
            eps_ok = (
                pd.notna(row.get("epsActual")) and pd.notna(row.get("epsEstimated"))
                and float(row["epsActual"]) > float(row["epsEstimated"])
            )
            # FMP field names vary; try both conventions
            rev_actual = row.get("revenueActual") or row.get("revenue")
            rev_est = row.get("revenueEstimated") or row.get("revenueEstimate")
            rev_ok = (
                pd.notna(rev_actual) and pd.notna(rev_est)
                and float(rev_actual) > float(rev_est)
            ) if rev_actual is not None and rev_est is not None else False

            if eps_ok and rev_ok:
                beats.add(pd.Timestamp(row["date"]))

        return beats
