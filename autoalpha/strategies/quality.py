"""Quality factor strategy.

Signal (cross-sectional z-scores, each quarterly):
  quality = z(ROE) - z(leverage) + z(net_margin)

  leverage = net_debt / (net_debt + market_cap)

Long top quintile by quality score; equal-weight.
Rebalance: quarterly — first trading day where the calendar quarter changes.

Returns {} on non-rebalance bars (positions held by executor).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from autoalpha.core.strategy import Strategy
from autoalpha.data.fetcher import get_fundamentals, VaultLeakError

logger = logging.getLogger(__name__)

_QUANTILE = 0.80   # top 20% = top quintile


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score, ignoring NaN."""
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


class QualityStrategy(Strategy):
    """Quality factor (ROE - leverage + net_margin), quarterly rebalance."""

    def __init__(
        self,
        quantile: float = _QUANTILE,
        fmp_api_key: Optional[str] = None,
    ):
        self._quantile = quantile
        self._api_key = fmp_api_key or os.environ.get("FMP_API_KEY", "")

        # date → {ticker: weight} pre-computed for each quarter in-sample
        self._quarterly_targets: dict[pd.Timestamp, dict[str, float]] = {}
        # Most recently applied q_date (after reporting lag); rebalance whenever a newer
        # q_date crosses bar_date — avoids the mismatch between calendar-quarter triggers
        # and the lagged availability date.
        self._last_applied_q_date: Optional[pd.Timestamp] = None
        self._last_targets: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> None:
        """Compute quality targets for every quarter in the in-sample window."""
        self._quarterly_targets = {}
        self._last_applied_q_date = None
        self._last_targets = {}

        if data.empty:
            return

        tickers = data.index.get_level_values("ticker").unique().tolist()
        dates = data.index.get_level_values("date")
        start = dates.min().date()
        end = dates.max().date()

        if not self._api_key:
            logger.warning("FMP_API_KEY not set — Quality strategy will produce no signals")
            return

        # Fetch fundamentals for all tickers, group by quarter
        fund_rows: list[dict] = []
        for ticker in tickers:
            try:
                df = get_fundamentals(ticker, start, end, api_key=self._api_key)
                if df.empty:
                    continue
                # Get most-recent-available market cap from OHLCV (Close × Volume proxy)
                # Full market cap not available from fundamentals alone — use identity weight
                for _, row in df.iterrows():
                    fund_rows.append({
                        "ticker": ticker,
                        "date": row["date"],
                        "roe": row.get("roe", float("nan")),
                        "net_margin": row.get("net_margin", float("nan")),
                        "net_debt": row.get("netDebt", float("nan")),
                    })
            except VaultLeakError:
                raise
            except Exception as exc:
                logger.warning("Failed to fetch fundamentals for %s: %s", ticker, exc)

        if not fund_rows:
            return

        fund_df = pd.DataFrame(fund_rows)
        fund_df["quarter"] = fund_df["date"].dt.to_period("Q")

        for quarter, grp in fund_df.groupby("quarter"):
            # Use most recent row per ticker for this quarter
            latest = grp.sort_values("date").groupby("ticker").last().reset_index()
            latest = latest.set_index("ticker")

            roe = latest["roe"].astype(float)
            net_margin = latest["net_margin"].astype(float)
            # leverage: net_debt / (net_debt + 1) as a unit-free proxy (no market cap)
            net_debt = latest["net_debt"].astype(float)
            leverage = net_debt / (net_debt.abs() + 1e9)  # normalised by 1B as denominator proxy

            quality = _zscore(roe) - _zscore(leverage) + _zscore(net_margin)
            quality = quality.dropna()
            if len(quality) < 5:
                continue

            threshold = quality.quantile(self._quantile)
            top = quality[quality >= threshold].index.tolist()
            if not top:
                continue

            weight = 1.0 / len(top)
            # Quarterly reports are typically available ~45 calendar days after quarter-end.
            # Apply a fixed reporting lag to avoid look-ahead bias.
            q_end = pd.Timestamp(str(quarter.end_time.date()))
            q_date = q_end + pd.DateOffset(days=45)
            self._quarterly_targets[q_date] = {t: weight for t in top}

    def predict(
        self,
        bar_data: pd.DataFrame,
        bar_date: Optional[pd.Timestamp] = None,
    ) -> dict[str, float]:
        """Return updated quality targets whenever new quarterly data becomes available.

        Rebalance trigger: the first bar after a new q_date (quarter-end + 45d lag)
        crosses bar_date.  This ensures the reporting lag is honoured and the data
        is applied as soon as it becomes available — not delayed to the next arbitrary
        calendar-quarter boundary.
        """
        if bar_date is None:
            active = set(bar_data.index)
            return {t: w for t, w in self._last_targets.items() if t in active}

        # Find most recent q_date at or before bar_date
        applicable = {k: v for k, v in self._quarterly_targets.items() if k <= bar_date}
        if applicable:
            latest_key = max(applicable)
            if latest_key != self._last_applied_q_date:
                self._last_targets = applicable[latest_key]
                self._last_applied_q_date = latest_key
                logger.debug(
                    "Quality rebalance on %s using data from %s: %d positions",
                    bar_date.date(), latest_key.date(), len(self._last_targets),
                )

        active = set(bar_data.index)
        return {t: w for t, w in self._last_targets.items() if t in active}
