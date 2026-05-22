"""Transaction cost model.

Default round-trip cost breakdown:
  half-spread   2.0 bps
  commission    0.5 bps
  market impact 3.0 bps
  ─────────────────────
  one-way       5.5 bps  → ~11 bps round-trip
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_HALF_SPREAD_BPS: float = 2.0
DEFAULT_COMMISSION_BPS: float = 0.5
DEFAULT_MARKET_IMPACT_BPS: float = 3.0
DEFAULT_ROUND_TRIP_BPS: float = 11.0


@dataclass
class CostModel:
    half_spread_bps: float = DEFAULT_HALF_SPREAD_BPS
    commission_bps: float = DEFAULT_COMMISSION_BPS
    market_impact_bps: float = DEFAULT_MARKET_IMPACT_BPS

    @property
    def one_way_bps(self) -> float:
        return self.half_spread_bps + self.commission_bps + self.market_impact_bps

    @property
    def round_trip_bps(self) -> float:
        return self.one_way_bps * 2

    def deduct(self, gross_returns: pd.Series, turnover: pd.Series) -> pd.Series:
        """Return net returns after transaction costs.

        turnover: daily fraction of portfolio traded (one-way).
        cost_per_day = turnover * round_trip_bps / 10_000
        """
        cost_per_day = turnover * (self.round_trip_bps / 10_000)
        return gross_returns - cost_per_day.reindex(gross_returns.index, fill_value=0.0)

    def stress(self, multiplier: float = 2.0) -> "CostModel":
        """Return a copy with all cost parameters scaled by multiplier."""
        return CostModel(
            half_spread_bps=self.half_spread_bps * multiplier,
            commission_bps=self.commission_bps * multiplier,
            market_impact_bps=self.market_impact_bps * multiplier,
        )


DEFAULT_COST_MODEL = CostModel()
