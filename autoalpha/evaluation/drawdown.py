"""Max drawdown computation and hard constraint check."""
from __future__ import annotations

import pandas as pd


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown from a daily return series.

    Returns a positive fraction (e.g., 0.25 = 25% drawdown).
    Returns 0.0 for empty input.
    """
    if returns.empty:
        return 0.0
    # Prepend NAV=1.0 so initial losses are captured (not just peak-to-trough within series)
    nav = pd.concat([pd.Series([1.0]), (1 + returns).cumprod()])
    rolling_max = nav.cummax()
    drawdowns = (nav - rolling_max) / rolling_max
    return float(-drawdowns.min())


def passes_drawdown_constraint(
    returns: pd.Series,
    threshold: float = 0.25,
) -> bool:
    """Return True if max drawdown is strictly below threshold (default 25%)."""
    return max_drawdown(returns) < threshold
