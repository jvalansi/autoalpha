"""Fama-French 5-factor alpha computation.

Fetches FF5 daily factors from Kenneth French's data library via pandas_datareader.
Returns alpha (intercept) return series and residual Sharpe after regressing out
the five systematic factors.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FF5_DATASET = "F-F_Research_Data_5_Factors_2x3_daily"


@lru_cache(maxsize=1)
def _fetch_ff5() -> pd.DataFrame:
    """Fetch and cache FF5 daily factors. Returns empty DataFrame on failure."""
    try:
        import pandas_datareader.data as pdr

        raw = pdr.DataReader(_FF5_DATASET, "famafrench")[0]
        return raw / 100.0  # convert from percent
    except Exception as exc:
        logger.warning("Failed to fetch FF5 factors: %s", exc)
        return pd.DataFrame()


def compute_alpha_returns(
    strategy_returns: pd.Series,
    factors: Optional[pd.DataFrame] = None,
    min_overlap: int = 30,
) -> tuple[pd.Series, float]:
    """Regress strategy returns on FF5 factors and return residual alpha series.

    Returns (alpha_series, alpha_annualized).
    alpha_series: daily residuals from the OLS regression (unexplained excess returns).
    alpha_annualized: annualized intercept (alpha) from the regression.

    Falls back to raw returns if factors are unavailable or overlap is too small.
    """
    if factors is None:
        factors = _fetch_ff5()

    if factors.empty:
        logger.warning("FF5 factors unavailable — using raw returns as alpha proxy")
        return strategy_returns.copy(), float("nan")

    # Align to common trading days
    common = strategy_returns.index.intersection(factors.index)
    if len(common) < min_overlap:
        logger.warning(
            "Only %d overlapping days with FF5 factors (need %d) — using raw returns",
            len(common),
            min_overlap,
        )
        return strategy_returns.copy(), float("nan")

    y = strategy_returns.loc[common]
    X = factors.loc[common]

    # Compute excess strategy returns (subtract risk-free rate)
    if "RF" in X.columns:
        excess_y = y - X["RF"]
        X_reg = X.drop(columns=["RF"])
    else:
        excess_y = y
        X_reg = X

    # OLS: excess_y = alpha + beta * factors + epsilon
    X_mat = np.column_stack([np.ones(len(X_reg)), X_reg.values])
    coeffs, _, _, _ = np.linalg.lstsq(X_mat, excess_y.values, rcond=None)

    alpha_daily = float(coeffs[0])
    alpha_annualized = alpha_daily * 252

    # Residuals are the strategy's unexplained returns (alpha component)
    factor_contribution = X_mat[:, 1:] @ coeffs[1:]
    residuals = excess_y.values - factor_contribution
    alpha_series = pd.Series(residuals, index=common, name="alpha")

    return alpha_series, alpha_annualized
