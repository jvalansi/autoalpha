"""Marginal Sharpe contribution.

Strategy 1: standalone alpha Sharpe (no existing portfolio).
Strategy 2+: Sharpe of alpha returns after regressing out the existing
             portfolio's alpha returns (diversification adjustment).

Stores and retrieves the existing portfolio alpha return series from memory.db.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from autoalpha.evaluation.sharpe import annualized_sharpe

logger = logging.getLogger(__name__)


def marginal_sharpe(
    strategy_alpha: pd.Series,
    portfolio_alpha: Optional[pd.Series] = None,
    periods_per_year: int = 252,
) -> float:
    """Compute marginal Sharpe contribution.

    strategy_alpha: daily alpha returns of the candidate strategy.
    portfolio_alpha: daily alpha returns of the existing portfolio.
                     Pass None for the first strategy.

    For strategy 1 (portfolio_alpha is None or empty): returns standalone Sharpe.
    For strategy 2+: returns Sharpe of the OLS residual after regressing
                     strategy_alpha on portfolio_alpha.
    """
    if portfolio_alpha is None or portfolio_alpha.empty:
        return annualized_sharpe(strategy_alpha, periods_per_year)

    common = strategy_alpha.index.intersection(portfolio_alpha.index)
    if len(common) < 10:
        logger.warning(
            "Only %d common dates between strategy and portfolio alpha — using standalone Sharpe",
            len(common),
        )
        return annualized_sharpe(strategy_alpha, periods_per_year)

    y = strategy_alpha.loc[common].values
    x = portfolio_alpha.loc[common].values

    # OLS without intercept: strategy_alpha = beta * portfolio_alpha + residual
    # Including an intercept would subtract the mean from residuals, zeroing out
    # the alpha return component and producing Sharpe ≈ 0 even for good strategies.
    X = x[:, np.newaxis]
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ coeffs

    residual_series = pd.Series(residuals, index=common)
    return annualized_sharpe(residual_series, periods_per_year)


def combine_portfolio_alpha(
    existing_portfolio_alpha: Optional[pd.Series],
    new_strategy_alpha: pd.Series,
    weight: float,
) -> pd.Series:
    """Blend a new strategy's alpha into the existing portfolio alpha series.

    Used after a strategy is accepted to update the portfolio alpha for the
    next marginal Sharpe evaluation.

    existing_portfolio_alpha: None if this is the first strategy.
    weight: allocation weight for the new strategy (0 to 1).
            Pass 1/n_active_signals for equal-weight allocation.
    """
    if existing_portfolio_alpha is None or existing_portfolio_alpha.empty:
        return new_strategy_alpha.copy()

    common = existing_portfolio_alpha.index.intersection(new_strategy_alpha.index)
    if common.empty:
        return existing_portfolio_alpha.copy()

    combined = (1 - weight) * existing_portfolio_alpha.loc[common] + weight * new_strategy_alpha.loc[common]
    return combined
