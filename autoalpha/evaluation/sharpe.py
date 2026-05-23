"""Sharpe ratio utilities including the Deflated Sharpe Ratio (DSR).

The DSR applies a multiple-testing correction: given T total strategies tested,
it discounts the observed Sharpe by the expected maximum SR that would arise
by chance alone. Reference: Bailey & López de Prado (2014).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

_EULER_GAMMA = 0.5772156649  # Euler-Mascheroni constant


def annualized_sharpe(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Standard annualized Sharpe ratio (mean/std × sqrt(T))."""
    if returns.empty or returns.std(ddof=1) == 0:
        return 0.0
    return float((returns.mean() / returns.std(ddof=1)) * math.sqrt(periods_per_year))


def probabilistic_sharpe(
    returns: pd.Series,
    benchmark_sr: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > benchmark_sr).

    Adjusts for non-normality (skewness, kurtosis) of the return distribution.
    benchmark_sr is an annualized Sharpe ratio.
    Returns a probability in [0, 1].
    """
    n = len(returns)
    if n < 5 or returns.std(ddof=1) == 0:
        return 0.0

    sr_hat = returns.mean() / returns.std(ddof=1)  # per-period
    skew = float(pd.Series(returns).skew()) if n >= 3 else 0.0
    kurt = float(pd.Series(returns).kurtosis()) if n >= 4 else 0.0

    # Convert annualized benchmark to per-period
    sr_star = benchmark_sr / math.sqrt(periods_per_year)

    # Variance of SR estimate (accounting for non-normality)
    variance = (
        1 + 0.5 * sr_hat**2 - skew * sr_hat + (kurt / 4) * sr_hat**2
    ) / (n - 1)
    z = (sr_hat - sr_star) / math.sqrt(max(variance, 1e-12))
    return float(norm.cdf(z))


def expected_max_sr(n_trials: int) -> float:
    """Expected maximum Sharpe ratio over n_trials independent trials.

    Uses the Bailey & de Prado (2014) approximation (per-period units).
    """
    if n_trials <= 1:
        return 0.0
    term1 = (1 - _EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
    term2 = _EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * math.e))
    return float(term1 + term2)


def deflated_sharpe(
    returns: pd.Series,
    n_trials: int,
    periods_per_year: int = 252,
) -> float:
    """Deflated Sharpe Ratio (DSR).

    Penalizes the Sharpe ratio for the number of strategies tested (selection bias).
    Returns a probability in [0, 1].
    Threshold for statistical significance: DSR > 0.95.

    n_trials: cumulative count of all strategies ever evaluated (from research memory).

    Benchmark SR formula (Bailey & López de Prado 2014, Eq. 8):
      SR* = E[max_SR_n] × SE(annualized SR)
      where SE ≈ sqrt(periods_per_year / (n - 1)) under the null hypothesis.
    This scales the benchmark correctly for backtest length.
    """
    if n_trials < 1:
        n_trials = 1

    n = len(returns)
    emax = expected_max_sr(n_trials)
    se_annual = math.sqrt(periods_per_year / max(n - 1, 1))
    benchmark_sr = emax * se_annual
    return probabilistic_sharpe(returns, benchmark_sr=benchmark_sr, periods_per_year=periods_per_year)


def passes_sharpe_threshold(
    returns: pd.Series,
    n_trials: int,
    threshold: float = 0.95,
) -> bool:
    """Return True if DSR > threshold (default 95% confidence level)."""
    return deflated_sharpe(returns, n_trials) > threshold
