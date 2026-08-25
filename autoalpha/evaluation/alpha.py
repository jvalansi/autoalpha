"""Factor-alpha computation.

Two complementary measures:

1. `compute_alpha_returns` — Fama-French 5-factor residuals. The canonical
   objective-function input (PLAN.md Phase 2). FF5 daily factors come from
   Kenneth French's data library via pandas_datareader and are published with
   a multi-week lag, so recent windows are only partially covered.

2. `compute_benchmark_alpha` — OLS of strategy returns on the book's own
   investable benchmark (equal-weight universe). Always available, covers the
   full window, and answers the operational question directly: is there edge
   over just holding the universe?

Both are needed. FF5 says "is this systematic-factor beta in disguise?";
benchmark alpha says "did we beat the thing we could have held instead?".
"""
from __future__ import annotations

import datetime as dt
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import t as _student_t

logger = logging.getLogger(__name__)

_FF5_DATASET = "F-F_Research_Data_5_Factors_2x3_daily"
_FF5_START = dt.date(2000, 1, 1)
_FF5_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "ff5_daily.parquet"
# Re-download at most once a day. FF updates monthly, so this is generous.
_FF5_MAX_AGE_DAYS = 1


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
    return age.days < _FF5_MAX_AGE_DAYS


@lru_cache(maxsize=1)
def _fetch_ff5() -> pd.DataFrame:
    """Fetch FF5 daily factors (as decimals), disk-cached.

    The research loop spawns one subprocess per backtest, so an in-process
    lru_cache alone would re-download the zip on every iteration. The parquet
    cache is what actually prevents that. Returns an empty DataFrame only if
    both the network and the cache fail.
    """
    if _cache_is_fresh(_FF5_CACHE):
        try:
            return pd.read_parquet(_FF5_CACHE)
        except Exception as exc:  # corrupt cache — fall through to refetch
            logger.warning("FF5 cache unreadable (%s) — refetching", exc)

    try:
        import pandas_datareader.data as pdr

        raw = pdr.DataReader(_FF5_DATASET, "famafrench", start=_FF5_START)[0]
        factors = raw / 100.0  # published in percent
        factors.index = pd.to_datetime(factors.index.astype(str))
        _FF5_CACHE.parent.mkdir(parents=True, exist_ok=True)
        factors.to_parquet(_FF5_CACHE)
        return factors
    except Exception as exc:
        logger.warning("Failed to fetch FF5 factors: %s", exc)

    # Network failed — a stale cache still beats raw returns.
    if _FF5_CACHE.exists():
        try:
            logger.warning("Using stale FF5 cache at %s", _FF5_CACHE)
            return pd.read_parquet(_FF5_CACHE)
        except Exception as exc:
            logger.warning("Stale FF5 cache unreadable: %s", exc)
    return pd.DataFrame()


def ff5_coverage(index: pd.DatetimeIndex) -> tuple[int, int]:
    """Return (n_covered, n_total) days of `index` present in the FF5 factors."""
    factors = _fetch_ff5()
    if factors.empty:
        return 0, len(index)
    return len(pd.DatetimeIndex(index).intersection(factors.index)), len(index)


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS with t-statistics. X must already include the intercept column.

    Returns (coeffs, t_stats, residuals).
    """
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coeffs
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    # Numerically perfect fit — the strategy is an exact linear function of the
    # regressors (e.g. a levered copy of the benchmark), so there is no residual
    # variation to test against. Report no evidence (t=0) rather than a t-stat
    # produced by dividing floating-point dust.
    if sigma2 <= 1e-12 * max(float(np.var(y)), 1e-300):
        return coeffs, np.zeros_like(coeffs), resid
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 1e-300))
    return coeffs, coeffs / se, resid


def compute_alpha_returns(
    strategy_returns: pd.Series,
    factors: Optional[pd.DataFrame] = None,
    min_overlap: int = 30,
) -> tuple[pd.Series, float]:
    """Regress strategy returns on FF5 factors and return residual alpha series.

    Returns (alpha_series, alpha_annualized).
    alpha_series: daily residuals (alpha + idiosyncratic noise, factor beta removed).
    alpha_annualized: annualized intercept from the regression.

    Falls back to raw returns if factors are unavailable or overlap is too small.
    """
    if factors is None:
        factors = _fetch_ff5()

    if factors.empty:
        logger.warning("FF5 factors unavailable — using raw returns as alpha proxy")
        return strategy_returns.copy(), float("nan")

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

    if "RF" in X.columns:
        excess_y = y - X["RF"]
        X_reg = X.drop(columns=["RF"])
    else:
        excess_y = y
        X_reg = X

    X_mat = np.column_stack([np.ones(len(X_reg)), X_reg.values])
    coeffs, _, _ = _ols(excess_y.values, X_mat)

    alpha_annualized = float(coeffs[0]) * 252

    # Residual = excess return with factor exposure stripped out; the intercept
    # is deliberately left in so the series' own mean is the alpha.
    factor_contribution = X_mat[:, 1:] @ coeffs[1:]
    alpha_series = pd.Series(excess_y.values - factor_contribution, index=common, name="alpha")

    return alpha_series, alpha_annualized


def ff5_alpha_stats(strategy_returns: pd.Series, min_overlap: int = 30) -> dict:
    """Full FF5 regression diagnostics: alpha, t-stat, factor betas, coverage."""
    factors = _fetch_ff5()
    n_total = len(strategy_returns)
    if factors.empty:
        return {"available": False, "reason": "FF5 factors unavailable",
                "n_overlap": 0, "n_total": n_total}

    common = strategy_returns.index.intersection(factors.index)
    if len(common) < min_overlap:
        return {
            "available": False,
            "reason": f"only {len(common)}/{n_total} days covered by FF5 "
                      f"(published through {factors.index.max().date()}); need {min_overlap}",
            "n_overlap": len(common),
            "n_total": n_total,
        }

    y = strategy_returns.loc[common]
    X = factors.loc[common]
    excess_y = (y - X["RF"]) if "RF" in X.columns else y
    X_reg = X.drop(columns=["RF"]) if "RF" in X.columns else X

    X_mat = np.column_stack([np.ones(len(X_reg)), X_reg.values])
    coeffs, tstats, resid = _ols(excess_y.values, X_mat)

    resid_std = float(np.std(resid, ddof=1))
    return {
        "available": True,
        "n_overlap": len(common),
        "n_total": n_total,
        "coverage_end": str(factors.index.max().date()),
        "alpha_daily": float(coeffs[0]),
        "alpha_annualized": float(coeffs[0]) * 252,
        "alpha_t": float(tstats[0]),
        "alpha_p": float(2 * _student_t.sf(abs(tstats[0]), df=max(len(common) - X_mat.shape[1], 1))),
        "betas": {c: float(b) for c, b in zip(X_reg.columns, coeffs[1:])},
        "information_ratio": float(coeffs[0] / resid_std * np.sqrt(252)) if resid_std > 0 else 0.0,
    }


def compute_benchmark_alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    min_overlap: int = 20,
) -> dict:
    """OLS of strategy returns on a single benchmark: alpha, t-stat, beta, IR.

    This is the "did we beat the thing we could have held instead?" test.
    Unlike FF5 it has no publication lag, so it covers the full window.
    """
    common = strategy_returns.index.intersection(benchmark_returns.index)
    n_total = len(strategy_returns)
    if len(common) < min_overlap:
        return {
            "available": False,
            "reason": f"only {len(common)} overlapping days (need {min_overlap})",
            "n_overlap": len(common),
            "n_total": n_total,
        }

    y = strategy_returns.loc[common].values
    x = benchmark_returns.loc[common].values
    X_mat = np.column_stack([np.ones(len(x)), x])
    coeffs, tstats, resid = _ols(y, X_mat)

    resid_std = float(np.std(resid, ddof=1))
    active = y - x  # naive excess, for the tracking-error view
    active_std = float(np.std(active, ddof=1))
    dof = max(len(common) - 2, 1)
    return {
        "available": True,
        "n_overlap": len(common),
        "n_total": n_total,
        "alpha_daily": float(coeffs[0]),
        "alpha_annualized": float(coeffs[0]) * 252,
        "alpha_t": float(tstats[0]),
        "alpha_p": float(2 * _student_t.sf(abs(tstats[0]), df=dof)),
        "beta": float(coeffs[1]),
        "beta_t": float(tstats[1]),
        "information_ratio": float(coeffs[0] / resid_std * np.sqrt(252)) if resid_std > 0 else 0.0,
        "mean_active_return_annualized": float(active.mean() * 252),
        "tracking_error_annualized": float(active_std * np.sqrt(252)),
        # Beta-stripped return series with the intercept ADDED BACK, so its mean
        # is the alpha and its Sharpe is the information ratio. (Raw OLS
        # residuals are mean-zero by construction — feeding those to a rolling
        # Sharpe would score every signal at exactly 0.) Matches the convention
        # in compute_alpha_returns.
        "residuals": pd.Series(resid + coeffs[0], index=common, name="benchmark_alpha"),
    }
