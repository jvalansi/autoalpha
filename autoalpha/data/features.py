"""Fractional differentiation (López de Prado Ch. 5).

find_min_d(series): grid search [0.0, 0.1, …, 1.0] then refine to 0.01 resolution.
                    Returns minimum d such that ADF p-value < 0.05 (stationarity).
fracdiff(series, d): apply fractional differencing with memory cutoff.

IMPORTANT: d must be computed per ticker per CPCV fold on in-sample data only.
           Never compute d on the full dataset — that introduces look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

ADF_THRESHOLD = 0.05
# Truncate fractional-diff weights below this value.
# 1e-5 requires >4000 bars for d=0.1; 1e-3 gives windows ≤74 bars for all d,
# which is practical for typical CPCV in-sample periods. Consistent with
# LdP AFML examples that use 1e-3.
MEMORY_CUTOFF = 1e-3


def _get_weights(d: float, size: int) -> np.ndarray:
    """Compute fractional differencing weights w_k for k=0..size-1."""
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
        if abs(w[-1]) < MEMORY_CUTOFF:
            break
    return np.array(w[::-1])  # oldest weight first


def fracdiff(series: pd.Series, d: float) -> pd.Series:
    """Apply fractional differencing with order d, dropping NaNs at the head.

    The window width is determined by MEMORY_CUTOFF, not by len(series).
    If the series is shorter than the window, an empty Series is returned.
    For d=0.4 the window is ~1459 bars; use in-sample series of at least
    that length, or increase MEMORY_CUTOFF to trade off memory vs. window size.
    """
    if d == 0:
        return series.copy()
    if d == 1:
        return series.diff().dropna()

    # Use a large cap so MEMORY_CUTOFF (not len(series)) determines window width.
    # With MEMORY_CUTOFF=1e-3 all windows are ≤74 bars; 10_000 is a safe upper bound.
    weights = _get_weights(d, max(len(series), 10_000))
    width = len(weights)
    result = {}
    idx = series.index

    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1: i + 1].values
        if len(window) == width and not np.isnan(window).any():
            result[idx[i]] = float(np.dot(weights, window))

    return pd.Series(result)


def _adf_pvalue(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 20:
        return 1.0
    try:
        return float(adfuller(clean, autolag="AIC")[1])
    except Exception:
        return 1.0


def find_min_d(series: pd.Series) -> float:
    """Find minimum d in [0, 1] such that the fracdiff series passes ADF at p<0.05.

    Grid search at 0.1 resolution, then refine to 0.01.
    Returns 1.0 if no d achieves stationarity.
    """
    # coarse grid
    coarse_d = None
    for d_raw in range(0, 11):
        d = d_raw / 10.0
        fd = fracdiff(series, d)
        if _adf_pvalue(fd) < ADF_THRESHOLD:
            coarse_d = d
            break

    if coarse_d is None:
        return 1.0

    if coarse_d == 0:
        return 0.0

    # refine: search [coarse_d - 0.1, coarse_d] at 0.01 resolution
    low = max(0.0, coarse_d - 0.1)
    best_d = coarse_d
    for d_int in range(int(low * 100), int(coarse_d * 100) + 1):
        d = d_int / 100.0
        fd = fracdiff(series, d)
        if _adf_pvalue(fd) < ADF_THRESHOLD:
            best_d = d
            break

    return best_d


def add_fracdiff_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    d_map: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Apply fractional differencing to selected columns in df.

    If d_map is provided, use those d values (for applying fold-specific d to OOS data).
    Otherwise, compute minimum d per column.

    Returns (transformed_df, d_map) where d_map = {col: d_used}.
    """
    if columns is None:
        columns = [c for c in ["Close", "Volume"] if c in df.columns]

    result = df.copy()
    used_d: dict[str, float] = {}

    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].astype(float)
        d = d_map[col] if (d_map and col in d_map) else find_min_d(series)
        used_d[col] = d
        result[f"{col}_fd"] = fracdiff(series, d)

    return result, used_d
