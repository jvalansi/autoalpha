"""Combinatorial Purged Cross-Validation (CPCV).

Reference: López de Prado, Advances in Financial Machine Learning, Ch. 12.

Key idea: split data into N groups; hold out k groups at a time as the test set.
This generates C(N, k) train/test splits — far more backtest paths than k-fold CV,
which reduces the probability of a spurious strategy passing all paths.

Purging:  removes training observations whose labels overlap with the test period.
          Gap = label horizon h (max triple-barrier expiry, typically 20 days).
Embargo:  additional buffer of 5 days after the purge gap.
"""
from __future__ import annotations

import itertools
from datetime import date
from typing import Iterator

import numpy as np
import pandas as pd


class CPCV:
    """Combinatorial Purged Cross-Validation.

    Generates C(n_splits, n_test_splits) (train_dates, test_dates) pairs.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_days: int = 20,
        embargo_days: int = 5,
    ):
        if n_test_splits >= n_splits:
            raise ValueError(f"n_test_splits ({n_test_splits}) must be < n_splits ({n_splits})")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    # ------------------------------------------------------------------
    # Core split generator
    # ------------------------------------------------------------------

    def split(
        self,
        dates: pd.DatetimeIndex,
    ) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Yield (train_dates, test_dates) for each of C(N, k) folds.

        Purging removes training dates within purge_days of any test date.
        Embargo removes an additional buffer after the last test date.
        """
        dates = dates.sort_values().unique()
        n = len(dates)
        if n < self.n_splits:
            raise ValueError(
                f"Not enough dates ({n}) for {self.n_splits} splits"
            )

        # Assign each date to a group (0-indexed)
        groups = np.empty(n, dtype=int)
        group_size = n // self.n_splits
        for g in range(self.n_splits):
            start = g * group_size
            end = (start + group_size) if g < self.n_splits - 1 else n
            groups[start:end] = g

        purge_td = pd.offsets.BDay(self.purge_days)
        embargo_td = pd.offsets.BDay(self.embargo_days)

        for test_groups in itertools.combinations(range(self.n_splits), self.n_test_splits):
            test_mask = np.isin(groups, test_groups)
            test_dates = dates[test_mask]
            if len(test_dates) == 0:
                continue

            # Purge per test group — applying across the full span would exclude
            # training groups that sit between non-adjacent test groups.
            purge_mask = np.zeros(n, dtype=bool)
            for g in test_groups:
                g_dates = dates[groups == g]
                g_start = g_dates.min()
                g_end = g_dates.max()
                purge_mask |= (dates >= g_start - purge_td) & (dates <= g_end + embargo_td)

            train_mask = ~test_mask & ~purge_mask
            train_dates = dates[train_mask]

            if len(train_dates) == 0:
                continue

            yield train_dates, test_dates

    # ------------------------------------------------------------------
    # Runner-compatible fold format
    # ------------------------------------------------------------------

    def to_runner_folds(
        self,
        dates: pd.DatetimeIndex,
    ) -> list[tuple[tuple[date, date], tuple[date, date]]]:
        """Convert CPCV splits to Runner-compatible ((in_start, in_end), (oos_start, oos_end)).

        Non-contiguous OOS periods (k > 1) are split into separate contiguous segments,
        each paired with the same training window.
        """
        folds: list[tuple[tuple[date, date], tuple[date, date]]] = []

        for train_dates, test_dates in self.split(dates):
            in_start = train_dates.min().date()
            in_end = train_dates.max().date()

            for seg_start, seg_end in _contiguous_segments(test_dates):
                folds.append(
                    ((in_start, in_end), (seg_start.date(), seg_end.date()))
                )

        return folds


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _contiguous_segments(
    dates: pd.DatetimeIndex,
    max_gap_days: int = 10,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a DatetimeIndex into contiguous segments (gap ≤ max_gap_days)."""
    if len(dates) == 0:
        return []

    dates = dates.sort_values()
    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    seg_start = dates[0]
    seg_end = dates[0]

    for d in dates[1:]:
        if (d - seg_end).days > max_gap_days:
            segments.append((seg_start, seg_end))
            seg_start = d
        seg_end = d

    segments.append((seg_start, seg_end))
    return segments
