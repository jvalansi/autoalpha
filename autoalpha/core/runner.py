"""Runner: wires Strategy + DataProvider + Executor for backtest and live modes."""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

from autoalpha.core.strategy import Strategy
from autoalpha.core.providers import DataProvider
from autoalpha.core.executors import Executor

logger = logging.getLogger(__name__)


class Runner:
    """Orchestrates the strategy evaluation loop.

    Backtest mode (folds provided):
        For each CPCV fold, call strategy.fit(in_sample_df) once, then
        strategy.predict(bar_series) on every out-of-sample bar.

    Live/paper mode (no folds):
        Call strategy.fit(full_history) once, then predict on every incoming bar.
    """

    def __init__(
        self,
        strategy: Strategy,
        provider: DataProvider,
        executor: Executor,
        tickers: list[str],
    ):
        self._strategy = strategy
        self._provider = provider
        self._executor = executor
        self._tickers = tickers

    def run_backtest(
        self,
        folds: list[tuple[pd.DatetimeIndex, tuple[date, date]]],
    ) -> pd.Series:
        """Run CPCV backtest. folds = [(train_dates, (oos_start, oos_end)), ...]

        train_dates is a DatetimeIndex of the purged training observations produced by
        CPCV.to_runner_folds().  The provider fetches the full [min, max] date range,
        then the history is filtered to the exact purged set before calling fit().
        This prevents look-ahead from purged boundary observations.

        Returns combined out-of-sample daily return series.
        """
        all_returns: list[pd.Series] = []

        for train_dates, (oos_start, oos_end) in folds:
            in_start = train_dates.min().date()
            in_end = train_dates.max().date()
            logger.info("Fold: in-sample %s→%s | OOS %s→%s", in_start, in_end, oos_start, oos_end)

            # Reset executor so each fold starts with clean positions and NAV.
            # This is critical for CPCV k>1 where the same date appears as OOS in
            # multiple folds — without reset, state bleeds across folds.
            self._executor.reset()

            full_history = self._provider.history(self._tickers, in_start, in_end)
            if not full_history.empty:
                date_level = full_history.index.get_level_values("date")
                in_sample = full_history[date_level.isin(train_dates)]
            else:
                in_sample = full_history
            self._strategy.fit(in_sample)

            prev_targets: dict[str, float] = {}
            for bar_date, bar_df in self._provider.bars(self._tickers, oos_start, oos_end):
                open_prices = bar_df["Open"].to_dict() if "Open" in bar_df.columns else {}
                # Execute previous bar's signals at today's open (next-bar fill model)
                if prev_targets and open_prices:
                    self._executor.execute(prev_targets, bar_date, open_prices)
                prev_targets = self._strategy.predict(bar_df, bar_date=bar_date)

            fold_returns = self._executor.returns()
            if not fold_returns.empty:
                all_returns.append(fold_returns)

        if not all_returns:
            return pd.Series(dtype=float)
        return pd.concat(all_returns).sort_index()

    def run_live(self, history_start: date, history_end: date) -> None:
        """Live/paper mode: fit on full history, then stream bars."""
        history = self._provider.history(self._tickers, history_start, history_end)
        self._strategy.fit(history)
        logger.info("Strategy fitted on %d bars", len(history))

        prev_targets: dict[str, float] = {}
        for bar_date, bar_df in self._provider.bars(self._tickers, history_end, date.today()):
            open_prices = bar_df["Open"].to_dict() if "Open" in bar_df.columns else {}
            if prev_targets and open_prices:
                self._executor.execute(prev_targets, bar_date, open_prices)
            prev_targets = self._strategy.predict(bar_df, bar_date=bar_date)
