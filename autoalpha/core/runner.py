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
        folds: list[tuple[tuple[date, date], tuple[date, date]]],
    ) -> pd.Series:
        """Run CPCV backtest. folds = [((in_start, in_end), (oos_start, oos_end)), ...]

        Returns combined out-of-sample daily return series.
        """
        all_returns: list[pd.Series] = []

        for (in_start, in_end), (oos_start, oos_end) in folds:
            logger.info("Fold: in-sample %s→%s | OOS %s→%s", in_start, in_end, oos_start, oos_end)

            in_sample = self._provider.history(self._tickers, in_start, in_end)
            self._strategy.fit(in_sample)

            prev_prices: dict[str, float] = {}
            for bar_date, bar_df in self._provider.bars(self._tickers, oos_start, oos_end):
                prices = bar_df["Open"].to_dict() if "Open" in bar_df.columns else {}
                targets = self._strategy.predict(bar_df)
                if targets and prev_prices:
                    self._executor.execute(targets, bar_date, prev_prices)
                prev_prices = bar_df["Close"].to_dict() if "Close" in bar_df.columns else {}

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

        prev_prices: dict[str, float] = {}
        for bar_date, bar_df in self._provider.bars(self._tickers, history_end, date.today()):
            prices = bar_df["Open"].to_dict() if "Open" in bar_df.columns else {}
            targets = self._strategy.predict(bar_df)
            if targets and prev_prices:
                self._executor.execute(targets, bar_date, prev_prices)
            prev_prices = bar_df["Close"].to_dict() if "Close" in bar_df.columns else {}
