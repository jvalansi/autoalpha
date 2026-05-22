"""SimExecutor and LiveExecutor."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_TRANSACTION_COST_BPS = 11  # ~11 bps round-trip (see evaluation/costs.py)


class Executor(ABC):
    """Converts target position fractions → orders and tracks P&L."""

    @abstractmethod
    def execute(
        self,
        targets: dict[str, float],
        bar_date: date,
        prices: dict[str, float],
    ) -> None:
        """Apply target positions. prices = {ticker: fill_price}."""

    @abstractmethod
    def portfolio_value(self) -> float:
        """Return current portfolio value."""

    @abstractmethod
    def returns(self) -> pd.Series:
        """Return daily return series (indexed by date)."""


class SimExecutor(Executor):
    """Simulated executor.

    Fill model: fills at next-bar open price (caller passes next open as prices).
    No partial fills. Transaction costs deducted at fill time.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
        overlay: float = 1.0,
    ):
        self._capital = initial_capital
        self._cost_bps = cost_bps
        self._overlay = overlay
        self._positions: dict[str, float] = {}  # ticker -> shares
        self._cash = initial_capital
        self._nav_history: dict[date, float] = {}

    def execute(self, targets: dict[str, float], bar_date: date, prices: dict[str, float]) -> None:
        nav = self._compute_nav(prices)
        self._nav_history[bar_date] = nav

        for ticker, target_frac in targets.items():
            price = prices.get(ticker)
            if price is None or price <= 0:
                logger.warning("No price for %s on %s — skipping", ticker, bar_date)
                continue

            adjusted_frac = target_frac * self._overlay
            target_value = adjusted_frac * nav
            target_shares = target_value / price
            current_shares = self._positions.get(ticker, 0.0)
            delta = target_shares - current_shares

            if abs(delta) < 1e-9:
                continue

            trade_value = abs(delta) * price
            cost = trade_value * (self._cost_bps / 10_000)
            self._cash -= delta * price + cost
            self._positions[ticker] = target_shares

        nav_after = self._compute_nav(prices)
        self._nav_history[bar_date] = nav_after

    def portfolio_value(self, prices: dict[str, float] | None = None) -> float:
        if prices is None:
            prices = {}
        equity = sum(
            shares * prices.get(ticker, 0.0)
            for ticker, shares in self._positions.items()
        )
        return self._cash + equity

    def returns(self) -> pd.Series:
        if len(self._nav_history) < 2:
            return pd.Series(dtype=float)
        nav = pd.Series(self._nav_history).sort_index()
        return nav.pct_change().dropna()

    def nav_series(self) -> pd.Series:
        return pd.Series(self._nav_history).sort_index()

    def _compute_nav(self, prices: dict[str, float]) -> float:
        equity = sum(
            shares * prices.get(ticker, 0.0)
            for ticker, shares in self._positions.items()
        )
        return self._cash + equity


class LiveExecutor(Executor):
    """Stub for live broker API. Subclass and implement _submit_order."""

    def __init__(self, overlay: float = 1.0):
        self._overlay = overlay
        self._returns: list[tuple[date, float]] = []

    def execute(self, targets: dict[str, float], bar_date: date, prices: dict[str, float]) -> None:
        for ticker, frac in targets.items():
            adjusted = frac * self._overlay
            self._submit_order(ticker, adjusted, prices.get(ticker, 0.0), bar_date)

    def portfolio_value(self) -> float:
        raise NotImplementedError("Implement via broker API")

    def returns(self) -> pd.Series:
        if not self._returns:
            return pd.Series(dtype=float)
        return pd.Series({d: r for d, r in self._returns}).sort_index()

    def _submit_order(self, ticker: str, target_frac: float, price: float, bar_date: date) -> None:
        raise NotImplementedError("Implement broker API call here")
