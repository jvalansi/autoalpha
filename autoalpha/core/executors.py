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

    @abstractmethod
    def reset(self) -> None:
        """Reset all positions, cash, and history to the initial state."""


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
        max_weight: float | None = 0.10,
    ):
        self._capital = initial_capital
        self._cost_bps = cost_bps
        self._overlay = overlay
        self._max_weight = max_weight
        self._positions: dict[str, float] = {}  # ticker -> shares
        self._cash = initial_capital
        self._nav_history: dict[date, float] = {}
        # Last known mark for each ticker, used to value positions on bars where
        # the ticker is absent from `prices` (e.g. dropped from the universe).
        # Without this, _compute_nav defaulted missing prices to 0 and produced
        # spurious single-day NAV craters.
        self._last_price: dict[str, float] = {}

    def reset(self) -> None:
        """Reset to initial state — call between CPCV folds."""
        self._positions = {}
        self._cash = self._capital
        self._nav_history = {}
        self._last_price = {}

    def _cap_weights(self, targets: dict[str, float]) -> dict[str, float]:
        """Cap each positive weight at max_weight, drop non-positive weights.

        Leftover gross (when caps bind) sits in cash rather than being
        redistributed. This is a conservative guardrail: a signal that wanted
        78% in one name ends up at the cap, with the rest uninvested, instead
        of taking the concentrated bet.
        """
        cap = self._max_weight
        if not targets:
            return {}
        if cap is None or cap >= 1.0:
            return {t: v for t, v in targets.items() if v > 0}
        return {t: min(v, cap) for t, v in targets.items() if v > 0}

    def execute(self, targets: dict[str, float], bar_date: date, prices: dict[str, float]) -> None:
        # Refresh last-known marks before we touch NAV, so today's prices win
        # over any stale mark when valuing existing positions.
        for ticker, price in prices.items():
            if price and price > 0:
                self._last_price[ticker] = price

        targets = self._cap_weights(targets)

        nav = self._compute_nav(prices)
        self._nav_history[bar_date] = nav

        # Close positions not in targets (strategy said "go to cash" for these).
        # Fall back to last-known mark when the ticker has no quote today, so
        # delisted/dropped names don't sit as zombie positions forever.
        for ticker in list(self._positions.keys()):
            if ticker not in targets:
                price = prices.get(ticker) or self._last_price.get(ticker, 0.0)
                if price and price > 0:
                    shares = self._positions.pop(ticker)
                    trade_value = abs(shares) * price
                    cost = trade_value * (self._cost_bps / 10_000)
                    self._cash += shares * price - cost
                else:
                    logger.warning(
                        "Cannot close %s on %s — no current or last-known price; "
                        "writing off position", ticker, bar_date,
                    )
                    self._positions.pop(ticker)

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
            return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        nav = pd.Series(self._nav_history).sort_index()
        nav.index = pd.to_datetime(nav.index)
        return nav.pct_change().dropna()

    def nav_series(self) -> pd.Series:
        return pd.Series(self._nav_history).sort_index()

    def _compute_nav(self, prices: dict[str, float]) -> float:
        equity = sum(
            shares * (prices.get(ticker) or self._last_price.get(ticker, 0.0))
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

    def reset(self) -> None:
        self._returns = []

    def _submit_order(self, ticker: str, target_frac: float, price: float, bar_date: date) -> None:
        raise NotImplementedError("Implement broker API call here")
