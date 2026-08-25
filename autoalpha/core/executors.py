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
    """Broker-agnostic live executor.

    Holds the target-fraction → order reconciliation logic that is identical
    across brokers; subclasses implement only the three broker calls:
    `account_equity`, `current_positions`, and `_place_order`.

    Fill model matches the backtest: orders are sized off the current account
    equity and submitted as market-on-open, so they fill at the next open —
    the same bar the SimExecutor fills on.
    """

    def __init__(
        self,
        overlay: float = 1.0,
        allow_fractional: bool = False,
        min_order_notional: float = 1.0,
        dry_run: bool = False,
    ):
        self._overlay = overlay
        self._allow_fractional = allow_fractional
        self._min_order_notional = min_order_notional
        self._dry_run = dry_run
        self._nav_history: dict[date, float] = {}
        self._orders: list[dict] = []

    # ------------------------------------------------------------------
    # Broker interface — implement in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def account_equity(self) -> float:
        """Current account equity (cash + market value of positions)."""

    @abstractmethod
    def current_positions(self) -> dict[str, float]:
        """Open positions as {ticker: signed share quantity}."""

    @abstractmethod
    def _place_order(self, symbol: str, qty: float, side: str) -> dict:
        """Submit a single order. side is 'buy' or 'sell'. Returns broker response."""

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def _target_shares(self, frac: float, equity: float, price: float) -> float:
        shares = (frac * self._overlay * equity) / price
        return shares if self._allow_fractional else float(int(shares))

    def execute(self, targets: dict[str, float], bar_date: date, prices: dict[str, float]) -> None:
        equity = self.account_equity()
        self._nav_history[bar_date] = equity

        current = self.current_positions()
        symbols = set(targets) | set(current)

        for symbol in sorted(symbols):
            price = prices.get(symbol)
            held = current.get(symbol, 0.0)
            frac = targets.get(symbol, 0.0)

            if frac <= 0:
                # Not in targets (or explicitly flat) — close the position.
                # Closing needs no price: qty is whatever we hold.
                if held != 0:
                    self._order(symbol, abs(held), "sell" if held > 0 else "buy", price)
                continue

            if price is None or price <= 0:
                logger.warning("No price for %s on %s — skipping order", symbol, bar_date)
                continue

            delta = self._target_shares(frac, equity, price) - held
            if abs(delta) * price < self._min_order_notional:
                continue
            if not self._allow_fractional and abs(delta) < 1:
                continue
            self._order(symbol, abs(delta), "buy" if delta > 0 else "sell", price)

    def _order(self, symbol: str, qty: float, side: str, price: float | None) -> None:
        if qty <= 0:
            return
        record = {"symbol": symbol, "qty": qty, "side": side, "price": price}
        if self._dry_run:
            logger.info("DRY RUN — would %s %s x%s", side, symbol, qty)
            record["dry_run"] = True
        else:
            record["response"] = self._place_order(symbol, qty, side)
        self._orders.append(record)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def portfolio_value(self) -> float:
        return self.account_equity()

    def orders(self) -> list[dict]:
        """Orders submitted (or simulated, under dry_run) this session."""
        return list(self._orders)

    def returns(self) -> pd.Series:
        if len(self._nav_history) < 2:
            return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        nav = pd.Series(self._nav_history).sort_index()
        nav.index = pd.to_datetime(nav.index)
        return nav.pct_change().dropna()

    def nav_series(self) -> pd.Series:
        return pd.Series(self._nav_history).sort_index()

    def reset(self) -> None:
        """Clear local history only — does NOT liquidate broker positions."""
        self._nav_history = {}
        self._orders = []
