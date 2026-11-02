"""Alpaca broker implementation of LiveExecutor.

Order type is market-on-open (`type=market`, `time_in_force=opg`) to match the
backtest fill model, which fills bar N's signal at bar N+1's open. Alpaca only
accepts `opg` orders while the market is closed (roughly after the previous
close and before 9:28 ET), and `opg` does not support fractional shares — hence
whole-share rounding by default. With `fractional_remainder=True` each order is
split: the whole shares go as `opg`, the fractional rest as a `day` market order
(Alpaca accepts fractional only with `day`), which queues for the regular open.
Names Alpaca doesn't mark `fractionable` keep whole shares only.

Credentials come from the environment:

    ALPACA_API_KEY
    ALPACA_SECRET_KEY
    ALPACA_BASE_URL   (default https://paper-api.alpaca.markets — paper account)

Point ALPACA_BASE_URL at https://api.alpaca.markets only when going live for real.
"""
from __future__ import annotations

import logging
import math
import os
import time
from datetime import date
from typing import Any, Optional

import requests

from autoalpha.core.executors import LiveExecutor

logger = logging.getLogger(__name__)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"


def to_alpaca(symbol: str) -> str:
    """yfinance ticker → Alpaca symbol: BRK-B → BRK.B, CTA-PA (preferred) → CTA.PRA."""
    base, sep, suffix = symbol.partition("-")
    if not sep:
        return symbol
    if len(suffix) == 2 and suffix[0] == "P":
        return f"{base}.PR{suffix[1]}"
    return f"{base}.{suffix}"


def from_alpaca(symbol: str) -> str:
    """Alpaca symbol → yfinance ticker (inverse of to_alpaca)."""
    base, sep, suffix = symbol.partition(".")
    if not sep:
        return symbol
    if len(suffix) == 3 and suffix.startswith("PR"):
        return f"{base}-P{suffix[2]}"
    return f"{base}-{suffix}"


class BrokerError(RuntimeError):
    """Raised when the broker API fails after retries, or rejects a request."""


class AlpacaExecutor(LiveExecutor):
    """Live executor backed by the Alpaca REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        overlay: float = 1.0,
        dry_run: bool = False,
        time_in_force: str = "opg",
        allow_fractional: bool = False,
        fractional_remainder: bool = False,
        min_order_notional: float = 1.0,
        timeout: float = 15.0,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ):
        # `opg` (market-on-open) cannot be fractional — Alpaca rejects it.
        if time_in_force == "opg" and allow_fractional:
            raise ValueError("Alpaca does not accept fractional shares for opg orders")
        if fractional_remainder and (time_in_force != "opg" or allow_fractional):
            raise ValueError("fractional_remainder splits opg orders; use allow_fractional for day orders")

        super().__init__(
            overlay=overlay,
            allow_fractional=allow_fractional or fractional_remainder,
            min_order_notional=min_order_notional,
            dry_run=dry_run,
        )
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self._base_url = (base_url or os.environ.get("ALPACA_BASE_URL", PAPER_BASE_URL)).rstrip("/")
        self._time_in_force = time_in_force
        self._fractional_remainder = fractional_remainder
        self._fractionable: set[str] | None = None
        self._timeout = timeout
        self._max_retries = max_retries
        self._session = session or requests.Session()

        if not (self._api_key and self._secret_key):
            raise BrokerError(
                "Alpaca credentials missing — set ALPACA_API_KEY and ALPACA_SECRET_KEY"
            )
        if self._base_url == LIVE_BASE_URL:
            logger.warning("AlpacaExecutor pointed at the REAL-MONEY endpoint (%s)", LIVE_BASE_URL)

    @property
    def is_paper(self) -> bool:
        return self._base_url != LIVE_BASE_URL

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Call the Alpaca API, retrying transient failures with backoff.

        4xx responses are the broker rejecting us (bad symbol, insufficient
        buying power, market not accepting opg orders) — retrying those just
        repeats the rejection, so they raise immediately. 429 (throttling) is
        the exception: it waits for the rate-limit window and retries.
        """
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                resp = self._session.request(
                    method, url, headers=self._headers(), timeout=self._timeout, **kwargs
                )
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("Alpaca %s %s failed (attempt %d/%d): %s",
                               method, path, attempt + 1, self._max_retries, exc)
                time.sleep(2 ** attempt)
                continue

            if resp.status_code < 400:
                return resp.json() if resp.content else {}

            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise BrokerError(f"Alpaca {method} {path} → {resp.status_code}: {resp.text[:300]}")

            # 429 is throttling (200 requests/min), not a rejection: the request was
            # not processed, so retrying is safe. Wait for the window to reset.
            delay = 2 ** attempt
            if resp.status_code == 429:
                try:
                    reset = float(resp.headers.get("X-RateLimit-Reset", ""))
                    delay = min(max(reset - time.time(), 1.0), 60.0)
                except ValueError:
                    delay = 2 ** (attempt + 2)
            last_exc = BrokerError(f"{resp.status_code}: {resp.text[:300]}")
            if attempt == self._max_retries - 1:
                break
            logger.warning("Alpaca %s %s → %d (attempt %d/%d), retrying in %.0fs",
                           method, path, resp.status_code, attempt + 1, self._max_retries, delay)
            time.sleep(delay)

        raise BrokerError(f"Alpaca {method} {path} failed after {self._max_retries} attempts: {last_exc}")

    # ------------------------------------------------------------------
    # Broker interface
    # ------------------------------------------------------------------

    def account_equity(self) -> float:
        account = self._request("GET", "/v2/account")
        if account.get("trading_blocked") or account.get("account_blocked"):
            raise BrokerError(f"Alpaca account is blocked: {account.get('status')}")
        return float(account["equity"])

    def current_positions(self) -> dict[str, float]:
        positions = self._request("GET", "/v2/positions")
        return {from_alpaca(p["symbol"]): float(p["qty"]) for p in positions}

    def fractionable(self) -> set[str]:
        """Alpaca symbols that accept fractional orders (one assets call per executor)."""
        if self._fractionable is None:
            assets = self._request("GET", "/v2/assets", params={"status": "active", "asset_class": "us_equity"})
            self._fractionable = {a["symbol"] for a in assets if a.get("fractionable")}
        return self._fractionable

    def _place_order(self, symbol: str, qty: float, side: str, price: float | None = None) -> dict:
        if self._fractional_remainder:
            return self._place_split(symbol, qty, side, price)
        payload = {
            "symbol": to_alpaca(symbol),
            "qty": str(qty if self._allow_fractional else int(qty)),
            "side": side,
            "type": "market",
            "time_in_force": self._time_in_force,
        }
        logger.info("Alpaca order: %s %s x%s (%s)", side, symbol, payload["qty"], self._time_in_force)
        return self._request("POST", "/v2/orders", json=payload)

    def _place_split(self, symbol: str, qty: float, side: str, price: float | None) -> dict:
        """Whole shares as opg, the fractional rest as a day market order.

        A rejected leg is recorded, not raised: one unfractionable or below-minimum
        remainder must not stop the rest of the book from being reconciled."""
        whole = math.floor(qty)  # never round up: a sell must not exceed the held qty
        rest = round(qty - whole, 9)
        sym = to_alpaca(symbol)
        if rest > 0 and sym not in self.fractionable():
            rest = 0.0
        # buys below the broker minimum are skipped; sells are always sent so a
        # fractional remainder can be closed out instead of lingering as dust
        if side == "buy" and price is not None and rest * price < self._min_order_notional:
            rest = 0.0
        legs = {}
        for leg, q, tif in (("opg", whole, "opg"), ("day", rest, "day")):
            if q <= 0:
                continue
            payload = {"symbol": sym, "qty": str(q), "side": side, "type": "market", "time_in_force": tif}
            logger.info("Alpaca order: %s %s x%s (%s)", side, symbol, q, tif)
            try:
                legs[leg] = self._request("POST", "/v2/orders", json=payload)
            except BrokerError as exc:
                logger.warning("Alpaca rejected %s leg for %s: %s", leg, symbol, exc)
                legs[leg] = {"error": str(exc)}
        return legs

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def cancel_open_orders(self) -> None:
        """Cancel all open orders. Stale opg orders from a prior run would
        otherwise fill alongside today's, doubling the intended position."""
        if self._dry_run:
            logger.info("DRY RUN — would cancel all open orders")
            return
        self._request("DELETE", "/v2/orders")

    def market_is_open(self) -> bool:
        return bool(self._request("GET", "/v2/clock").get("is_open"))

    def execute(self, targets: dict[str, float], bar_date: date, prices: dict[str, float],
                hold: frozenset[str] = frozenset()) -> None:
        # Reconcile against a clean order book so we never stack duplicate opg orders.
        self.cancel_open_orders()
        if self._fractional_remainder:
            self.fractionable()  # fail before the first order, not halfway through the book
        super().execute(targets, bar_date, prices, hold)
