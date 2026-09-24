"""Risk controls for live trading, ported from earnings-trader/src/risk.py.

Blocks new entries when a limit is breached; never blocks exits.

  1. Daily loss limit  — equity change since the day's first evaluation <= -daily_loss_limit
  2. Drawdown breaker  — equity <= peak equity * (1 - max_drawdown)
  3. Manual kill switch — the halt file exists, or TRADING_HALTED=1 in the environment

Gates 1 and 2 latch (persisted in the state file) until resume(), so a bad day cannot
silently resume trading the next morning. The kill switch is on exactly while the
file/env var is present. Unlike earnings-trader, equity comes straight from the broker
account (the system of record) rather than being reconstructed from a trade log.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RiskStatus:
    entries_allowed: bool
    reasons: list[str] = field(default_factory=list)
    equity: float = 0.0
    peak_equity: float = 0.0
    drawdown_pct: float = 0.0
    daily_pnl_pct: float = 0.0

    def line(self) -> str:
        if self.entries_allowed:
            return (f"🟢 Risk OK — equity ${self.equity:,.0f} | DD {self.drawdown_pct:.1%} | "
                    f"day {self.daily_pnl_pct:+.1%}")
        return f"🛑 ENTRIES HALTED — {'; '.join(self.reasons)} | equity ${self.equity:,.0f}"


class RiskGuard:
    def __init__(self, state_path: Path, halt_path: Path,
                 daily_loss_limit: float = 0.04, max_drawdown: float = 0.15):
        self.state_path = Path(state_path)
        self.halt_path = Path(halt_path)
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown = max_drawdown

    def _load(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.rename(self.state_path)

    def _kill_switch(self) -> str | None:
        if self.halt_path.exists():
            note = self.halt_path.read_text().strip()
            return f"manual halt ({note or self.halt_path})"
        if os.getenv("TRADING_HALTED", "").lower() in ("1", "yes", "true"):
            return "manual halt (TRADING_HALTED set)"
        return None

    def evaluate(self, equity: float, today: str) -> RiskStatus:
        state = self._load()
        peak = max(float(state.get("peak_equity", equity)), equity)
        drawdown = (peak - equity) / peak if peak > 0 else 0.0

        if state.get("day_open_date") != today:
            state["day_open_equity"] = float(state.get("last_equity", equity))
            state["day_open_date"] = today
        day_open = float(state["day_open_equity"])
        daily = (equity - day_open) / day_open if day_open > 0 else 0.0

        reasons = []
        if daily <= -self.daily_loss_limit:
            reasons.append(f"daily loss {daily:.1%} breached {-self.daily_loss_limit:.0%}")
        if drawdown >= self.max_drawdown:
            reasons.append(f"drawdown {drawdown:.1%} breached {self.max_drawdown:.0%}")
        if reasons and not state.get("halted_since"):
            state["halted_since"] = datetime.now(timezone.utc).isoformat()
            state["halt_reasons"] = reasons
            logger.error("RISK HALT triggered: %s", "; ".join(reasons))
        elif state.get("halted_since"):
            reasons = list(state.get("halt_reasons", [])) + reasons
        if kill := self._kill_switch():
            reasons.append(kill)

        state.update(peak_equity=peak, last_equity=equity, last_evaluated=today)
        self._save(state)
        return RiskStatus(not reasons, reasons, equity, peak, drawdown, daily)

    def halt(self, reason: str = "") -> None:
        self.halt_path.parent.mkdir(parents=True, exist_ok=True)
        self.halt_path.write_text(reason or "halted manually")

    def resume(self) -> None:
        """Clear the kill switch and any latched breach."""
        self.halt_path.unlink(missing_ok=True)
        state = self._load()
        state.pop("halted_since", None)
        state.pop("halt_reasons", None)
        self._save(state)
