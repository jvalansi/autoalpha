"""Signal library with Darwinian weights.

Each validated signal starts at weight 1.0 and is updated daily based on its
63-day rolling alpha Sharpe:

  weight = clip(rolling_sharpe / TARGET_SHARPE, FLOOR, CEILING)

Decay:  weight pinned at floor for ≥ 63 consecutive days → status = 'decayed'
Death:  weight pinned at floor for ≥ 126 consecutive days → status = 'dead'
        (removed from live trading; kept in DB for audit)

Inspired by ATLAS (General Intelligence Capital) Darwinian weighting scheme.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from autoalpha.evaluation.sharpe import annualized_sharpe

logger = logging.getLogger(__name__)

_WEIGHT_FLOOR = 0.3
_WEIGHT_CEILING = 2.5
_TARGET_SHARPE = 0.5   # Sharpe at which weight == 1.0
_ROLLING_WINDOW = 63   # trading days
_DECAY_DAYS = 63       # days at floor → decayed
_DEATH_DAYS = 126      # days at floor → dead

_DEFAULT_DB = Path(__file__).resolve().parents[2] / "research" / "memory.db"


def _init_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS signal_library (
            id            INTEGER PRIMARY KEY,
            name          TEXT    UNIQUE NOT NULL,
            status        TEXT    NOT NULL DEFAULT 'active',
            weight        REAL    NOT NULL DEFAULT 1.0,
            days_at_floor INTEGER NOT NULL DEFAULT 0,
            created_at    TEXT    NOT NULL DEFAULT (date('now')),
            updated_at    TEXT    NOT NULL DEFAULT (date('now'))
        );
        CREATE TABLE IF NOT EXISTS weight_history (
            id             INTEGER PRIMARY KEY,
            signal_name    TEXT    NOT NULL,
            as_of_date     TEXT    NOT NULL,
            weight         REAL    NOT NULL,
            rolling_sharpe REAL,
            UNIQUE(signal_name, as_of_date)
        );
    """)
    conn.commit()


class SignalLibrary:
    """Manages validated signals with Darwinian weights backed by SQLite."""

    def __init__(self, db_path: Path = _DEFAULT_DB):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        _init_tables(self._conn)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_signal(self, name: str) -> None:
        """Register a new signal at weight 1.0 (no-op if already exists)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO signal_library (name) VALUES (?)", (name,)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_weight(self, name: str) -> float:
        row = self._conn.execute(
            "SELECT weight FROM signal_library WHERE name = ?", (name,)
        ).fetchone()
        return float(row[0]) if row else 1.0

    def active_signals(self) -> list[str]:
        """Names of all non-dead signals (includes 'decayed' at floor weight)."""
        rows = self._conn.execute(
            "SELECT name FROM signal_library WHERE status != 'dead'"
        ).fetchall()
        return [r[0] for r in rows]

    def all_weights(self) -> dict[str, float]:
        """Return {name: weight} for all active signals."""
        rows = self._conn.execute(
            "SELECT name, weight FROM signal_library WHERE status != 'dead'"
        ).fetchall()
        return {r[0]: float(r[1]) for r in rows}

    def weight_history(self, name: str) -> pd.DataFrame:
        rows = self._conn.execute(
            """SELECT as_of_date, weight, rolling_sharpe
               FROM weight_history WHERE signal_name = ?
               ORDER BY as_of_date""",
            (name,),
        ).fetchall()
        if not rows:
            return pd.DataFrame(columns=["weight", "rolling_sharpe"])
        df = pd.DataFrame(rows, columns=["date", "weight", "rolling_sharpe"])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    # ------------------------------------------------------------------
    # Status management
    # ------------------------------------------------------------------

    def set_status(self, name: str, status: str) -> None:
        """Force a signal's status ('active' / 'decayed' / 'dead')."""
        self._conn.execute(
            "UPDATE signal_library SET status = ?, updated_at = date('now') WHERE name = ?",
            (status, name),
        )
        self._conn.commit()

    def sync_active(self, active_names: list[str]) -> dict[str, list[str]]:
        """Reconcile the library against the hypothesis book's active set.

        Signals the book still trades are registered (weight 1.0 if new);
        signals it no longer trades — pruned, retired, errored — are marked
        dead so they stop drawing weight and stop appearing in reports.

        Returns {'added': [...], 'retired': [...]}.
        """
        active = set(active_names)
        known = {
            r[0]: r[1]
            for r in self._conn.execute("SELECT name, status FROM signal_library").fetchall()
        }

        added = sorted(active - set(known))
        for name in added:
            self.add_signal(name)

        retired = sorted(n for n, st in known.items() if n not in active and st != "dead")
        for name in retired:
            self.set_status(name, "dead")

        if added or retired:
            logger.info("Library sync: +%d added, %d retired", len(added), len(retired))
        return {"added": added, "retired": retired}

    # ------------------------------------------------------------------
    # Weight updates
    # ------------------------------------------------------------------

    def update_weights(
        self,
        alpha_returns: dict[str, pd.Series],
        as_of: date,
    ) -> None:
        """Update Darwinian weights based on 63-day rolling alpha Sharpe.

        alpha_returns: {signal_name: daily_alpha_return_series}
        as_of: end date of the rolling window (usually today)
        """
        window_end = pd.Timestamp(as_of)
        window_start = window_end - pd.offsets.BDay(_ROLLING_WINDOW)

        for name, ret in alpha_returns.items():
            if ret.empty:
                continue

            window_ret = ret[(ret.index >= window_start) & (ret.index <= window_end)]
            if len(window_ret) < 10:
                continue

            rolling_sr = annualized_sharpe(window_ret)
            new_weight = float(
                np.clip(rolling_sr / _TARGET_SHARPE, _WEIGHT_FLOOR, _WEIGHT_CEILING)
            )

            row = self._conn.execute(
                "SELECT weight, days_at_floor, status FROM signal_library WHERE name = ?",
                (name,),
            ).fetchone()

            if row is None:
                self.add_signal(name)
                days_at_floor, prev_status = 0, "active"
            else:
                _, days_at_floor, prev_status = row

            # Only advance the floor counter once per calendar date. The nightly
            # job can be re-run (retry, manual invocation) and must not age a
            # signal toward death several times for the same day.
            already_counted = self._conn.execute(
                "SELECT 1 FROM weight_history WHERE signal_name = ? AND as_of_date = ?",
                (name, as_of.isoformat()),
            ).fetchone() is not None

            if new_weight <= _WEIGHT_FLOOR + 1e-6:
                if not already_counted:
                    days_at_floor += 1
            else:
                days_at_floor = 0

            if days_at_floor >= _DEATH_DAYS:
                new_status = "dead"
            elif days_at_floor >= _DECAY_DAYS:
                new_status = "decayed"
            else:
                new_status = "active"

            self._conn.execute(
                """UPDATE signal_library
                   SET weight = ?, days_at_floor = ?, status = ?, updated_at = ?
                   WHERE name = ?""",
                (new_weight, days_at_floor, new_status, as_of.isoformat(), name),
            )
            self._conn.execute(
                """INSERT OR REPLACE INTO weight_history
                       (signal_name, as_of_date, weight, rolling_sharpe)
                   VALUES (?, ?, ?, ?)""",
                (name, as_of.isoformat(), new_weight, rolling_sr),
            )

            if new_status == "dead" and prev_status != "dead":
                logger.info("Signal '%s' declared dead after %d days at floor", name, days_at_floor)
            elif new_status == "decayed" and prev_status == "active":
                logger.info("Signal '%s' decayed after %d days at floor", name, days_at_floor)

        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SignalLibrary":
        return self

    def __exit__(self, *_) -> None:
        self.close()
