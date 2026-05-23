"""Persistent SQLite store for all hypothesis evaluations.

Coexists with the signal_library / weight_history tables used by SignalLibrary.
The trial_number column is monotonically increasing across all sessions and is
never reset — it feeds into deflated_sharpe() as the n_trials argument.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from autoalpha.research.hypothesis import Hypothesis

_DEFAULT_DB = Path(__file__).resolve().parents[2] / "research" / "memory.db"

_DDL = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    status            TEXT    NOT NULL DEFAULT 'pending',
    sharpe            REAL,
    dsr               REAL,
    max_drawdown      REAL,
    cost_usd          REAL    NOT NULL DEFAULT 0.0,
    trial_number      INTEGER NOT NULL,
    refinement_count  INTEGER NOT NULL DEFAULT 0,
    cohort            TEXT,
    hypothesis_json   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio_alpha (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id   INTEGER NOT NULL REFERENCES hypotheses(id),
    as_of_date      TEXT    NOT NULL,
    alpha_return    REAL    NOT NULL,
    UNIQUE(hypothesis_id, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status);
CREATE INDEX IF NOT EXISTS idx_hypotheses_trial  ON hypotheses(trial_number);
CREATE INDEX IF NOT EXISTS idx_portfolio_alpha_hyp ON portfolio_alpha(hypothesis_id);
"""


class HypothesisMemory:
    """Persistent store for hypothesis evaluations and portfolio alpha returns."""

    def __init__(self, db_path: Path = _DEFAULT_DB) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Trial numbering
    # ------------------------------------------------------------------

    def next_trial_number(self) -> int:
        """Return MAX(trial_number) + 1, or 1 if table is empty."""
        row = self._conn.execute("SELECT MAX(trial_number) FROM hypotheses").fetchone()
        current = row[0]
        return 1 if current is None else current + 1

    def current_trial_count(self) -> int:
        """Return MAX(trial_number), or 0 if no rows exist."""
        row = self._conn.execute("SELECT MAX(trial_number) FROM hypotheses").fetchone()
        return row[0] or 0

    # ------------------------------------------------------------------
    # Hypothesis lifecycle
    # ------------------------------------------------------------------

    def store_hypothesis(
        self,
        hypothesis: Hypothesis,
        trial_number: int,
        cost_usd: float,
    ) -> int:
        """Insert a pending hypothesis. Returns the new row id."""
        cur = self._conn.execute(
            """INSERT INTO hypotheses
               (status, cost_usd, trial_number, cohort, hypothesis_json)
               VALUES ('pending', ?, ?, ?, ?)""",
            (cost_usd, trial_number, hypothesis.cohort, hypothesis.to_json()),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_result(
        self,
        hyp_id: int,
        status: str,
        sharpe: float,
        dsr: float,
        max_drawdown: float,
        additional_cost_usd: float,
        observation: str = "",
        justification: str = "",
    ) -> None:
        """Update hypothesis after backtest + LLM interpretation."""
        # Merge observation/justification into stored JSON
        row = self._conn.execute(
            "SELECT hypothesis_json FROM hypotheses WHERE id = ?", (hyp_id,)
        ).fetchone()
        if row:
            hyp = Hypothesis.from_json(row["hypothesis_json"])
            hyp.observation = observation
            hyp.justification = justification
            updated_json = hyp.to_json()
        else:
            updated_json = None

        self._conn.execute(
            """UPDATE hypotheses
               SET status = ?, sharpe = ?, dsr = ?, max_drawdown = ?,
                   cost_usd = cost_usd + ?,
                   hypothesis_json = COALESCE(?, hypothesis_json)
               WHERE id = ?""",
            (status, sharpe, dsr, max_drawdown, additional_cost_usd, updated_json, hyp_id),
        )
        self._conn.commit()

    def update_status(self, hyp_id: int, status: str) -> None:
        self._conn.execute(
            "UPDATE hypotheses SET status = ? WHERE id = ?", (status, hyp_id)
        )
        self._conn.commit()

    def increment_refinement_count(self, hyp_id: int) -> None:
        self._conn.execute(
            "UPDATE hypotheses SET refinement_count = refinement_count + 1 WHERE id = ?",
            (hyp_id,),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Portfolio alpha
    # ------------------------------------------------------------------

    def store_portfolio_alpha(
        self,
        hypothesis_id: int,
        alpha_returns: pd.Series,
    ) -> None:
        """Bulk insert alpha returns for an accepted strategy."""
        rows = [
            (hypothesis_id, str(ts.date()), float(ret))
            for ts, ret in alpha_returns.items()
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO portfolio_alpha (hypothesis_id, as_of_date, alpha_return) VALUES (?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def get_portfolio_alpha(self) -> Optional[pd.Series]:
        """Return equal-weight combined alpha from all active hypotheses, or None."""
        ids = [
            r["id"]
            for r in self._conn.execute(
                "SELECT id FROM hypotheses WHERE status = 'active'"
            ).fetchall()
        ]
        if not ids:
            return None

        rows = self._conn.execute(
            f"SELECT as_of_date, alpha_return, hypothesis_id FROM portfolio_alpha "
            f"WHERE hypothesis_id IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall()
        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["date", "alpha_return", "hypothesis_id"])
        df["date"] = pd.to_datetime(df["date"])
        pivot = df.pivot_table(index="date", columns="hypothesis_id", values="alpha_return")
        return pivot.mean(axis=1)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all_for_trace(self) -> list[dict]:
        """Return all hypotheses ordered by trial_number for LLM trace."""
        rows = self._conn.execute(
            """SELECT trial_number, status, sharpe, dsr, cohort, hypothesis_json
               FROM hypotheses ORDER BY trial_number"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_pending_refinement(self, max_refinements: int = 3) -> Optional[dict]:
        """Return most-recent rejected hypothesis eligible for refinement, or None.

        Includes sharpe/dsr/max_drawdown so callers don't need to access _conn.
        """
        row = self._conn.execute(
            """SELECT id, refinement_count, hypothesis_json,
                      sharpe, dsr, max_drawdown
               FROM hypotheses
               WHERE status = 'rejected' AND refinement_count < ?
               ORDER BY trial_number DESC LIMIT 1""",
            (max_refinements,),
        ).fetchone()
        return dict(row) if row else None

    def get_active_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM hypotheses WHERE status = 'active'"
        ).fetchone()
        return row[0]

    def get_cohort_weight_summary(self, signal_weights: dict[str, float]) -> dict[str, float]:
        """Return {cohort: avg_darwinian_weight} for active hypotheses.

        signal_weights: {concise_reason: weight} from SignalLibrary.all_weights().
        """
        rows = self._conn.execute(
            """SELECT cohort, hypothesis_json FROM hypotheses WHERE status = 'active'"""
        ).fetchall()
        cohort_weights: dict[str, list[float]] = {}
        for row in rows:
            hyp = Hypothesis.from_json(row["hypothesis_json"])
            w = signal_weights.get(hyp.concise_reason, 1.0)
            cohort_weights.setdefault(row["cohort"], []).append(w)
        return {c: sum(ws) / len(ws) for c, ws in cohort_weights.items()}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "HypothesisMemory":
        return self

    def __exit__(self, *_) -> None:
        self.close()
