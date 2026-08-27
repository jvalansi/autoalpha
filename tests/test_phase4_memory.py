"""Tests for HypothesisMemory — SQLite persistence layer."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from autoalpha.research.hypothesis import Hypothesis
from autoalpha.research.memory import HypothesisMemory

_VALID_KNOWLEDGE = (
    "Earnings surprise creates a persistent mispricing because analysts anchored "
    "to prior estimates are slow to revise, leading to a drift in price over the "
    "subsequent weeks as consensus adjusts to the new fundamental reality."
)

_HYP_KWARGS = dict(
    hypothesis="Stocks with positive earnings surprise outperform over 20 days.",
    reason="Analyst anchoring causes post-earnings drift.",
    concise_reason="earnings drift",
    knowledge=_VALID_KNOWLEDGE,
    predict_body="        return {}",
    cohort="quality",
)


@pytest.fixture
def mem(tmp_path):
    db = tmp_path / "test.db"
    m = HypothesisMemory(db_path=db)
    yield m
    m.close()


class TestTrialNumbering:
    def test_first_trial_is_one(self, mem):
        assert mem.next_trial_number() == 1

    def test_increments_after_store(self, mem):
        hyp = Hypothesis(**_HYP_KWARGS)
        mem.store_hypothesis(hyp, trial_number=1, cost_usd=0.01)
        assert mem.next_trial_number() == 2

    def test_current_count_zero_on_empty(self, mem):
        assert mem.current_trial_count() == 0

    def test_current_count_equals_max_trial(self, mem):
        hyp = Hypothesis(**_HYP_KWARGS)
        mem.store_hypothesis(hyp, trial_number=5, cost_usd=0.0)
        assert mem.current_trial_count() == 5


class TestHypothesisLifecycle:
    def test_store_returns_id(self, mem):
        hyp = Hypothesis(**_HYP_KWARGS)
        hyp_id = mem.store_hypothesis(hyp, trial_number=1, cost_usd=0.02)
        assert isinstance(hyp_id, int) and hyp_id > 0

    def test_initial_status_pending(self, mem):
        hyp = Hypothesis(**_HYP_KWARGS)
        hyp_id = mem.store_hypothesis(hyp, trial_number=1, cost_usd=0.0)
        row = mem._conn.execute(
            "SELECT status FROM hypotheses WHERE id = ?", (hyp_id,)
        ).fetchone()
        assert row[0] == "pending"

    def test_update_result_sets_metrics(self, mem):
        hyp = Hypothesis(**_HYP_KWARGS)
        hyp_id = mem.store_hypothesis(hyp, trial_number=1, cost_usd=0.0)
        mem.update_result(
            hyp_id,
            status="accepted",
            sharpe=1.2,
            dsr=0.97,
            max_drawdown=0.12,
            additional_cost_usd=0.03,
            observation="Strong across all folds.",
            justification="DSR > 0.95 and DD < 25%.",
        )
        row = mem._conn.execute(
            "SELECT status, sharpe, dsr, max_drawdown FROM hypotheses WHERE id = ?",
            (hyp_id,),
        ).fetchone()
        assert row[0] == "accepted"
        assert abs(row[1] - 1.2) < 1e-6
        assert abs(row[2] - 0.97) < 1e-6

    def test_update_result_persists_observation_in_json(self, mem):
        hyp = Hypothesis(**_HYP_KWARGS)
        hyp_id = mem.store_hypothesis(hyp, trial_number=1, cost_usd=0.0)
        mem.update_result(
            hyp_id,
            status="rejected",
            sharpe=0.3,
            dsr=0.4,
            max_drawdown=0.18,
            additional_cost_usd=0.0,
            observation="Weak signal in quality cohort.",
            justification="DSR < 0.95.",
        )
        row = mem._conn.execute(
            "SELECT hypothesis_json FROM hypotheses WHERE id = ?", (hyp_id,)
        ).fetchone()
        restored = Hypothesis.from_json(row[0])
        assert restored.observation == "Weak signal in quality cohort."
        assert restored.justification == "DSR < 0.95."

    def test_increment_refinement_count(self, mem):
        hyp = Hypothesis(**_HYP_KWARGS)
        hyp_id = mem.store_hypothesis(hyp, trial_number=1, cost_usd=0.0)
        mem.increment_refinement_count(hyp_id)
        mem.increment_refinement_count(hyp_id)
        row = mem._conn.execute(
            "SELECT refinement_count FROM hypotheses WHERE id = ?", (hyp_id,)
        ).fetchone()
        assert row[0] == 2

    def test_get_pending_refinement_returns_rejected(self, mem):
        hyp = Hypothesis(**_HYP_KWARGS)
        hyp_id = mem.store_hypothesis(hyp, trial_number=1, cost_usd=0.0)
        mem.update_result(
            hyp_id, status="rejected", sharpe=0.4, dsr=0.5, max_drawdown=0.12,
            additional_cost_usd=0.0,
        )
        pending = mem.get_pending_refinement(max_refinements=3)
        assert pending is not None
        assert pending["id"] == hyp_id

    def test_get_pending_refinement_includes_metrics(self, mem):
        hyp = Hypothesis(**_HYP_KWARGS)
        hyp_id = mem.store_hypothesis(hyp, trial_number=1, cost_usd=0.0)
        mem.update_result(
            hyp_id, status="rejected", sharpe=0.35, dsr=0.62, max_drawdown=0.18,
            additional_cost_usd=0.0,
        )
        pending = mem.get_pending_refinement(max_refinements=3)
        assert pending is not None
        assert abs(pending["sharpe"] - 0.35) < 1e-9
        assert abs(pending["dsr"] - 0.62) < 1e-9
        assert abs(pending["max_drawdown"] - 0.18) < 1e-9

    def test_get_pending_refinement_none_when_at_max(self, mem):
        hyp = Hypothesis(**_HYP_KWARGS)
        hyp_id = mem.store_hypothesis(hyp, trial_number=1, cost_usd=0.0)
        mem.update_status(hyp_id, "rejected")
        for _ in range(3):
            mem.increment_refinement_count(hyp_id)
        pending = mem.get_pending_refinement(max_refinements=3)
        assert pending is None

    def test_get_active_count(self, mem):
        assert mem.get_active_count() == 0
        hyp = Hypothesis(**_HYP_KWARGS)
        hyp_id = mem.store_hypothesis(hyp, trial_number=1, cost_usd=0.0)
        mem.update_status(hyp_id, "active")
        assert mem.get_active_count() == 1

    def test_accepted_status_not_counted_as_active(self, mem):
        # Verifies that 'accepted' (LLM output) != 'active' (DB status for live signals).
        # loop._accept() must call update_status(id, 'active') explicitly.
        hyp = Hypothesis(**_HYP_KWARGS)
        hyp_id = mem.store_hypothesis(hyp, trial_number=1, cost_usd=0.0)
        mem.update_result(
            hyp_id, status="accepted", sharpe=1.5, dsr=0.97, max_drawdown=0.10,
            additional_cost_usd=0.0,
        )
        assert mem.get_active_count() == 0  # 'accepted' ≠ 'active'
        mem.update_status(hyp_id, "active")
        assert mem.get_active_count() == 1


class TestPortfolioAlpha:
    def test_store_and_retrieve_alpha(self, mem):
        hyp = Hypothesis(**_HYP_KWARGS)
        hyp_id = mem.store_hypothesis(hyp, trial_number=1, cost_usd=0.0)
        mem.update_status(hyp_id, "active")

        alpha = pd.Series(
            [0.001, -0.002, 0.003],
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
        )
        mem.store_portfolio_alpha(hyp_id, alpha)

        combined = mem.get_portfolio_alpha()
        assert combined is not None
        assert len(combined) == 3

    def test_get_portfolio_alpha_none_when_no_active(self, mem):
        result = mem.get_portfolio_alpha()
        assert result is None

    def test_equal_weight_across_two_signals(self, mem):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        alpha1 = pd.Series([0.01] * 5, index=dates)
        alpha2 = pd.Series([0.02] * 5, index=dates)

        for i, alpha in enumerate([alpha1, alpha2], start=1):
            hyp = Hypothesis(**{**_HYP_KWARGS, "concise_reason": f"signal {i}"})
            hyp_id = mem.store_hypothesis(hyp, trial_number=i, cost_usd=0.0)
            mem.update_status(hyp_id, "active")
            mem.store_portfolio_alpha(hyp_id, alpha)

        combined = mem.get_portfolio_alpha()
        assert combined is not None
        # equal-weight mean of 0.01 and 0.02 = 0.015
        assert abs(combined.mean() - 0.015) < 1e-9


class TestContextManager:
    def test_context_manager(self, tmp_path):
        db = tmp_path / "ctx.db"
        with HypothesisMemory(db_path=db) as mem:
            assert mem.current_trial_count() == 0
        # Connection should be closed; re-opening should work fine
        with HypothesisMemory(db_path=db) as mem2:
            assert mem2.current_trial_count() == 0


class TestTrialReturnStorage:
    """Return series must be kept for every trial, not just the winners.

    Effective-N clustering — the only defensible way to shrink the Deflated
    Sharpe penalty — is impossible retroactively if rejects were discarded.
    """

    def _memory(self, tmpdir):
        from autoalpha.research.memory import HypothesisMemory
        return HypothesisMemory(db_path=Path(tmpdir) / "t.db")

    def _store(self, m, name="test signal"):
        from autoalpha.research.hypothesis import Hypothesis
        h = Hypothesis(
            hypothesis="Cheap high-quality firms with rising estimates outperform",
            reason="Analysts update estimates slowly after fundamental improvement",
            knowledge="Investors underreact to gradual fundamental improvement because the "
                      "information arrives with no salient event to trigger repricing, so the "
                      "adjustment appears over subsequent months rather than immediately",
            cohort="value",
            concise_reason=name,
            predict_body="return {}",
        )
        return m.store_hypothesis(h, trial_number=1, cost_usd=0.0)

    def test_round_trip_preserves_series(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmpdir:
            m = self._memory(tmpdir)
            hid = self._store(m)
            dates = pd.date_range("2020-01-01", periods=300, freq="B").strftime("%Y-%m-%d").tolist()
            rets = list(np.random.default_rng(0).normal(0.0004, 0.01, 300))
            m.store_trial_returns(hid, dates, rets)
            got = m.get_trial_returns(hid)
            assert len(got) == 300
            assert got.iloc[0] == pytest.approx(rets[0])
            assert got.index[0] == pd.Timestamp("2020-01-01")
            m.close()

    def test_empty_series_is_a_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = self._memory(tmpdir)
            hid = self._store(m)
            m.store_trial_returns(hid, [], [])
            assert m.get_trial_returns(hid) is None
            m.close()

    def test_missing_trial_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = self._memory(tmpdir)
            assert m.get_trial_returns(999) is None
            m.close()

    def test_rejected_trials_are_retrievable_for_clustering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = self._memory(tmpdir)
            keep, drop = self._store(m, "kept"), self._store(m, "dropped")
            dates = pd.date_range("2020-01-01", periods=40, freq="B").strftime("%Y-%m-%d").tolist()
            m.store_trial_returns(keep, dates, [0.001] * 40)
            m.store_trial_returns(drop, dates, [-0.001] * 40)
            m.update_status(drop, "rejected")

            everything = m.all_trial_returns()
            assert set(everything) == {keep, drop}
            assert set(m.all_trial_returns(statuses=["rejected"])) == {drop}
            m.close()

    def test_storing_twice_replaces_not_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = self._memory(tmpdir)
            hid = self._store(m)
            dates = pd.date_range("2020-01-01", periods=10, freq="B").strftime("%Y-%m-%d").tolist()
            m.store_trial_returns(hid, dates, [0.001] * 10)
            m.store_trial_returns(hid, dates, [0.002] * 10)
            assert m.get_trial_returns(hid).iloc[0] == pytest.approx(0.002)
            assert m._conn.execute("SELECT COUNT(*) FROM trial_returns").fetchone()[0] == 1
            m.close()
