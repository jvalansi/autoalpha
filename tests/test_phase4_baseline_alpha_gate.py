"""Tests for ResearchLoop._baseline_alpha_passes — the marginal-alpha gate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoalpha.research.hypothesis import Hypothesis
from autoalpha.research.loop import ResearchLoop, _min_alpha_t, _MIN_BASELINE_OVERLAP_DAYS
from autoalpha.research.subprocess_runner import BacktestResult


_KNOWLEDGE = (
    "Earnings surprise creates a persistent mispricing because analysts anchored "
    "to prior estimates are slow to revise, leading to a drift in price over the "
    "subsequent weeks as consensus adjusts to the new fundamental reality."
)

_HYP_KWARGS = dict(
    hypothesis="Stocks with positive earnings surprise outperform over 20 days.",
    reason="Analyst anchoring causes post-earnings drift.",
    concise_reason="earnings drift",
    knowledge=_KNOWLEDGE,
    predict_body="        return {}",
    cohort="quality",
)


@pytest.fixture
def loop(tmp_path):
    db = tmp_path / "test.db"
    rl = ResearchLoop(data_path="unused.parquet", db_path=db)
    yield rl
    rl._memory.close()


def _make_result(returns: list[float], dates: pd.DatetimeIndex) -> BacktestResult:
    return BacktestResult(
        sharpe=1.0,
        dsr=0.9,
        max_drawdown=0.1,
        activity_rate=1.0,
        returns=returns,
        return_dates=[d.strftime("%Y-%m-%d") for d in dates],
    )


def _seed_active_signal(loop: ResearchLoop, alpha: pd.Series) -> int:
    """Insert an active hypothesis with the given portfolio alpha series."""
    hyp = Hypothesis(**_HYP_KWARGS)
    hyp_id = loop._memory.store_hypothesis(hyp, trial_number=1, cost_usd=0.0)
    loop._memory.update_status(hyp_id, "active")
    loop._memory.store_portfolio_alpha(hyp_id, alpha)
    return hyp_id


class TestBaselineAlphaGate:
    def test_empty_book_bootstraps(self, loop):
        dates = pd.date_range("2024-01-02", periods=60, freq="B")
        result = _make_result([0.001] * 60, dates)
        passes, note = loop._baseline_alpha_passes(result)
        assert passes is True
        assert "bootstrap" in note

    def test_empty_candidate_passes(self, loop):
        result = _make_result([], pd.DatetimeIndex([]))
        passes, note = loop._baseline_alpha_passes(result)
        assert passes is True
        assert "no candidate" in note

    def test_thin_overlap_defers(self, loop):
        dates = pd.date_range("2024-01-02", periods=200, freq="B")
        rng = np.random.default_rng(0)
        _seed_active_signal(loop, pd.Series(rng.normal(0, 0.01, 200), index=dates))

        cand_dates = pd.date_range("2025-01-02", periods=20, freq="B")  # no overlap
        result = _make_result([0.001] * 20, cand_dates)
        passes, note = loop._baseline_alpha_passes(result)
        assert passes is True
        assert "insufficient" in note or "overlap" in note

    def test_redundant_signal_fails(self, loop):
        dates = pd.date_range("2024-01-02", periods=250, freq="B")
        rng = np.random.default_rng(42)
        book = pd.Series(rng.normal(0.0005, 0.01, 250), index=dates)
        _seed_active_signal(loop, book)

        # Candidate = scaled copy of book + tiny noise → high β, near-zero α
        noise = rng.normal(0, 0.0005, 250)
        cand_returns = (book.values * 1.05 + noise).tolist()
        result = _make_result(cand_returns, dates)
        passes, note = loop._baseline_alpha_passes(result)
        assert passes is False
        assert "redundant" in note
        # sanity: numeric t shown in note
        assert "t=" in note

    def test_orthogonal_positive_alpha_passes(self, loop):
        dates = pd.date_range("2024-01-02", periods=250, freq="B")
        rng = np.random.default_rng(7)
        book = pd.Series(rng.normal(0.0003, 0.01, 250), index=dates)
        _seed_active_signal(loop, book)

        # Candidate independent of book with strong positive drift
        cand = rng.normal(0.003, 0.008, 250)
        result = _make_result(cand.tolist(), dates)
        passes, note = loop._baseline_alpha_passes(result)
        assert passes is True
        assert "α=" in note

    def test_threshold_constants_sane(self):
        assert _MIN_BASELINE_OVERLAP_DAYS >= 20

    def test_threshold_scales_down_with_n(self):
        # Sparse book → strict bar; crowded book → relaxed bar; floor at 0.5
        assert _min_alpha_t(0) > _min_alpha_t(50) > _min_alpha_t(150)
        assert _min_alpha_t(0) <= 2.0
        assert _min_alpha_t(10_000) == 0.5  # floor

    def test_redundant_signal_still_fails_at_full_book(self, loop):
        # Confirm the gate still rejects clear duplicates even when the threshold
        # has decayed near its floor — duplicates have t ≪ 0, way below any floor.
        dates = pd.date_range("2024-01-02", periods=250, freq="B")
        rng = np.random.default_rng(99)
        book = pd.Series(rng.normal(0.0005, 0.01, 250), index=dates)
        _seed_active_signal(loop, book)

        noise = rng.normal(0, 0.0005, 250)
        cand_returns = (book.values * 1.05 + noise).tolist()
        result = _make_result(cand_returns, dates)
        passes, _ = loop._baseline_alpha_passes(result)
        assert passes is False
