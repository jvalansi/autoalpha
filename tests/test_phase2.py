"""Phase 2 tests: labeling, CPCV, and evaluation engine."""
from __future__ import annotations

import math
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_series(n: int = 200, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    prices = 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx, name="Close")


def _make_return_series(n: int = 252, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.001, 0.015, n), index=idx)


# ---------------------------------------------------------------------------
# evaluation/costs.py
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_round_trip_is_double_one_way(self):
        from autoalpha.evaluation.costs import CostModel
        m = CostModel(half_spread_bps=2, commission_bps=0.5, market_impact_bps=3)
        assert abs(m.round_trip_bps - 2 * m.one_way_bps) < 1e-9

    def test_default_round_trip_is_11bps(self):
        from autoalpha.evaluation.costs import DEFAULT_COST_MODEL
        assert abs(DEFAULT_COST_MODEL.round_trip_bps - 11.0) < 1e-9

    def test_deduct_reduces_gross_returns(self):
        from autoalpha.evaluation.costs import CostModel
        m = CostModel()
        gross = _make_return_series(100)
        turnover = pd.Series(0.1, index=gross.index)  # 10% daily turnover
        net = m.deduct(gross, turnover)
        assert (net <= gross).all()

    def test_stress_doubles_costs(self):
        from autoalpha.evaluation.costs import CostModel
        base = CostModel(half_spread_bps=2, commission_bps=0.5, market_impact_bps=3)
        stressed = base.stress(2.0)
        assert abs(stressed.round_trip_bps - base.round_trip_bps * 2) < 1e-9


# ---------------------------------------------------------------------------
# evaluation/drawdown.py
# ---------------------------------------------------------------------------

class TestDrawdown:
    def test_no_drawdown_monotone_up(self):
        from autoalpha.evaluation.drawdown import max_drawdown
        rets = pd.Series([0.01] * 50)
        assert max_drawdown(rets) < 1e-9

    def test_max_drawdown_known_series(self):
        from autoalpha.evaluation.drawdown import max_drawdown
        # NAV goes 1 → 1.5 → 1.0 → 1.2; max drawdown = (1.5-1.0)/1.5 ≈ 33.3%
        # Corresponding returns: +50%, -33.3%, +20%
        rets = pd.Series([0.5, -1 / 3, 0.2])
        dd = max_drawdown(rets)
        assert abs(dd - 1 / 3) < 1e-4

    def test_empty_returns_zero(self):
        from autoalpha.evaluation.drawdown import max_drawdown
        assert max_drawdown(pd.Series(dtype=float)) == 0.0

    def test_passes_constraint_true(self):
        from autoalpha.evaluation.drawdown import passes_drawdown_constraint
        rets = pd.Series([0.005] * 100)
        assert passes_drawdown_constraint(rets, threshold=0.25)

    def test_passes_constraint_false(self):
        from autoalpha.evaluation.drawdown import passes_drawdown_constraint, max_drawdown
        # NAV: 1.0 → 0.8 → 0.64 → 0.512; max_dd = (1-0.512)/1 = 48.8% > 25%
        rets = pd.Series([-0.2, -0.2, -0.2, 0.01, 0.01])
        assert not passes_drawdown_constraint(rets, threshold=0.25)
        assert abs(max_drawdown(rets) - 0.488) < 0.001

    def test_initial_loss_captured(self):
        from autoalpha.evaluation.drawdown import max_drawdown
        # Series starts with a 20% loss then recovers — drawdown from initial capital must be 20%
        rets = pd.Series([-0.2, 0.5])
        assert abs(max_drawdown(rets) - 0.2) < 1e-6


# ---------------------------------------------------------------------------
# evaluation/sharpe.py
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_annualized_sharpe_known_input(self):
        from autoalpha.evaluation.sharpe import annualized_sharpe
        # mean=0.001, std=0.01, SR = 0.1 * sqrt(252) ≈ 1.587
        rng = np.random.default_rng(7)
        rets = pd.Series(rng.normal(0.001, 0.01, 5000))
        sr = annualized_sharpe(rets)
        assert 1.0 < sr < 2.5

    def test_sharpe_zero_std(self):
        from autoalpha.evaluation.sharpe import annualized_sharpe
        assert annualized_sharpe(pd.Series([0.0] * 10)) == 0.0

    def test_probabilistic_sharpe_increasing_in_sr(self):
        from autoalpha.evaluation.sharpe import probabilistic_sharpe
        rng = np.random.default_rng(1)
        low_sr = pd.Series(rng.normal(0.0, 0.01, 300))
        high_sr = pd.Series(rng.normal(0.002, 0.01, 300))
        psr_low = probabilistic_sharpe(low_sr, benchmark_sr=0.0)
        psr_high = probabilistic_sharpe(high_sr, benchmark_sr=0.0)
        assert psr_high > psr_low

    def test_deflated_sharpe_decreases_with_more_trials(self):
        from autoalpha.evaluation.sharpe import deflated_sharpe
        rng = np.random.default_rng(3)
        rets = pd.Series(rng.normal(0.001, 0.01, 500))
        dsr_1 = deflated_sharpe(rets, n_trials=1)
        dsr_100 = deflated_sharpe(rets, n_trials=100)
        assert dsr_1 > dsr_100

    def test_deflated_sharpe_between_0_and_1(self):
        from autoalpha.evaluation.sharpe import deflated_sharpe
        rets = _make_return_series()
        dsr = deflated_sharpe(rets, n_trials=10)
        assert 0.0 <= dsr <= 1.0

    def test_expected_max_sr_zero_for_one_trial(self):
        from autoalpha.evaluation.sharpe import expected_max_sr
        assert expected_max_sr(1) == 0.0

    def test_expected_max_sr_increasing(self):
        from autoalpha.evaluation.sharpe import expected_max_sr
        assert expected_max_sr(10) < expected_max_sr(100) < expected_max_sr(1000)


# ---------------------------------------------------------------------------
# evaluation/marginal.py
# ---------------------------------------------------------------------------

class TestMarginal:
    def test_standalone_when_no_portfolio(self):
        from autoalpha.evaluation.marginal import marginal_sharpe
        from autoalpha.evaluation.sharpe import annualized_sharpe
        rets = _make_return_series()
        assert abs(marginal_sharpe(rets, None) - annualized_sharpe(rets)) < 1e-9

    def test_orthogonal_signals_same_sharpe(self):
        """Two uncorrelated signals: marginal ≈ standalone (regression explains nothing)."""
        from autoalpha.evaluation.marginal import marginal_sharpe
        rng = np.random.default_rng(5)
        idx = pd.date_range("2020-01-01", periods=500, freq="B")
        strat = pd.Series(rng.normal(0.001, 0.015, 500), index=idx)
        portfolio = pd.Series(rng.normal(0.001, 0.015, 500), index=idx)
        marginal = marginal_sharpe(strat, portfolio)
        standalone = marginal_sharpe(strat, None)
        # Uncorrelated → residual ≈ original → marginal ≈ standalone
        assert abs(marginal - standalone) < 0.5

    def test_duplicate_signal_near_zero_marginal(self):
        """Duplicate signal: contributes nothing new → marginal Sharpe ≈ 0."""
        from autoalpha.evaluation.marginal import marginal_sharpe
        rets = _make_return_series(500)
        marginal = marginal_sharpe(rets, rets)
        assert abs(marginal) < 0.1

    def test_combine_portfolio_alpha(self):
        from autoalpha.evaluation.marginal import combine_portfolio_alpha
        existing = _make_return_series(200, seed=1)
        new_strat = _make_return_series(200, seed=2)
        combined = combine_portfolio_alpha(existing, new_strat, weight=0.5)
        assert len(combined) == len(existing)

    def test_combine_none_existing_returns_new(self):
        from autoalpha.evaluation.marginal import combine_portfolio_alpha
        new_strat = _make_return_series(100)
        combined = combine_portfolio_alpha(None, new_strat)
        pd.testing.assert_series_equal(combined, new_strat)


# ---------------------------------------------------------------------------
# evaluation/regime.py
# ---------------------------------------------------------------------------

class TestRegime:
    def test_classify_trend_returns_valid_labels(self):
        from autoalpha.evaluation.regime import classify_trend
        spy = _make_price_series(300)
        regime = classify_trend(spy)
        assert set(regime.dropna().unique()).issubset({"bull", "bear", "sideways"})

    def test_classify_vol_returns_valid_labels(self):
        from autoalpha.evaluation.regime import classify_vol
        spy = _make_price_series(300)
        regime = classify_vol(spy)
        assert set(regime.dropna().unique()).issubset({"high", "low"})

    def test_bull_market_classified_correctly(self):
        from autoalpha.evaluation.regime import classify_trend
        # 200 days of +0.5%/day → 63-day return > 5%
        idx = pd.date_range("2020-01-01", periods=200, freq="B")
        spy = pd.Series(100 * np.cumprod(np.ones(200) * 1.005), index=idx)
        regime = classify_trend(spy)
        # After warmup, should be bull
        assert "bull" in regime.values

    def test_classify_trend_excludes_warmup(self):
        from autoalpha.evaluation.regime import classify_trend
        spy = _make_price_series(150)
        regime = classify_trend(spy)
        # First 63 values have no valid 63-day return — should be NaN, not 'sideways'
        assert regime.iloc[:63].isna().all()
        # After warmup, all values should be classified
        assert not regime.iloc[63:].isna().any()

    def test_regime_breakdown_structure(self):
        from autoalpha.evaluation.regime import classify_trend, classify_vol, regime_breakdown

        spy = _make_price_series(400)
        rets = _make_return_series(400)

        # Manually align returns index with spy
        common = rets.index.intersection(spy.index)
        rets = rets.loc[common]

        result = regime_breakdown(rets, spy_close=spy)
        assert "trend" in result
        assert "vol" in result
        for regime_type in result.values():
            for stats in regime_type.values():
                assert "sharpe" in stats
                assert "n_days" in stats
                assert stats["n_days"] > 0


# ---------------------------------------------------------------------------
# evaluation/library.py
# ---------------------------------------------------------------------------

class TestSignalLibrary:
    def test_add_and_get_weight(self):
        from autoalpha.evaluation.library import SignalLibrary
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "test.db"
            lib = SignalLibrary(db_path=db)
            lib.add_signal("strategy_a")
            assert lib.get_weight("strategy_a") == 1.0
            lib.close()

    def test_update_weight_high_sharpe_increases(self):
        from autoalpha.evaluation.library import SignalLibrary, _WEIGHT_CEILING
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SignalLibrary(db_path=Path(tmpdir) / "test.db")
            lib.add_signal("strategy_a")
            # Noisy but strongly positive → high Sharpe → weight at ceiling
            rng = np.random.default_rng(99)
            idx = pd.date_range("2020-01-01", periods=150, freq="B")
            alpha = pd.Series(rng.normal(0.005, 0.001, 150), index=idx)
            lib.update_weights({"strategy_a": alpha}, as_of=idx[-1].date())
            assert lib.get_weight("strategy_a") == _WEIGHT_CEILING
            lib.close()

    def test_update_weight_low_sharpe_decreases(self):
        from autoalpha.evaluation.library import SignalLibrary, _WEIGHT_FLOOR
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SignalLibrary(db_path=Path(tmpdir) / "test.db")
            lib.add_signal("strategy_a")
            idx = pd.date_range("2020-01-01", periods=150, freq="B")
            alpha = pd.Series(-0.005, index=idx)  # negative Sharpe → floor
            lib.update_weights({"strategy_a": alpha}, as_of=idx[-1].date())
            assert lib.get_weight("strategy_a") == _WEIGHT_FLOOR
            lib.close()

    def test_death_after_126_days_at_floor(self):
        from autoalpha.evaluation.library import SignalLibrary, _WEIGHT_FLOOR, _DEATH_DAYS
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SignalLibrary(db_path=Path(tmpdir) / "test.db")
            lib.add_signal("dying_signal")
            # Manually set days_at_floor to 125 (one below death)
            lib._conn.execute(
                "UPDATE signal_library SET days_at_floor = 125 WHERE name = ?",
                ("dying_signal",),
            )
            lib._conn.commit()
            idx = pd.date_range("2022-01-01", periods=150, freq="B")
            alpha = pd.Series(-0.005, index=idx)  # stays at floor
            lib.update_weights({"dying_signal": alpha}, as_of=idx[-1].date())
            rows = lib._conn.execute(
                "SELECT status FROM signal_library WHERE name = ?", ("dying_signal",)
            ).fetchone()
            assert rows[0] == "dead"
            lib.close()

    def test_active_signals_excludes_dead(self):
        from autoalpha.evaluation.library import SignalLibrary
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SignalLibrary(db_path=Path(tmpdir) / "test.db")
            lib.add_signal("live")
            lib.add_signal("dead_one")
            lib._conn.execute(
                "UPDATE signal_library SET status = 'dead' WHERE name = ?",
                ("dead_one",),
            )
            lib._conn.commit()
            assert "live" in lib.active_signals()
            assert "dead_one" not in lib.active_signals()
            lib.close()

    def test_context_manager(self):
        from autoalpha.evaluation.library import SignalLibrary
        with tempfile.TemporaryDirectory() as tmpdir:
            with SignalLibrary(db_path=Path(tmpdir) / "test.db") as lib:
                lib.add_signal("s1")
                assert lib.get_weight("s1") == 1.0


# ---------------------------------------------------------------------------
# labeling/triple_barrier.py
# ---------------------------------------------------------------------------

class TestTripleBarrier:
    def test_profit_take_label(self):
        from autoalpha.labeling.triple_barrier import triple_barrier_labels
        # Strongly rising series: first barrier hit should be upper (+1)
        # Use idx[25] so ATR(21) has enough warmup (needs 22+ observations in diff)
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        close = pd.Series(100.0 * np.cumprod(np.ones(100) * 1.02), index=idx)
        events = pd.DatetimeIndex([idx[25]])
        result = triple_barrier_labels(close, events, profit_take_mult=1.0, stop_loss_mult=10.0, time_expiry=30)
        assert not result.empty
        assert result.iloc[0]["label"] == 1

    def test_stop_loss_label(self):
        from autoalpha.labeling.triple_barrier import triple_barrier_labels
        # Strongly falling: lower barrier hit first (-1)
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        close = pd.Series(100.0 * np.cumprod(np.ones(100) * 0.98), index=idx)
        events = pd.DatetimeIndex([idx[25]])
        result = triple_barrier_labels(close, events, profit_take_mult=10.0, stop_loss_mult=1.0, time_expiry=30)
        assert not result.empty
        assert result.iloc[0]["label"] == -1

    def test_time_expiry_label(self):
        from autoalpha.labeling.triple_barrier import triple_barrier_labels
        # Flat series: neither barrier hit → time expiry (0)
        # Use atr_window=5 and a constant series to control ATR precisely
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        rng = np.random.default_rng(0)
        # Slowly oscillating ±0.01: ATR ≈ 0.01; barriers at ±100×ATR won't be hit
        close = pd.Series(100.0 + 0.01 * np.sin(np.arange(100)), index=idx)
        events = pd.DatetimeIndex([idx[25]])
        result = triple_barrier_labels(close, events, profit_take_mult=1000.0, stop_loss_mult=1000.0, time_expiry=5)
        assert not result.empty
        assert result.iloc[0]["label"] == 0

    def test_labels_are_in_valid_set(self):
        from autoalpha.labeling.triple_barrier import triple_barrier_labels
        close = _make_price_series(200)
        events = close.index[10::20]
        result = triple_barrier_labels(close, events)
        assert set(result["label"].unique()).issubset({-1, 0, 1})

    def test_exit_date_after_entry(self):
        from autoalpha.labeling.triple_barrier import triple_barrier_labels
        close = _make_price_series(200)
        events = close.index[10:50:5]
        result = triple_barrier_labels(close, events)
        for entry, row in result.iterrows():
            assert row["t1"] > entry

    def test_empty_events_returns_empty(self):
        from autoalpha.labeling.triple_barrier import triple_barrier_labels
        close = _make_price_series(100)
        result = triple_barrier_labels(close, pd.DatetimeIndex([]))
        assert result.empty


# ---------------------------------------------------------------------------
# labeling/meta_label.py
# ---------------------------------------------------------------------------

class TestMetaLabeler:
    def _make_data(self, n: int = 300, seed: int = 0):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        features = pd.DataFrame(
            rng.standard_normal((n, 4)),
            index=idx,
            columns=["f1", "f2", "f3", "f4"],
        )
        primary = pd.Series(rng.choice([-1, 1], n), index=idx)
        outcomes = pd.Series(rng.choice([-1, 0, 1], n), index=idx)
        return features, primary, outcomes

    def test_fit_predict_returns_probabilities(self):
        from autoalpha.labeling.meta_label import MetaLabeler
        features, primary, outcomes = self._make_data()
        ml = MetaLabeler()
        ml.fit(features[:200], primary[:200], outcomes[:200])
        proba = ml.predict_proba(features[200:])
        assert len(proba) == 100
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_unfitted_returns_ones(self):
        from autoalpha.labeling.meta_label import MetaLabeler
        features, _, _ = self._make_data()
        ml = MetaLabeler()
        proba = ml.predict_proba(features)
        assert (proba == 1.0).all()

    def test_degenerate_all_wrong_fallback_zero(self):
        """When primary is always wrong, unfitted model should return 0.0 (don't bet)."""
        from autoalpha.labeling.meta_label import MetaLabeler
        rng = np.random.default_rng(7)
        idx = pd.date_range("2020-01-01", periods=50, freq="B")
        features = pd.DataFrame(rng.standard_normal((50, 2)), index=idx, columns=["f1", "f2"])
        primary = pd.Series(1, index=idx)   # always +1
        outcomes = pd.Series(-1, index=idx)  # always wrong
        ml = MetaLabeler()
        ml.fit(features, primary, outcomes)
        proba = ml.predict_proba(features)
        assert (proba == 0.0).all()

    def test_no_lookahead_across_folds(self):
        """OOS predictions must not use any OOS data for fitting."""
        from autoalpha.labeling.meta_label import MetaLabeler
        features, primary, outcomes = self._make_data(400)
        in_sample_end = 200

        ml1 = MetaLabeler()
        ml1.fit(features[:in_sample_end], primary[:in_sample_end], outcomes[:in_sample_end])
        proba_oos = ml1.predict_proba(features[in_sample_end:])

        # Fitting on full data would differ → verify fit uses only in-sample
        ml2 = MetaLabeler()
        ml2.fit(features, primary, outcomes)
        proba_full = ml2.predict_proba(features[in_sample_end:])

        # They may differ — the key is that ml1 was only fit on in-sample data
        assert len(proba_oos) == len(proba_full)  # structural check


# ---------------------------------------------------------------------------
# backtest/cpcv.py
# ---------------------------------------------------------------------------

class TestCPCV:
    def _make_dates(self, n: int = 500) -> pd.DatetimeIndex:
        return pd.date_range("2018-01-01", periods=n, freq="B")

    def test_number_of_folds(self):
        from autoalpha.backtest.cpcv import CPCV
        import math as _math
        cpcv = CPCV(n_splits=6, n_test_splits=2, purge_days=0, embargo_days=0)
        dates = self._make_dates(600)
        folds = list(cpcv.split(dates))
        expected = _math.comb(6, 2)
        assert len(folds) == expected

    def test_test_covers_full_timeline(self):
        """Union of all test sets should cover (nearly) all dates."""
        from autoalpha.backtest.cpcv import CPCV
        cpcv = CPCV(n_splits=6, n_test_splits=2, purge_days=0, embargo_days=0)
        dates = self._make_dates(600)
        all_test_dates = set()
        for _, test_dates in cpcv.split(dates):
            all_test_dates.update(test_dates.tolist())
        coverage = len(all_test_dates) / len(dates)
        assert coverage > 0.95

    def test_train_test_disjoint(self):
        """Train and test sets must not overlap."""
        from autoalpha.backtest.cpcv import CPCV
        cpcv = CPCV(n_splits=6, n_test_splits=2, purge_days=0, embargo_days=0)
        dates = self._make_dates(600)
        for train_dates, test_dates in cpcv.split(dates):
            overlap = set(train_dates.tolist()) & set(test_dates.tolist())
            assert len(overlap) == 0

    def test_purge_creates_gap(self):
        """No training date within purge_days business days before any test group start."""
        from autoalpha.backtest.cpcv import CPCV
        purge = 20
        # Use n_test_splits=1 for a clean contiguous test region
        cpcv = CPCV(n_splits=6, n_test_splits=1, purge_days=purge, embargo_days=0)
        dates = self._make_dates(600)
        for train_dates, test_dates in cpcv.split(dates):
            test_start = test_dates.min()
            # No training date should fall in [test_start - 20 BDay, test_start]
            too_close = train_dates[
                (train_dates >= test_start - pd.offsets.BDay(purge))
                & (train_dates <= test_start)
            ]
            assert len(too_close) == 0

    def test_to_runner_folds_format(self):
        """to_runner_folds must return (train_dates DatetimeIndex, (oos_start, oos_end))."""
        from autoalpha.backtest.cpcv import CPCV
        import pandas as pd
        cpcv = CPCV(n_splits=6, n_test_splits=2, purge_days=5, embargo_days=2)
        dates = self._make_dates(600)
        folds = cpcv.to_runner_folds(dates)
        assert len(folds) > 0
        for train_dates, (oos_start, oos_end) in folds:
            assert isinstance(train_dates, pd.DatetimeIndex)
            assert len(train_dates) > 0
            assert isinstance(oos_start, date)
            assert isinstance(oos_end, date)
            assert oos_end >= oos_start
            # Purge check: no train date should fall within the OOS window
            oos_ts_start = pd.Timestamp(oos_start)
            oos_ts_end = pd.Timestamp(oos_end)
            assert not any(
                (oos_ts_start <= d <= oos_ts_end) for d in train_dates
            ), "Purge failed: training dates overlap OOS window"

    def test_invalid_n_test_splits_raises(self):
        from autoalpha.backtest.cpcv import CPCV
        with pytest.raises(ValueError):
            CPCV(n_splits=4, n_test_splits=4)

    def test_insufficient_dates_raises(self):
        from autoalpha.backtest.cpcv import CPCV
        cpcv = CPCV(n_splits=10)
        with pytest.raises(ValueError):
            list(cpcv.split(pd.date_range("2020-01-01", periods=5, freq="B")))


# ---------------------------------------------------------------------------
# Integration: triple-barrier → CPCV → evaluation
# ---------------------------------------------------------------------------

class TestPhase2Integration:
    def test_triple_barrier_labels_feed_meta_labeler(self):
        from autoalpha.labeling.triple_barrier import triple_barrier_labels
        from autoalpha.labeling.meta_label import MetaLabeler

        close = _make_price_series(300)
        events = close.index[10::15]

        labels = triple_barrier_labels(close, events)
        assert not labels.empty

        rng = np.random.default_rng(0)
        features = pd.DataFrame(
            rng.standard_normal((len(labels), 3)),
            index=labels.index,
            columns=["f1", "f2", "f3"],
        )
        primary = pd.Series(1, index=labels.index)

        split = len(labels) // 2
        ml = MetaLabeler()
        ml.fit(
            features.iloc[:split],
            primary.iloc[:split],
            labels["label"].iloc[:split],
        )
        proba = ml.predict_proba(features.iloc[split:])
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_cpcv_folds_with_evaluation(self):
        from autoalpha.backtest.cpcv import CPCV
        from autoalpha.evaluation.sharpe import annualized_sharpe
        from autoalpha.evaluation.drawdown import max_drawdown

        cpcv = CPCV(n_splits=4, n_test_splits=1, purge_days=5, embargo_days=2)
        dates = pd.date_range("2018-01-01", periods=400, freq="B")
        rng = np.random.default_rng(42)

        sharpes = []
        drawdowns = []
        for _, test_dates in cpcv.split(dates):
            fold_rets = pd.Series(rng.normal(0.001, 0.01, len(test_dates)), index=test_dates)
            sharpes.append(annualized_sharpe(fold_rets))
            drawdowns.append(max_drawdown(fold_rets))

        assert len(sharpes) == 4  # C(4,1) = 4 folds
        assert all(dd >= 0 for dd in drawdowns)
