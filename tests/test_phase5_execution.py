"""Phase 5 tests: alpha measurement, position sizing, live executor, weight sync."""
from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# evaluation/alpha.py
# ---------------------------------------------------------------------------

class TestBenchmarkAlpha:
    def test_pure_beta_has_no_alpha(self):
        from autoalpha.evaluation.alpha import compute_benchmark_alpha
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=250, freq="B")
        bm = pd.Series(rng.normal(0.0005, 0.01, 250), index=idx)
        # Levered benchmark plus zero-mean idiosyncratic noise: beta 1.5, no alpha
        strat = bm * 1.5 + pd.Series(rng.normal(0.0, 0.003, 250), index=idx)

        stats = compute_benchmark_alpha(strat, bm)
        assert stats["available"]
        assert stats["beta"] == pytest.approx(1.5, abs=0.05)
        assert abs(stats["alpha_t"]) < 2.0  # indistinguishable from zero

    def test_exact_beta_replica_reports_no_evidence(self):
        from autoalpha.evaluation.alpha import compute_benchmark_alpha
        rng = np.random.default_rng(7)
        idx = pd.date_range("2024-01-01", periods=120, freq="B")
        bm = pd.Series(rng.normal(0.0005, 0.01, 120), index=idx)
        stats = compute_benchmark_alpha(bm * 1.5, bm)  # zero residual variance
        assert stats["beta"] == pytest.approx(1.5, abs=1e-6)
        assert stats["alpha_t"] == 0.0

    def test_constant_edge_shows_positive_alpha(self):
        from autoalpha.evaluation.alpha import compute_benchmark_alpha
        rng = np.random.default_rng(1)
        idx = pd.date_range("2024-01-01", periods=250, freq="B")
        bm = pd.Series(rng.normal(0.0005, 0.01, 250), index=idx)
        # 10bps/day of alpha on top of beta 1, with idiosyncratic noise
        strat = bm + 0.001 + pd.Series(rng.normal(0.0, 0.002, 250), index=idx)

        stats = compute_benchmark_alpha(strat, bm)
        assert stats["alpha_daily"] == pytest.approx(0.001, abs=3e-4)
        assert stats["alpha_t"] > 5
        assert stats["beta"] == pytest.approx(1.0, abs=0.05)
        assert stats["information_ratio"] > 3

    def test_residual_series_carries_the_alpha(self):
        from autoalpha.evaluation.alpha import compute_benchmark_alpha
        from autoalpha.evaluation.sharpe import annualized_sharpe
        rng = np.random.default_rng(2)
        idx = pd.date_range("2024-01-01", periods=250, freq="B")
        bm = pd.Series(rng.normal(0.0005, 0.01, 250), index=idx)
        strat = bm * 0.2 + 0.001 + pd.Series(rng.normal(0.0, 0.002, 250), index=idx)

        stats = compute_benchmark_alpha(strat, bm)
        resid = stats["residuals"]
        # Mean of the returned series IS the alpha — a mean-zero residual would
        # make every rolling-Sharpe weight update score zero.
        assert resid.mean() == pytest.approx(stats["alpha_daily"], abs=1e-12)
        assert annualized_sharpe(resid) == pytest.approx(stats["information_ratio"], rel=1e-6)

    def test_short_overlap_reports_unavailable(self):
        from autoalpha.evaluation.alpha import compute_benchmark_alpha
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        s = pd.Series(0.001, index=idx)
        stats = compute_benchmark_alpha(s, s)
        assert stats["available"] is False
        assert stats["n_overlap"] == 5

    def test_ff5_falls_back_to_raw_when_factors_empty(self):
        from autoalpha.evaluation.alpha import compute_alpha_returns
        idx = pd.date_range("2024-01-01", periods=60, freq="B")
        rets = pd.Series(0.001, index=idx)
        alpha, ann = compute_alpha_returns(rets, factors=pd.DataFrame())
        pd.testing.assert_series_equal(alpha, rets)
        assert np.isnan(ann)


# ---------------------------------------------------------------------------
# execution/sizer.py
# ---------------------------------------------------------------------------

class TestSizer:
    def test_kelly_leverage_neutral_on_short_sample(self):
        from autoalpha.execution.sizer import kelly_leverage
        assert kelly_leverage(pd.Series([0.01] * 5)) == 1.0
        assert kelly_leverage(pd.Series(dtype=float)) == 1.0

    def test_kelly_leverage_clips_negative_edge_to_zero(self):
        from autoalpha.execution.sizer import kelly_leverage
        rng = np.random.default_rng(3)
        losing = pd.Series(rng.normal(-0.002, 0.01, 100))
        assert kelly_leverage(losing) == 0.0

    def test_kelly_leverage_respects_ceiling(self):
        from autoalpha.execution.sizer import kelly_leverage, MAX_KELLY_LEVERAGE
        strong = pd.Series(np.full(100, 0.002))  # zero variance is guarded
        assert kelly_leverage(strong) == 1.0
        rng = np.random.default_rng(4)
        good = pd.Series(rng.normal(0.005, 0.005, 100))
        assert kelly_leverage(good) == MAX_KELLY_LEVERAGE

    def test_quarter_kelly_scaling(self):
        from autoalpha.execution.sizer import size_position
        # 0.10 raw × 0.25 kelly × 1.0 lev × 1.0 weight = 0.025
        assert size_position(0.10) == pytest.approx(0.025)

    def test_position_cap_binds(self):
        from autoalpha.execution.sizer import size_position, MAX_POSITION
        assert size_position(1.0, kelly_lev=2.0, darwinian_weight=2.5) == MAX_POSITION

    def test_low_meta_confidence_skips_trade(self):
        from autoalpha.execution.sizer import size_position
        assert size_position(0.10, meta_confidence=0.4) == 0.0

    def test_darwinian_weight_scales_position(self):
        from autoalpha.execution.sizer import size_position
        base = size_position(0.02, darwinian_weight=1.0)
        floored = size_position(0.02, darwinian_weight=0.3)
        assert floored == pytest.approx(base * 0.3)

    def test_combine_sums_overlaps_then_caps(self):
        from autoalpha.execution.sizer import PositionSizer, MAX_POSITION
        sizer = PositionSizer()
        book = sizer.combine({"a": {"AAPL": 0.03}, "b": {"AAPL": 0.04, "MSFT": 0.01}})
        assert book["AAPL"] == MAX_POSITION  # 0.07 summed, capped at 0.05
        assert book["MSFT"] == pytest.approx(0.01)

    def test_combine_scales_down_to_gross_cap(self):
        from autoalpha.execution.sizer import PositionSizer
        sizer = PositionSizer(max_gross=0.10)
        book = sizer.combine({str(i): {f"T{i}": 0.05} for i in range(5)})
        assert sum(book.values()) == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# core/executors.py — LiveExecutor reconciliation
# ---------------------------------------------------------------------------

class FakeBroker:
    """LiveExecutor subclass with an in-memory broker, for reconciliation tests."""

    def __init__(self, equity=100_000.0, positions=None, **kwargs):
        from autoalpha.core.executors import LiveExecutor

        class _Impl(LiveExecutor):
            def account_equity(_self):
                return equity

            def current_positions(_self):
                return dict(positions or {})

            def _place_order(_self, symbol, qty, side):
                _self.placed.append((symbol, qty, side))
                return {"id": f"{symbol}-{side}"}

        self.impl = _Impl(**kwargs)
        self.impl.placed = []


class TestLiveExecutor:
    def test_opens_position_in_whole_shares(self):
        broker = FakeBroker(equity=100_000)
        # 5% of 100k = 5000 / 100 = 50 shares
        broker.impl.execute({"AAPL": 0.05}, date(2026, 1, 5), {"AAPL": 100.0})
        assert broker.impl.placed == [("AAPL", 50.0, "buy")]

    def test_rounds_down_fractional_shares(self):
        broker = FakeBroker(equity=10_000)
        # 5% of 10k = 500 / 300 = 1.67 shares → 1 whole share
        broker.impl.execute({"AAPL": 0.05}, date(2026, 1, 5), {"AAPL": 300.0})
        assert broker.impl.placed == [("AAPL", 1.0, "buy")]

    def test_closes_position_absent_from_targets(self):
        broker = FakeBroker(equity=100_000, positions={"MSFT": 20.0})
        broker.impl.execute({}, date(2026, 1, 5), {"MSFT": 50.0})
        assert broker.impl.placed == [("MSFT", 20.0, "sell")]

    def test_closes_position_without_a_price(self):
        # A halted/delisted name has no quote but must still be exited.
        broker = FakeBroker(equity=100_000, positions={"XYZ": 7.0})
        broker.impl.execute({}, date(2026, 1, 5), {})
        assert broker.impl.placed == [("XYZ", 7.0, "sell")]

    def test_trims_oversized_position(self):
        broker = FakeBroker(equity=100_000, positions={"AAPL": 80.0})
        broker.impl.execute({"AAPL": 0.05}, date(2026, 1, 5), {"AAPL": 100.0})
        assert broker.impl.placed == [("AAPL", 30.0, "sell")]

    def test_skips_sub_one_share_delta(self):
        broker = FakeBroker(equity=100_000, positions={"AAPL": 50.0})
        broker.impl.execute({"AAPL": 0.0505}, date(2026, 1, 5), {"AAPL": 100.0})
        assert broker.impl.placed == []

    def test_skips_ticker_without_price(self):
        broker = FakeBroker(equity=100_000)
        broker.impl.execute({"AAPL": 0.05}, date(2026, 1, 5), {})
        assert broker.impl.placed == []

    def test_overlay_scales_targets(self):
        broker = FakeBroker(equity=100_000, overlay=0.5)
        broker.impl.execute({"AAPL": 0.05}, date(2026, 1, 5), {"AAPL": 100.0})
        assert broker.impl.placed == [("AAPL", 25.0, "buy")]

    def test_dry_run_places_nothing_but_records(self):
        broker = FakeBroker(equity=100_000, dry_run=True)
        broker.impl.execute({"AAPL": 0.05}, date(2026, 1, 5), {"AAPL": 100.0})
        assert broker.impl.placed == []
        assert broker.impl.orders()[0]["dry_run"] is True

    def test_returns_from_nav_history(self):
        from autoalpha.core.executors import LiveExecutor

        equities = iter([100_000.0, 101_000.0])

        class _Impl(LiveExecutor):
            def account_equity(self):
                return next(equities)

            def current_positions(self):
                return {}

            def _place_order(self, symbol, qty, side):
                return {}

        ex = _Impl()
        ex.execute({}, date(2026, 1, 5), {})
        ex.execute({}, date(2026, 1, 6), {})
        rets = ex.returns()
        assert len(rets) == 1
        assert rets.iloc[0] == pytest.approx(0.01)


class TestAlpacaExecutor:
    def test_requires_credentials(self, monkeypatch):
        from autoalpha.execution.alpaca import AlpacaExecutor, BrokerError
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(BrokerError):
            AlpacaExecutor()

    def test_defaults_to_paper_endpoint(self, monkeypatch):
        from autoalpha.execution.alpaca import AlpacaExecutor
        monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
        ex = AlpacaExecutor(api_key="k", secret_key="s")
        assert ex.is_paper

    def test_rejects_fractional_opg(self):
        from autoalpha.execution.alpaca import AlpacaExecutor
        with pytest.raises(ValueError):
            AlpacaExecutor(api_key="k", secret_key="s", allow_fractional=True)

    def test_client_error_is_not_retried(self):
        from autoalpha.execution.alpaca import AlpacaExecutor, BrokerError

        class _Resp:
            status_code = 422
            text = "insufficient buying power"
            content = b"x"

        calls = []

        class _Session:
            def request(self, method, url, **kwargs):
                calls.append(url)
                return _Resp()

        ex = AlpacaExecutor(api_key="k", secret_key="s", session=_Session())
        with pytest.raises(BrokerError):
            ex.account_equity()
        assert len(calls) == 1  # no retry on a 4xx rejection

    def test_order_payload_is_market_on_open(self):
        from autoalpha.execution.alpaca import AlpacaExecutor

        sent = {}

        class _Resp:
            status_code = 200
            content = b"{}"

            def json(self):
                return {"id": "order-1"}

        class _Session:
            def request(self, method, url, **kwargs):
                sent.update({"method": method, "url": url, "json": kwargs.get("json")})
                return _Resp()

        ex = AlpacaExecutor(api_key="k", secret_key="s", session=_Session())
        ex._place_order("AAPL", 10.0, "buy")
        assert sent["json"] == {
            "symbol": "AAPL", "qty": "10", "side": "buy",
            "type": "market", "time_in_force": "opg",
        }


# ---------------------------------------------------------------------------
# evaluation/library.py — sync + idempotence
# ---------------------------------------------------------------------------

class TestLibrarySync:
    def test_sync_adds_new_and_retires_missing(self):
        from autoalpha.evaluation.library import SignalLibrary
        with tempfile.TemporaryDirectory() as tmpdir:
            with SignalLibrary(db_path=Path(tmpdir) / "t.db") as lib:
                lib.add_signal("keeper")
                lib.add_signal("pruned")
                result = lib.sync_active(["keeper", "fresh"])
                assert result == {"added": ["fresh"], "retired": ["pruned"]}
                assert set(lib.all_weights()) == {"keeper", "fresh"}

    def test_rerun_same_day_does_not_double_age_signal(self):
        from autoalpha.evaluation.library import SignalLibrary
        with tempfile.TemporaryDirectory() as tmpdir:
            with SignalLibrary(db_path=Path(tmpdir) / "t.db") as lib:
                lib.add_signal("dud")
                idx = pd.date_range("2026-01-01", periods=40, freq="B")
                losing = pd.Series(-0.005, index=idx)
                as_of = idx[-1].date()

                lib.update_weights({"dud": losing}, as_of=as_of)
                lib.update_weights({"dud": losing}, as_of=as_of)  # nightly retry

                days = lib._conn.execute(
                    "SELECT days_at_floor FROM signal_library WHERE name = 'dud'"
                ).fetchone()[0]
                assert days == 1

    def test_distinct_days_do_age_signal(self):
        from autoalpha.evaluation.library import SignalLibrary
        with tempfile.TemporaryDirectory() as tmpdir:
            with SignalLibrary(db_path=Path(tmpdir) / "t.db") as lib:
                lib.add_signal("dud")
                idx = pd.date_range("2026-01-01", periods=40, freq="B")
                losing = pd.Series(-0.005, index=idx)
                lib.update_weights({"dud": losing}, as_of=idx[-2].date())
                lib.update_weights({"dud": losing}, as_of=idx[-1].date())
                days = lib._conn.execute(
                    "SELECT days_at_floor FROM signal_library WHERE name = 'dud'"
                ).fetchone()[0]
                assert days == 2
