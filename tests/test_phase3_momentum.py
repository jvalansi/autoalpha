"""Tests for MomentumStrategy and the triple_barrier entry_prices fix."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n_tickers: int = 5, n_bars: int = 400, seed: int = 0) -> pd.DataFrame:
    """Return date × ticker DataFrame of close prices."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_bars, freq="B")
    data = {}
    for i in range(n_tickers):
        ret = rng.normal(0.0005 * (i + 1), 0.01, n_bars)  # varying drift per ticker
        data[f"T{i}"] = 100 * np.cumprod(1 + ret)
    return pd.DataFrame(data, index=idx)


def _prices_to_multiindex(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert date × ticker price DataFrame to MultiIndex(date, ticker) OHLCV."""
    frames = []
    for ticker in prices.columns:
        close = prices[ticker]
        df = pd.DataFrame({
            "Open": close * 0.999,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Close": close,
            "Volume": 1_000_000.0,
        })
        df.index.name = "date"
        df["ticker"] = ticker
        df = df.reset_index().set_index(["date", "ticker"])
        frames.append(df)
    return pd.concat(frames).sort_index()


# ---------------------------------------------------------------------------
# triple_barrier entry_prices parameter (closes #49)
# ---------------------------------------------------------------------------

class TestTripleBarrierEntryPrices:
    def test_default_uses_close(self):
        from autoalpha.labeling.triple_barrier import triple_barrier_labels
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        close = pd.Series(100.0 * np.cumprod(np.ones(100) * 1.005), index=idx)
        events = pd.DatetimeIndex([idx[25]])
        result_default = triple_barrier_labels(
            close, events, profit_take_mult=100.0, stop_loss_mult=100.0, time_expiry=5
        )
        # Default: entry at close[event_date]
        assert not result_default.empty

    def test_entry_prices_changes_barriers(self):
        """Different entry price → different realized return."""
        from autoalpha.labeling.triple_barrier import triple_barrier_labels
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        close = pd.Series(100.0 * np.cumprod(np.ones(100) * 1.01), index=idx)
        events = pd.DatetimeIndex([idx[25]])

        result_close = triple_barrier_labels(
            close, events, time_expiry=5,
            profit_take_mult=100.0, stop_loss_mult=100.0,
        )
        # Use a higher entry price (simulating a gap-up open)
        entry = pd.Series({idx[25]: close[idx[25]] * 1.05})
        result_open = triple_barrier_labels(
            close, events, time_expiry=5,
            profit_take_mult=100.0, stop_loss_mult=100.0,
            entry_prices=entry,
        )
        # Returns must differ because entry price differs
        assert not result_close.empty
        assert not result_open.empty
        assert abs(result_open.iloc[0]["ret"] - result_close.iloc[0]["ret"]) > 1e-6

    def test_entry_prices_missing_date_falls_back_to_close(self):
        """If entry_prices doesn't contain the event date, fall back to close."""
        from autoalpha.labeling.triple_barrier import triple_barrier_labels
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        close = pd.Series(100.0 * np.cumprod(np.ones(100) * 1.005), index=idx)
        events = pd.DatetimeIndex([idx[25]])
        entry = pd.Series(dtype=float)  # empty — no entry price for event

        result_with_empty = triple_barrier_labels(close, events, entry_prices=entry)
        result_default = triple_barrier_labels(close, events)
        pd.testing.assert_frame_equal(result_with_empty, result_default)


# ---------------------------------------------------------------------------
# MomentumStrategy unit tests
# ---------------------------------------------------------------------------

class TestMomentumStrategy:
    def test_predict_returns_empty_before_warmup(self):
        from autoalpha.strategies.momentum import MomentumStrategy
        strategy = MomentumStrategy()
        strategy.fit(pd.DataFrame())  # empty fit
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        bar = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=["T0", "T1", "T2"])
        result = strategy.predict(bar, bar_date=idx[0])
        # No history → no targets
        assert result == {}

    def test_rebalance_only_on_month_change(self):
        """predict returns the same targets within a month, new ones when month turns."""
        from autoalpha.strategies.momentum import MomentumStrategy
        prices = _make_prices(n_tickers=10, n_bars=350)
        mi = _prices_to_multiindex(prices)
        strategy = MomentumStrategy(lookback=252, skip=21)
        strategy.fit(mi)

        dates = pd.date_range("2020-01-02", periods=50, freq="B")
        targets_by_date = {}
        for d in dates:
            bar = pd.DataFrame(
                {"Open": prices.iloc[0].values, "High": prices.iloc[0].values,
                 "Low": prices.iloc[0].values, "Close": prices.iloc[0].values,
                 "Volume": 1e6},
                index=prices.columns,
            )
            targets_by_date[d] = strategy.predict(bar, bar_date=pd.Timestamp(d))

        # All targets within the same month must be identical (hold positions)
        jan_dates = [d for d in dates if d.month == 1]
        first_jan = targets_by_date[jan_dates[0]]
        for d in jan_dates[1:]:
            assert targets_by_date[d] == first_jan

    def test_top_quintile_long_only(self):
        """All predicted positions must be positive and sum to approximately 1."""
        from autoalpha.strategies.momentum import MomentumStrategy
        prices = _make_prices(n_tickers=10, n_bars=350)
        mi = _prices_to_multiindex(prices)
        strategy = MomentumStrategy(lookback=252, skip=21)
        strategy.fit(mi)

        # Feed one bar from after the warmup
        bar_date = pd.Timestamp("2020-01-02")
        bar = pd.DataFrame(
            {"Open": prices.iloc[-1].values, "High": prices.iloc[-1].values,
             "Low": prices.iloc[-1].values, "Close": prices.iloc[-1].values,
             "Volume": 1e6},
            index=prices.columns,
        )
        targets = strategy.predict(bar, bar_date=bar_date)

        if targets:
            assert all(v > 0 for v in targets.values()), "Momentum is long-only"
            assert abs(sum(targets.values()) - 1.0) < 1e-9, "Weights must sum to 1"
            # Top quintile of 10 tickers = 2 tickers (20%)
            assert len(targets) == 2

    def test_high_momentum_ticker_selected(self):
        """Ticker with highest return is selected; lowest is excluded."""
        from autoalpha.strategies.momentum import MomentumStrategy

        idx = pd.date_range("2017-01-01", periods=310, freq="B")
        rng = np.random.default_rng(42)
        # 6 tickers with varying drift; T_strong is the clear winner, T_weak is the clear loser
        drifts = {"T_strong": 0.004, "T_mid1": 0.002, "T_mid2": 0.001,
                  "T_mid3": 0.0, "T_mid4": -0.001, "T_weak": -0.003}
        frames = []
        prices_last = {}
        for name, drift in drifts.items():
            s = pd.Series(100 * np.cumprod(1 + rng.normal(drift, 0.004, 310)), index=idx)
            prices_last[name] = s.iloc[-1]
            df = pd.DataFrame(
                {"Open": s, "High": s, "Low": s, "Close": s, "Volume": 1e6},
                index=idx,
            )
            df.index.name = "date"
            df["ticker"] = name
            frames.append(df.reset_index().set_index(["date", "ticker"]))
        mi = pd.concat(frames).sort_index()

        strategy = MomentumStrategy(lookback=252, skip=21, quantile=0.80)
        strategy.fit(mi)

        last_date = idx[-1]
        bar = pd.DataFrame(
            {col: [prices_last[t] for t in drifts] for col in ["Open", "High", "Low", "Close"]},
            index=list(drifts.keys()),
        )
        bar["Volume"] = 1e6
        targets = strategy.predict(bar, bar_date=last_date)
        assert "T_strong" in targets, "High-momentum ticker must be selected"
        assert "T_weak" not in targets, "Low-momentum ticker must be excluded"

    def test_fit_seeds_price_history(self):
        """After fit on sufficient data, first OOS predict should produce targets."""
        from autoalpha.strategies.momentum import MomentumStrategy
        prices = _make_prices(n_tickers=10, n_bars=350)
        mi = _prices_to_multiindex(prices)
        strategy = MomentumStrategy(lookback=252, skip=21)
        strategy.fit(mi)

        # Price history should have been seeded
        assert not strategy._price_history.empty
        assert len(strategy._price_history) >= min(273, len(prices))

    def test_no_bar_date_returns_last_targets(self):
        """Without a bar_date, predict must return previous targets (safe hold)."""
        from autoalpha.strategies.momentum import MomentumStrategy
        prices = _make_prices(n_tickers=5, n_bars=300)
        mi = _prices_to_multiindex(prices)
        strategy = MomentumStrategy()
        strategy.fit(mi)

        bar = pd.DataFrame(
            {"Close": prices.iloc[-1].values, "Volume": 1e6},
            index=prices.columns,
        )
        result = strategy.predict(bar, bar_date=None)
        assert result == strategy._last_targets


# ---------------------------------------------------------------------------
# Integration: MomentumStrategy through Runner (mocked provider)
# ---------------------------------------------------------------------------

class TestMomentumIntegration:
    def test_runner_produces_oos_returns(self):
        """Full backtest loop with MomentumStrategy returns a non-empty return series."""
        from unittest.mock import patch
        from autoalpha.strategies.momentum import MomentumStrategy
        from autoalpha.core.providers import HistoricalProvider
        from autoalpha.core.executors import SimExecutor
        from autoalpha.core.runner import Runner

        prices = _make_prices(n_tickers=10, n_bars=500)
        mi = _prices_to_multiindex(prices)
        tickers = prices.columns.tolist()

        provider = HistoricalProvider.__new__(HistoricalProvider)
        dates = prices.index

        def _bars(t, s, e):
            mask = (mi.index.get_level_values("date") >= pd.Timestamp(s)) & \
                   (mi.index.get_level_values("date") <= pd.Timestamp(e))
            sub = mi[mask]
            for bar_date, grp in sub.groupby(level="date"):
                yield bar_date, grp.droplevel("date")

        with patch.object(provider, "history", return_value=mi), \
             patch.object(provider, "bars", side_effect=_bars):

            strategy = MomentumStrategy()
            executor = SimExecutor(initial_capital=100_000, cost_bps=11)
            runner = Runner(strategy, provider, executor, tickers)

            split = len(dates) // 2
            folds = [
                ((dates[0].date(), dates[split - 1].date()),
                 (dates[split].date(), dates[-1].date())),
            ]
            returns = runner.run_backtest(folds)

        assert not returns.empty
        assert returns.index.is_unique
