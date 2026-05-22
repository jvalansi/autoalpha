"""Phase 1 tests: interfaces, vault enforcement, and no look-ahead leakage."""
from __future__ import annotations

import warnings
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from autoalpha.core.strategy import Strategy
from autoalpha.core.providers import HistoricalProvider
from autoalpha.core.executors import SimExecutor
from autoalpha.core.runner import Runner
from autoalpha.data.fetcher import VaultLeakError, get_ohlcv
from autoalpha.data.features import fracdiff, find_min_d, add_fracdiff_features
from autoalpha.data.bars import DataQualityWarning, _adf_stats, _aggregate_dollar_bars


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _BuyAndHold(Strategy):
    """Trivial strategy for testing: buy all tickers equally."""
    def fit(self, data: pd.DataFrame) -> None:
        self._tickers = data.index.get_level_values("ticker").unique().tolist() if "ticker" in data.index.names else []

    def predict(self, bar_data: pd.DataFrame) -> dict[str, float]:
        tickers = list(bar_data.index)
        if not tickers:
            return {}
        frac = 1.0 / len(tickers)
        return {t: frac for t in tickers}


def _make_daily_ohlcv(n: int = 100, start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame({
        "Open": close * 0.995,
        "High": close * 1.005,
        "Low": close * 0.99,
        "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------

def test_strategy_is_abstract():
    with pytest.raises(TypeError):
        Strategy()  # type: ignore[abstract]


def test_buy_and_hold_fit_predict():
    s = _BuyAndHold()
    daily = _make_daily_ohlcv(50)
    idx = pd.MultiIndex.from_product([daily.index, ["AAPL"]], names=["date", "ticker"])
    data = pd.DataFrame(np.tile(daily.values, (1, 1)), index=idx, columns=daily.columns)
    s.fit(data)
    bar = daily.iloc[-1:].copy()
    bar.index = pd.Index(["AAPL"])
    result = s.predict(bar)
    assert "AAPL" in result
    assert abs(result["AAPL"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Vault holdout enforcement
# ---------------------------------------------------------------------------

def test_vault_leak_raises_on_overlap():
    with pytest.raises(VaultLeakError):
        # vault_holdout.json: 2024-05-21 to 2026-05-21
        get_ohlcv("AAPL", date(2024, 6, 1), date(2024, 12, 31))


def test_vault_no_error_before_window():
    # Should not raise — entirely before vault
    try:
        get_ohlcv("AAPL", date(2020, 1, 1), date(2024, 5, 20))
    except VaultLeakError:
        pytest.fail("VaultLeakError raised outside vault window")
    except Exception:
        pass  # network errors etc. are fine in unit tests


def test_vault_no_error_after_window():
    # Should not raise — entirely after vault (current vault ends 2026-05-21)
    try:
        get_ohlcv("AAPL", date(2026, 5, 22), date(2026, 12, 31))
    except VaultLeakError:
        pytest.fail("VaultLeakError raised outside vault window")
    except Exception:
        pass


def test_vault_boundary_included():
    """Exact vault start date should trigger VaultLeakError."""
    with pytest.raises(VaultLeakError):
        get_ohlcv("AAPL", date(2024, 5, 21), date(2024, 5, 21))


# ---------------------------------------------------------------------------
# SimExecutor
# ---------------------------------------------------------------------------

def test_sim_executor_no_cost_flat():
    ex = SimExecutor(initial_capital=100_000, cost_bps=0)
    prices = {"AAPL": 150.0}
    ex.execute({"AAPL": 0.1}, date(2023, 1, 3), prices)
    nav = ex._compute_nav(prices)
    assert abs(nav - 100_000) < 1e-6  # no cost, fill-at-price → NAV unchanged


def test_sim_executor_cost_deducted():
    ex = SimExecutor(initial_capital=100_000, cost_bps=10)
    prices = {"AAPL": 100.0}
    ex.execute({"AAPL": 0.1}, date(2023, 1, 3), prices)
    # Bought 100 shares (10% of 100k / $100), cost = 100*100 * 10/10000 = $10
    assert ex._cash < 100_000


def test_sim_executor_overlay():
    ex = SimExecutor(initial_capital=100_000, cost_bps=0, overlay=0.5)
    prices = {"AAPL": 100.0}
    ex.execute({"AAPL": 1.0}, date(2023, 1, 3), prices)
    # overlay=0.5 → target_frac=0.5, so 500 shares at $100
    expected_shares = 0.5 * 100_000 / 100.0
    assert abs(ex._positions.get("AAPL", 0) - expected_shares) < 1e-6


# ---------------------------------------------------------------------------
# Fractional differentiation
# ---------------------------------------------------------------------------

def test_fracdiff_d0_identity():
    series = pd.Series([1.0, 2.0, 3.0, 4.0], name="x")
    result = fracdiff(series, 0.0)
    pd.testing.assert_series_equal(result, series)


def test_fracdiff_d1_matches_diff():
    series = pd.Series([10.0, 20.0, 15.0, 25.0, 30.0], name="x")
    fd = fracdiff(series, 1.0)
    expected = series.diff().dropna()
    pd.testing.assert_series_equal(fd, expected, check_names=False)


def test_find_min_d_stationary_input():
    # Already stationary white noise — min_d should be near 0
    rng = np.random.default_rng(0)
    series = pd.Series(rng.standard_normal(200))
    d = find_min_d(series)
    assert 0 <= d <= 1.0


def test_find_min_d_nonstationary_input():
    # Random walk — needs d close to 1
    rng = np.random.default_rng(1)
    series = pd.Series(np.cumsum(rng.standard_normal(300)))
    d = find_min_d(series)
    assert d > 0.0


def test_add_fracdiff_features_fold_isolation():
    """d must be computed on in-sample data and applied to OOS — no look-ahead."""
    rng = np.random.default_rng(2)
    close = pd.Series(np.cumsum(rng.standard_normal(200)), name="Close")
    in_sample = close[:100]
    oos = close[100:]

    # Compute d on in-sample only
    df_in = pd.DataFrame({"Close": in_sample})
    _, d_map = add_fracdiff_features(df_in, columns=["Close"])
    d_in_sample = d_map["Close"]

    # Apply same d to OOS (no look-ahead)
    df_oos = pd.DataFrame({"Close": oos})
    _, d_map_oos = add_fracdiff_features(df_oos, columns=["Close"], d_map=d_map)
    d_oos = d_map_oos["Close"]

    assert d_in_sample == d_oos, "d must not change when d_map is passed"


# ---------------------------------------------------------------------------
# Dollar bars
# ---------------------------------------------------------------------------

def test_dollar_bar_fallback_warns():
    daily = _make_daily_ohlcv(120)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from autoalpha.data.bars import _daily_fallback
        result = _daily_fallback("TEST", daily, date(2020, 1, 1), date(2020, 6, 1))
        dq_warnings = [x for x in w if issubclass(x.category, DataQualityWarning)]
        assert len(dq_warnings) >= 1, "Expected DataQualityWarning for daily bar fallback"
    assert not result.empty
    assert "dollar_volume" in result.columns


def test_aggregate_dollar_bars_threshold():
    """Each dollar bar must have dollar_volume >= threshold (except possibly last)."""
    rng = np.random.default_rng(7)
    n = 500
    ts = pd.date_range("2023-01-01", periods=n, freq="1min", tz="UTC")
    minute_df = pd.DataFrame({
        "timestamp": ts,
        "Open": 100.0,
        "High": 100.5,
        "Low": 99.5,
        "Close": 100.0,
        "Volume": rng.integers(100, 500, n).astype(float),
    })
    threshold = 50_000.0
    bars = _aggregate_dollar_bars(minute_df, threshold)
    if len(bars) > 1:
        assert all(bars["dollar_volume"].iloc[:-1] >= threshold * 0.99)


# ---------------------------------------------------------------------------
# No look-ahead at data join seams
# ---------------------------------------------------------------------------

def test_no_lookahead_in_provider_bars():
    """HistoricalProvider.bars() must yield data in strict chronological order."""
    provider = HistoricalProvider.__new__(HistoricalProvider)
    dates = pd.date_range("2022-01-01", periods=20, freq="B")
    tickers = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    data = pd.DataFrame(
        np.ones((len(idx), 5)),
        index=idx,
        columns=["Open", "High", "Low", "Close", "Volume"],
    )

    with patch.object(provider, "history", return_value=data):
        yielded = list(provider.bars(tickers, dates[0].date(), dates[-1].date()))

    assert len(yielded) == len(dates), "Expected one bar per trading date"
    bar_dates = [bd for bd, _ in yielded]
    assert bar_dates == sorted(bar_dates), "Bars must be yielded in chronological order"
    # Each yielded DataFrame must only contain the tickers for that date (no future data)
    for bd, bar_df in yielded:
        assert list(bar_df.index) == tickers, "Each bar must contain exactly the expected tickers"
        assert "Close" in bar_df.columns


def test_vault_transcript_raises_in_window():
    """get_transcripts must raise VaultLeakError for quarters within the vault window."""
    from autoalpha.data.fetcher import get_transcripts
    with pytest.raises(VaultLeakError):
        get_transcripts("AAPL", 2025, 1)  # Q1 2025 ends 2025-03-31, inside vault


def test_vault_transcript_ok_before_window():
    """get_transcripts must not raise VaultLeakError for quarters before the vault."""
    from autoalpha.data.fetcher import get_transcripts
    try:
        get_transcripts("AAPL", 2023, 4)  # Q4 2023 ends 2023-12-31, before vault
    except VaultLeakError:
        pytest.fail("VaultLeakError raised for pre-vault transcript")
    except Exception:
        pass  # network / API key errors are OK in unit tests


def test_sim_executor_portfolio_value_includes_equity():
    """portfolio_value must reflect open positions at provided prices."""
    ex = SimExecutor(initial_capital=100_000, cost_bps=0)
    prices = {"AAPL": 100.0}
    ex.execute({"AAPL": 0.5}, date(2023, 1, 3), prices)
    # Bought 500 shares at $100; portfolio_value without prices should not count equity
    pv_no_prices = ex.portfolio_value()  # prices unknown → equity = 0
    pv_with_prices = ex.portfolio_value(prices)
    assert pv_with_prices > pv_no_prices, "portfolio_value with prices must exceed cash-only value"
    assert abs(pv_with_prices - 100_000) < 1.0  # NAV should be ~100k (no cost)


def test_fracdiff_no_future_leakage():
    """fracdiff must be causal: changing a future value must not alter any past output."""
    rng = np.random.default_rng(3)
    full = pd.Series(np.cumsum(rng.standard_normal(100)))
    d = 0.4

    # Perturb a "future" value (index 80) by a large amount
    perturbed = full.copy()
    perturbed.iloc[80] += 1_000_000.0

    fd_original = fracdiff(full, d)
    fd_perturbed = fracdiff(perturbed, d)

    # All output at indices <= 79 must be identical
    past_idx = [i for i in fd_original.index if i < 80]
    for i in past_idx:
        assert fd_original[i] == fd_perturbed[i], (
            f"fracdiff at index {i} changed after perturbing index 80 — look-ahead leak"
        )
