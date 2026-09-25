import numpy as np
import pandas as pd

from autoalpha.core.executors import SimExecutor
from autoalpha.strategies.ml_cross_section import MLCrossSection, forward_returns


def _panel(n_tickers=100, n_days=160, seed=0):
    """Synthetic panel where roe fully determines each stock's daily drift."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    roe = rng.normal(size=n_tickers)
    drift = 0.002 * roe
    opens = 50 * np.exp(np.cumsum(drift + rng.normal(0, 0.001, (n_days, n_tickers)), axis=0))
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    df = pd.DataFrame({"Open": opens.ravel(), "Close": opens.ravel(), "roe": np.tile(roe, n_days)}, index=idx)
    return df, pd.Series(roe, index=tickers)


def test_forward_return_starts_at_next_open():
    df, _ = _panel(n_tickers=2, n_days=30)
    opens = df["Open"].unstack()
    fwd = forward_returns(df, horizon=5).unstack()
    assert np.isclose(fwd.iloc[0, 0], opens.iloc[6, 0] / opens.iloc[1, 0] - 1)
    assert fwd.iloc[-6:].isna().all().all()  # no label without 6 future opens


def test_learns_signal_and_holds_top_decile_monthly():
    df, roe = _panel()
    s = MLCrossSection()
    s.fit(df)
    d0, d1 = pd.Timestamp("2020-06-01"), pd.Timestamp("2020-06-02")
    bars = df.xs(df.index.levels[0][-1], level="date")
    picks = s.predict(bars, d0)
    assert len(picks) == 10 and set(picks) <= set(roe.nlargest(25).index)  # top quartile
    assert s.predict(bars.iloc[::-1], d1) == picks  # same month: targets unchanged


def test_hold_unchanged_skips_drift_rebalancing():
    prices = [{"A": 100.0, "B": 100.0}, {"A": 110.0, "B": 100.0}]
    trades = {}
    for hold in (False, True):
        ex = SimExecutor(initial_capital=1000, cost_bps=0, max_weight=None, hold_unchanged=hold)
        ex.execute({"A": 0.5, "B": 0.5}, pd.Timestamp("2024-01-02").date(), prices[0])
        ex.execute({"A": 0.5, "B": 0.5}, pd.Timestamp("2024-01-03").date(), prices[1])
        trades[hold] = ex._positions["A"]
    assert trades[True] == 5.0            # still the original 5 shares
    assert trades[False] < 5.0            # trimmed back to 50%
