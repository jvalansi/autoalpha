import numpy as np
import pandas as pd

from autoalpha.strategies.etf_trend import ETFTrend, add_trend_features


def test_features_use_only_past_closes():
    closes = pd.DataFrame({"A": np.arange(1, 301, dtype=float)}, index=pd.bdate_range("2020-01-01", periods=300))
    f = add_trend_features(closes)
    changed = closes.copy()
    changed.iloc[-1] = 1e6  # a future spike must not move earlier features
    g = add_trend_features(changed)
    pd.testing.assert_frame_equal(f["trend"].iloc[:-1], g["trend"].iloc[:-1])
    assert f["trend"].iloc[:252].isna().all().all()


def test_inverse_vol_weights_and_cash_for_downtrends():
    bar = pd.DataFrame({"trend": [0.1, -0.2, 0.05], "vol": [0.10, 0.10, 0.20]}, index=["A", "B", "C"])
    s = ETFTrend()
    w = s.predict(bar, pd.Timestamp("2024-01-02"))
    assert set(w) == {"A", "C"}                     # B's trend is down → cash
    assert np.isclose(w["A"], 0.4) and np.isclose(w["C"], 0.2)   # 1/vol over all three: 10,10,5 of 25
    assert s.predict(bar.assign(trend=-1.0), pd.Timestamp("2024-01-03")) == w  # holds until next month
