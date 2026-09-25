import pandas as pd

from autoalpha.evaluation.alpha import equal_weight_benchmark


def test_equal_weight_benchmark_is_open_to_open_and_skips_bad_prices():
    opens = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 0.0, 50.0]},
                         index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    bm = equal_weight_benchmark(opens)
    assert list(bm.index) == list(opens.index[1:])
    assert bm.round(6).tolist() == [0.10, 0.10]  # B's zero price is missing, not a -100% day


def test_benchmark_skips_names_below_price_floor():
    opens = pd.DataFrame({"A": [100.0, 110.0], "P": [1.0, 3.0]},
                         index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert equal_weight_benchmark(opens).round(6).tolist() == [0.10]  # P's +200% ignored


def test_ff5_recovers_beta_split_across_days():
    import numpy as np
    from autoalpha.evaluation.alpha import ff5_alpha_stats
    import autoalpha.evaluation.alpha as alpha_mod
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=500)
    f = pd.DataFrame(rng.normal(0, 0.01, (500, 5)), index=idx,
                     columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA"]).assign(RF=0.0)
    # open-to-open strategy: 60% of today's market move, 40% of yesterday's
    y = 0.6 * f["Mkt-RF"] + 0.4 * f["Mkt-RF"].shift(1) + rng.normal(0, 0.001, 500)
    orig = alpha_mod._fetch_ff5
    alpha_mod._fetch_ff5 = lambda: f
    try:
        st = ff5_alpha_stats(y.dropna())
    finally:
        alpha_mod._fetch_ff5 = orig
    assert abs(st["betas"]["Mkt-RF"] - 1.0) < 0.02
