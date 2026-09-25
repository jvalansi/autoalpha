import pandas as pd

from autoalpha.evaluation.alpha import equal_weight_benchmark


def test_equal_weight_benchmark_is_open_to_open_and_skips_bad_prices():
    opens = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 0.0, 50.0]},
                         index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    bm = equal_weight_benchmark(opens)
    assert list(bm.index) == list(opens.index[1:])
    assert bm.round(6).tolist() == [0.10, 0.10]  # B's zero price is missing, not a -100% day
