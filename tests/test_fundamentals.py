import numpy as np
import pandas as pd

from autoalpha.data.fundamentals import FUND_COLS, pit_fundamentals


def test_pit_fundamentals_uses_filing_date():
    table = pd.DataFrame({
        "ticker": ["A", "A", "A"],
        "period_date": pd.to_datetime(["2024-03-31", "2024-06-30", "2023-12-31"]),
        "filing_date": pd.to_datetime(["2024-05-01", "2024-08-01", "2024-09-01"]),  # last: late amendment
        **{c: [1.0, 2.0, 99.0] for c in FUND_COLS},
    })
    bars = pd.DataFrame({"date": pd.to_datetime(["2024-04-01", "2024-05-01", "2024-05-02", "2024-08-02",
                                                 "2024-09-03"]),
                         "ticker": ["A"] * 5})
    roe = pit_fundamentals(bars, table)["roe"].tolist()
    # Nothing before the Q1 filing; visible the bar after filing; amendment of an older quarter ignored
    assert np.isnan(roe[0]) and np.isnan(roe[1])
    assert roe[2:] == [1.0, 2.0, 2.0]
