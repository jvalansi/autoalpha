import numpy as np
import pandas as pd

from autoalpha.data.event_features import event_features


def test_event_features():
    dates = pd.bdate_range("2024-01-01", periods=15)
    bars = pd.DataFrame({
        "date": list(dates) * 2,
        "ticker": ["A"] * 15 + ["B"] * 15,
        "Open": [10.0] * 30,
        "Close": [float(i + 1) for i in range(15)] * 2,
        "sector": ["Technology"] * 15 + ["Martian"] * 15,
    }).sample(frac=1, random_state=0)  # row order must not matter
    calendar = pd.DataFrame({
        "ticker": ["A", "A", "B", "B"],
        "date": pd.to_datetime(["2023-12-29", "2024-01-03", "2024-01-05", "2024-01-06"]),
        "time": ["amc", "bmo", "amc", ""],  # B's second report falls on a Saturday
    })
    sector_closes = pd.DataFrame({"XLK": np.arange(1.0, 16.0), "SPY": 100.0}, index=dates)

    out = event_features(bars, calendar, sector_closes)
    a = out[bars["ticker"] == "A"].set_index(bars.loc[bars["ticker"] == "A", "date"]).sort_index()
    b = out[bars["ticker"] == "B"].set_index(bars.loc[bars["ticker"] == "B", "date"]).sort_index()

    assert np.isnan(a["gap_1d"].iloc[0]) and a["gap_1d"].iloc[1] == 10.0 / 1.0 - 1
    assert a["ret_10d"].iloc[10] == 11.0 / 1.0 - 1 and np.isnan(a["ret_10d"].iloc[9])
    assert a["sector_ret_1d"].iloc[2] == 3.0 / 2.0 - 1
    assert (b["sector_ret_1d"].iloc[1:] == 0).all()  # unknown sector falls back to SPY
    # A: pre-history report ignored; bmo on Jan 3 reacts that day (bar 2)
    assert a["days_since_earnings"].iloc[:2].isna().all()
    assert list(a["days_since_earnings"].iloc[2:5]) == [0, 1, 2]
    # B: amc on Fri Jan 5 reacts Mon Jan 8 (bar 5); the Saturday report also reacts Mon
    assert b["days_since_earnings"].iloc[:5].isna().all()
    assert list(b["days_since_earnings"].iloc[5:7]) == [0, 1]
