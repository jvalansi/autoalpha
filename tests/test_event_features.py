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


def test_latest_surprise():
    from autoalpha.data.event_features import latest_surprise
    bars = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-09", "2024-01-09"]),
                         "ticker": ["A", "A", "A", "B"]})
    calendar = pd.DataFrame({
        "ticker": ["A", "A", "B"],
        "date": pd.to_datetime(["2024-01-03", "2024-01-09", "2024-01-02"]),
        "eps": [1.1, np.nan, 0.5], "epsEstimated": [1.0, 2.0, 0.0],  # A's 01-09 row not yet reported
        "revenue": [90.0, np.nan, 10.0], "revenueEstimated": [100.0, 5.0, 8.0],
    })
    out = latest_surprise(bars, calendar)
    assert np.isnan(out["earnings_surprise"].iloc[0])
    assert np.isclose(out["earnings_surprise"].iloc[1], 0.1) and np.isclose(out["revenue_surprise"].iloc[2], -0.1)
    assert np.isnan(out["earnings_surprise"].iloc[3]) and np.isclose(out["revenue_surprise"].iloc[3], 0.25)
