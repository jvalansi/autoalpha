import pandas as pd

from autoalpha.strategies.pead_et import EarningsTraderPEAD


def _bar(close, days=None, eps=0.1, rev=0.1, gap=0.05, ret_10d=0.05, ret_1d=0.04, sector=0.0):
    return pd.DataFrame({"High": [close + 1], "Low": [close - 1], "Close": [close], "days_since_earnings": [days],
                         "earnings_surprise": [eps], "revenue_surprise": [rev], "gap_1d": [gap],
                         "ret_10d": [ret_10d], "ret_1d": [ret_1d], "sector_ret_1d": [sector]}, index=["A"])


def test_entry_trailing_stop_and_time_exit():
    d = pd.bdate_range("2024-01-01", periods=30)
    s = EarningsTraderPEAD()
    assert s.predict(_bar(100, days=5), d[0]) == {}
    assert s.predict(_bar(100, days=0, gap=0.01), d[1]) == {}          # gap below 3%
    assert s.predict(_bar(100, days=0), d[2]) == {"A": 0.1}            # enters, stop = 100 - 2.5*ATR(2) = 95
    assert s.predict(_bar(110, days=1), d[3]) == {"A": 0.1}            # TR 11 -> ATR 2.64, stop trails to 103.4
    assert s.predict(_bar(103, days=2), d[4]) == {}                     # stop hit
    assert s.predict(_bar(100, days=0), d[5]) == {"A": 0.1}            # re-entry on a new event
    for i in range(6, 16):
        out = s.predict(_bar(100, days=i), d[i])
    assert out == {}                                                    # 10-bar time exit
