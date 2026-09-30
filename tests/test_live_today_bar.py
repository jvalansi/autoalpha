import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_live_pead import today_bar

TODAY = pd.Timestamp("2026-09-29")


def test_uses_todays_daily_bar():
    snap = {"latestTrade": {"p": 61.3},
            "dailyBar": {"t": "2026-09-29T04:00:00Z", "o": 60.4, "h": 61.0, "l": 60.0, "c": 61.0}}
    assert today_bar(snap, TODAY) == {"Open": 60.4, "High": 61.3, "Low": 60.0, "Close": 61.3}


def test_yesterdays_daily_bar_falls_back_to_last_trade():
    snap = {"latestTrade": {"p": 60.5},
            "dailyBar": {"t": "2026-09-28T04:00:00Z", "o": 57.2, "h": 58.7, "l": 56.3, "c": 56.5}}
    assert today_bar(snap, TODAY) == {"Open": 60.5, "High": 60.5, "Low": 60.5, "Close": 60.5}


def test_no_trade_is_skipped():
    assert today_bar({}, TODAY) is None
