import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_live_pead import on_today_session, today_bar

TODAY = pd.Timestamp("2026-10-22")
STAMP = int(pd.Timestamp("2026-10-22 09:31", tz="America/New_York").timestamp())
CLOSES, OPEN = [106.81, 106.59], 106.98  # LNN 10/20 and 10/21 closes, 10/21 open


def quote(**kw):
    return {"price": 111.86, "open": 111.86, "previousClose": 106.59, "dayHigh": 112.0, "dayLow": 111.0,
            "timestamp": STAMP, **kw}


def test_uses_todays_session():
    assert today_bar(quote(), CLOSES, OPEN, TODAY) == {"Open": 111.86, "High": 112.0, "Low": 111.0, "Close": 111.86}


def test_rejects_quote_still_on_yesterdays_session_even_after_a_small_move():
    # yesterday moved only 0.2%; a stale quote's previousClose is the day before's close
    assert today_bar(quote(previousClose=106.81, open=OPEN), CLOSES, OPEN, TODAY) is None


def test_flat_two_days_falls_back_on_the_open():
    flat = [12.5, 12.5]
    assert today_bar(quote(previousClose=12.5, open=12.4), flat, 12.4, TODAY) is None  # yesterday's open
    assert today_bar(quote(previousClose=12.5, open=13.1), flat, 12.4, TODAY) is not None


def test_rejects_bad_print_and_stale_or_missing_fields():
    assert today_bar(quote(previousClose=94.0), [80.0, 100.0], OPEN, TODAY) is None  # >5% off
    assert today_bar(quote(timestamp=STAMP - 86400), CLOSES, OPEN, TODAY) is None
    assert today_bar(quote(timestamp=None), CLOSES, OPEN, TODAY) is None
    assert today_bar(None, CLOSES, OPEN, TODAY) is None


def test_held_fallback_needs_todays_session_but_not_an_open():
    assert today_bar(quote(open=None), CLOSES, OPEN, TODAY) is None
    assert on_today_session(quote(open=None), CLOSES, OPEN, TODAY)
    assert not on_today_session(quote(open=None, previousClose=106.81), CLOSES, OPEN, TODAY)
