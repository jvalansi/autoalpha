import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_llm_news as ev
import score_press_releases as sp


def test_releases_only_before_reaction_open():
    rows = [{"date": "2025-01-29 16:30:00", "title": "X Reports Q4 Results"},              # prior evening: in
            {"date": "2025-01-30 09:29:00", "title": "X Reports Fourth Quarter Revenue"},  # pre-open: in
            {"date": "2025-01-30 09:00:00", "title": "X Declares Quarterly Dividend"},     # not results: out
            {"date": "2025-01-30 08:00:00", "title": "X to Report Q1 Results on May 1"},   # scheduling: out
            {"date": "2025-01-30 09:30:00", "title": "X Quarter Results"},                 # at the open: out
            {"date": "2025-01-29 07:00:00", "title": "X Quarter Results"},                 # priced on prev bar: out
            {"date": "bad", "title": "X Quarter Results"}]                                 # unparseable: skipped
    stamps = pd.to_datetime([r["date"] for r in rows], errors="coerce")  # same parse as the loader
    cache = {"X": [(ts, r) for ts, r in zip(stamps, rows) if pd.notna(ts)]}
    got = sp.releases_for("X", pd.Timestamp("2025-01-30"), pd.Timestamp("2025-01-29"), cache)
    assert [r["title"] for r in got] == ["X Reports Fourth Quarter Revenue", "X Reports Q4 Results"]


def test_results_title_filter():
    assert sp.is_results_title("CHORD ENERGY REPORTS FIRST QUARTER 2026 RESULTS AND DECLARES BASE DIVIDEND")
    assert sp.is_results_title("AXON REPORTS Q2 2024 REVENUE OF $504 MILLION")
    assert not sp.is_results_title("AES RESCHEDULES FOURTH QUARTER & FULL YEAR 2025 FINANCIAL REVIEW CONFERENCE CALL")
    assert not sp.is_results_title("NVIDIA SETS CONFERENCE CALL FOR THIRD-QUARTER FINANCIAL RESULTS")
    assert not sp.is_results_title("X ANNOUNCES DATE OF SECOND QUARTER EARNINGS RELEASE")


def test_outcomes_are_abnormal_open_to_open_over_horizon(monkeypatch):
    monkeypatch.setattr(ev, "HORIZON", 2)
    dates = pd.bdate_range("2025-01-02", periods=5)
    a = [10.0, 11.0, 12.1, 13.31, 14.641]   # +10%/bar
    b = [20.0, 20.0, 20.0, 20.0, 20.0]      # flat
    bars = pd.DataFrame({
        "ticker": ["A"] * 5 + ["B"] * 5, "date": list(dates) * 2,
        "Open": a + b, "Close": [x * 1.01 for x in a] + b,
        "gap_1d": 0.0, "earnings_surprise": 0.0, "revenue_surprise": 0.0})
    out = ev.event_outcomes(bars).set_index(["ticker", "date"])
    # A over open(d0)->open(d2): +21%; equal-weight benchmark: (1.05)^2 - 1 = 10.25%
    assert round(out.loc[("A", dates[0]), "drift0"], 6) == round(0.21 - 0.1025, 6)
    # entering a bar late uses open(d1)->open(d3), same numbers here
    assert round(out.loc[("A", dates[0]), "drift1"], 6) == round(0.21 - 0.1025, 6)
    # no future bar -> NaN, never a partial window
    assert pd.isna(out.loc[("A", dates[3]), "drift0"])
    # intraday: A +1% open->close vs benchmark mean (+1%, 0%) = 0.5%; first bar has no prior close -> NaN bench
    assert round(out.loc[("A", dates[1]), "intraday"], 6) == 0.005
