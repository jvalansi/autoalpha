"""Tests for PEAD, Quality, EarningsRevisions, and EarningsNLP strategies."""
from __future__ import annotations

import math
from datetime import date
from unittest.mock import patch, MagicMock
from typing import Optional

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mi_ohlcv(tickers: list[str], n_bars: int = 300, seed: int = 0) -> pd.DataFrame:
    """MultiIndex(date, ticker) OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    frames = []
    for ticker in tickers:
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n_bars))
        df = pd.DataFrame({
            "Open": close * 0.999,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Close": close,
            "Volume": 1_000_000.0,
        }, index=idx)
        df.index.name = "date"
        df["ticker"] = ticker
        frames.append(df.reset_index().set_index(["date", "ticker"]))
    return pd.concat(frames).sort_index()


def _bar(ticker: str, close: float, open_: float = 0.0, prev_close: float = 0.0) -> pd.DataFrame:
    o = open_ if open_ else close * 0.999
    return pd.DataFrame(
        {"Open": [o], "High": [close * 1.001], "Low": [close * 0.999],
         "Close": [close], "Volume": [1_000_000.0]},
        index=pd.Index([ticker], name="ticker"),
    )


# ---------------------------------------------------------------------------
# PEAD strategy
# ---------------------------------------------------------------------------

class TestPEADStrategy:
    def _make_earnings_df(self, ticker, beat=True) -> pd.DataFrame:
        return pd.DataFrame([{
            "date": pd.Timestamp("2018-06-01"),
            "epsActual": 2.0 if beat else 0.5,
            "epsEstimated": 1.0,
            "revenueActual": 5e9 if beat else 1e9,
            "revenueEstimated": 4e9,
        }])

    def test_no_signal_without_api_key(self):
        from autoalpha.strategies.pead import PEADStrategy
        strat = PEADStrategy(fmp_api_key="")
        data = _mi_ohlcv(["AAPL"])
        strat.fit(data)
        bar = _bar("AAPL", close=150.0)
        result = strat.predict(bar, bar_date=pd.Timestamp("2018-06-01"))
        assert result == {}

    def test_no_signal_on_non_earnings_bar(self):
        from autoalpha.strategies.pead import PEADStrategy
        strat = PEADStrategy(fmp_api_key="TEST")
        data = _mi_ohlcv(["AAPL"])
        with patch("autoalpha.strategies.pead.get_earnings", return_value=self._make_earnings_df("AAPL")):
            strat.fit(data)
        bar = _bar("AAPL", close=150.0)
        result = strat.predict(bar, bar_date=pd.Timestamp("2018-05-01"))
        assert result == {}

    def test_entry_on_beat_with_ah_confirmation(self):
        from autoalpha.strategies.pead import PEADStrategy
        strat = PEADStrategy(fmp_api_key="TEST", ah_threshold=0.01)
        data = _mi_ohlcv(["AAPL"])
        with patch("autoalpha.strategies.pead.get_earnings", return_value=self._make_earnings_df("AAPL")):
            strat.fit(data)

        earnings_date = pd.Timestamp("2018-06-01")

        # Bar before earnings: set prev_close low enough
        bar_prev = _bar("AAPL", close=100.0)
        strat.predict(bar_prev, bar_date=pd.Timestamp("2018-05-31"))

        # Bar on earnings date: close is 3% above prev_close → AH confirmed
        bar_earn = _bar("AAPL", close=103.0)
        result = strat.predict(bar_earn, bar_date=earnings_date)
        assert "AAPL" in result
        assert abs(result["AAPL"] - 0.02) < 1e-9

    def test_no_entry_when_ah_not_confirmed(self):
        from autoalpha.strategies.pead import PEADStrategy
        strat = PEADStrategy(fmp_api_key="TEST", ah_threshold=0.01)
        data = _mi_ohlcv(["AAPL"])
        with patch("autoalpha.strategies.pead.get_earnings", return_value=self._make_earnings_df("AAPL")):
            strat.fit(data)

        bar_prev = _bar("AAPL", close=103.0)
        strat.predict(bar_prev, bar_date=pd.Timestamp("2018-05-31"))

        # Close barely above prev_close (0.5% < 1% threshold)
        bar_earn = _bar("AAPL", close=103.5)
        result = strat.predict(bar_earn, bar_date=pd.Timestamp("2018-06-01"))
        assert "AAPL" not in result

    def test_no_entry_when_eps_miss(self):
        from autoalpha.strategies.pead import PEADStrategy
        strat = PEADStrategy(fmp_api_key="TEST", ah_threshold=0.0)
        data = _mi_ohlcv(["AAPL"])
        with patch("autoalpha.strategies.pead.get_earnings",
                   return_value=self._make_earnings_df("AAPL", beat=False)):
            strat.fit(data)

        bar_prev = _bar("AAPL", close=100.0)
        strat.predict(bar_prev, bar_date=pd.Timestamp("2018-05-31"))
        bar_earn = _bar("AAPL", close=105.0)
        result = strat.predict(bar_earn, bar_date=pd.Timestamp("2018-06-01"))
        assert "AAPL" not in result

    def test_exit_after_hold_bars(self):
        from autoalpha.strategies.pead import PEADStrategy
        strat = PEADStrategy(fmp_api_key="TEST", ah_threshold=0.0, hold_bars=3)
        data = _mi_ohlcv(["AAPL"])
        with patch("autoalpha.strategies.pead.get_earnings",
                   return_value=self._make_earnings_df("AAPL")):
            strat.fit(data)

        dates = pd.date_range("2018-05-31", periods=6, freq="B")

        # Seed prev_close
        strat.predict(_bar("AAPL", close=100.0), bar_date=dates[0])

        # Entry: earnings beat on dates[1] with sufficient gap
        res_entry = strat.predict(_bar("AAPL", close=103.0), bar_date=dates[1])
        assert "AAPL" in res_entry  # entry signal

        # Hold bars 2, 3
        for d in dates[2:4]:
            res = strat.predict(_bar("AAPL", close=104.0), bar_date=d)
            assert res.get("AAPL") != 0.0

        # Exit on bar 4 (hold_bars=3: bars_held reaches 3)
        res_exit = strat.predict(_bar("AAPL", close=104.0), bar_date=dates[4])
        assert res_exit.get("AAPL") == 0.0

    def test_no_double_entry_while_holding(self):
        from autoalpha.strategies.pead import PEADStrategy
        earnings_df = pd.DataFrame([
            {"date": pd.Timestamp("2018-06-01"), "epsActual": 2.0, "epsEstimated": 1.0,
             "revenueActual": 5e9, "revenueEstimated": 4e9},
            {"date": pd.Timestamp("2018-06-04"), "epsActual": 2.0, "epsEstimated": 1.0,
             "revenueActual": 5e9, "revenueEstimated": 4e9},
        ])
        strat = PEADStrategy(fmp_api_key="TEST", ah_threshold=0.0, hold_bars=10)
        data = _mi_ohlcv(["AAPL"])
        with patch("autoalpha.strategies.pead.get_earnings", return_value=earnings_df):
            strat.fit(data)

        dates = pd.date_range("2018-05-31", periods=5, freq="B")
        strat.predict(_bar("AAPL", close=100.0), bar_date=dates[0])
        strat.predict(_bar("AAPL", close=103.0), bar_date=dates[1])   # entry
        assert "AAPL" in strat._holdings

        # Second earnings date while holding — should NOT enter again
        res = strat.predict(_bar("AAPL", close=104.0), bar_date=dates[2])
        assert res.get("AAPL") != 0.02


# ---------------------------------------------------------------------------
# Quality strategy
# ---------------------------------------------------------------------------

class TestQualityStrategy:
    def _make_fund_df(self, ticker: str, roe: float = 0.2, net_margin: float = 0.15) -> pd.DataFrame:
        return pd.DataFrame([{
            "date": pd.Timestamp("2018-03-31"),
            "roe": roe,
            "net_margin": net_margin,
            "netDebt": 1e9,
            "netIncome": 1e8,
            "revenue": 5e8,
        }])

    def test_no_signal_without_api_key(self):
        from autoalpha.strategies.quality import QualityStrategy
        strat = QualityStrategy(fmp_api_key="")
        strat.fit(_mi_ohlcv(["AAPL"]))
        bar = pd.DataFrame({"Close": [150.0], "Volume": [1e6]}, index=pd.Index(["AAPL"]))
        assert strat.predict(bar, bar_date=pd.Timestamp("2018-04-02")) == {}

    def test_rebalance_only_on_quarter_change(self):
        from autoalpha.strategies.quality import QualityStrategy
        strat = QualityStrategy(fmp_api_key="TEST", quantile=0.0)

        fund_df = pd.concat([
            self._make_fund_df("AAPL", roe=0.3),
            self._make_fund_df("MSFT", roe=0.2),
            self._make_fund_df("GOOGL", roe=0.1),
            self._make_fund_df("AMZN", roe=0.05),
            self._make_fund_df("META", roe=0.04),
        ])
        fund_df["ticker"] = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        with patch("autoalpha.strategies.quality.get_fundamentals",
                   side_effect=lambda t, *a, **kw: fund_df[fund_df["ticker"] == t].drop(columns="ticker")):
            strat.fit(_mi_ohlcv(["AAPL", "MSFT", "GOOGL", "AMZN", "META"]))

        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        bar = pd.DataFrame({"Close": [1.0] * 5, "Volume": [1e6] * 5},
                           index=pd.Index(tickers))

        # Before reporting lag (< 45d after Q1 end 2018-03-31, so before 2018-05-15) → no targets
        t1 = strat.predict(bar, bar_date=pd.Timestamp("2018-04-02"))
        t2 = strat.predict(bar, bar_date=pd.Timestamp("2018-04-15"))
        assert t1 == {}
        assert t2 == {}

        # First bar on or after reporting lag (~May 15) → Q1 targets applied immediately
        t3 = strat.predict(bar, bar_date=pd.Timestamp("2018-05-20"))
        assert isinstance(t3, dict)

        # Subsequent bars in the same period use the same targets (no new q_date crossed)
        t4 = strat.predict(bar, bar_date=pd.Timestamp("2018-06-01"))
        assert t4 == t3

    def test_top_quintile_selected(self):
        from autoalpha.strategies.quality import QualityStrategy
        strat = QualityStrategy(fmp_api_key="TEST", quantile=0.80)

        # 6 tickers; AAPL has best quality (high ROE, low leverage, high margin)
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
        roes = [0.5, 0.3, 0.2, 0.1, 0.05, 0.04]
        fund_dfs = {}
        for t, r in zip(tickers, roes):
            fund_dfs[t] = pd.DataFrame([{
                "date": pd.Timestamp("2018-03-31"),
                "roe": r, "net_margin": r * 0.5, "netDebt": 1e8 / r,
                "netIncome": 1e8, "revenue": 5e8,
            }])

        with patch("autoalpha.strategies.quality.get_fundamentals",
                   side_effect=lambda t, *a, **kw: fund_dfs.get(t, pd.DataFrame())):
            strat.fit(_mi_ohlcv(tickers))

        bar = pd.DataFrame({"Close": [1.0] * 6, "Volume": [1e6] * 6},
                           index=pd.Index(tickers))
        # After 45-day reporting lag from Q1 end (2018-03-31 + 45d = 2018-05-15)
        targets = strat.predict(bar, bar_date=pd.Timestamp("2018-05-20"))

        if targets:
            assert "AAPL" in targets  # highest quality must be selected
            assert "NVDA" not in targets  # lowest quality must be excluded
            assert all(v > 0 for v in targets.values())
            assert abs(sum(targets.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Earnings Revisions strategy
# ---------------------------------------------------------------------------

class TestEarningsRevisionsStrategy:
    def _make_yoy_estimates(self, q3_eps: tuple[float, float]) -> pd.DataFrame:
        """Two years of quarterly estimates with controllable Q3 values for YoY comparison."""
        # 8 quarters: Q1-Q4 2017, Q1-Q4 2018
        dates = pd.date_range("2017-01-01", periods=8, freq="QE")
        eps = [1.0, 1.0, q3_eps[0], 1.0, 1.0, 1.0, q3_eps[1], 1.0]
        return pd.DataFrame({"date": dates, "estimatedEpsAvg": eps})

    def test_no_signal_without_api_key(self):
        from autoalpha.strategies.earnings_revisions import EarningsRevisionsStrategy
        strat = EarningsRevisionsStrategy(fmp_api_key="")
        strat.fit(_mi_ohlcv(["AAPL"]))
        bar = pd.DataFrame({"Close": [150.0]}, index=pd.Index(["AAPL"]))
        assert strat.predict(bar, bar_date=pd.Timestamp("2018-06-01")) == {}

    def test_revision_detection(self):
        from autoalpha.strategies.earnings_revisions import EarningsRevisionsStrategy
        strat = EarningsRevisionsStrategy(fmp_api_key="TEST", revision_threshold=0.05)
        # Q3 2017: 1.0, Q3 2018: 1.10 → 10% YoY growth, exceeds threshold
        estimates = self._make_yoy_estimates((1.0, 1.10))

        with patch("autoalpha.strategies.earnings_revisions.get_estimates",
                   return_value=estimates):
            strat.fit(_mi_ohlcv(["AAPL"]))

        # Q3 2018 date should trigger a signal
        rev_date = estimates["date"].iloc[6]  # 7th row = Q3 2018
        bar = pd.DataFrame({"Close": [150.0]}, index=pd.Index(["AAPL"]))
        result = strat.predict(bar, bar_date=rev_date)
        assert "AAPL" in result
        assert abs(result["AAPL"] - 0.02) < 1e-9

    def test_no_revision_when_below_threshold(self):
        from autoalpha.strategies.earnings_revisions import EarningsRevisionsStrategy
        strat = EarningsRevisionsStrategy(fmp_api_key="TEST", revision_threshold=0.05)
        # Q3 2017: 1.0, Q3 2018: 1.02 → only 2% YoY, below threshold
        estimates = self._make_yoy_estimates((1.0, 1.02))

        with patch("autoalpha.strategies.earnings_revisions.get_estimates",
                   return_value=estimates):
            strat.fit(_mi_ohlcv(["AAPL"]))

        rev_date = estimates["date"].iloc[6]
        bar = pd.DataFrame({"Close": [150.0]}, index=pd.Index(["AAPL"]))
        result = strat.predict(bar, bar_date=rev_date)
        assert "AAPL" not in result

    def test_exit_after_hold_bars(self):
        from autoalpha.strategies.earnings_revisions import EarningsRevisionsStrategy
        strat = EarningsRevisionsStrategy(fmp_api_key="TEST", revision_threshold=0.05,
                                          hold_bars=2)
        estimates = self._make_yoy_estimates((1.0, 1.10))

        with patch("autoalpha.strategies.earnings_revisions.get_estimates",
                   return_value=estimates):
            strat.fit(_mi_ohlcv(["AAPL"]))

        rev_date = estimates["date"].iloc[6]  # Q3 2018
        bar = pd.DataFrame({"Close": [150.0]}, index=pd.Index(["AAPL"]))

        strat.predict(bar, bar_date=rev_date)  # entry
        strat.predict(bar, bar_date=rev_date + pd.offsets.BDay(1))  # hold (bars_held = 1)
        result = strat.predict(bar, bar_date=rev_date + pd.offsets.BDay(2))  # exit (bars_held = 2 >= 2)
        assert result.get("AAPL") == 0.0


# ---------------------------------------------------------------------------
# Earnings NLP strategy
# ---------------------------------------------------------------------------

class TestEarningsNLPStrategy:
    def test_score_positive_transcript(self):
        from autoalpha.strategies.earnings_nlp import _score_transcript
        text = "We beat expectations with record growth and strong momentum exceeded all targets."
        score = _score_transcript(text)
        assert score > 0

    def test_score_negative_transcript(self):
        from autoalpha.strategies.earnings_nlp import _score_transcript
        text = "Revenue declined below estimates, weak demand with challenging uncertain conditions."
        score = _score_transcript(text)
        assert score < 0

    def test_score_empty_transcript(self):
        from autoalpha.strategies.earnings_nlp import _score_transcript
        assert _score_transcript("") == 0.0

    def test_no_signal_without_api_key(self):
        from autoalpha.strategies.earnings_nlp import EarningsNLPStrategy
        strat = EarningsNLPStrategy(fmp_api_key="")
        strat.fit(_mi_ohlcv(["AAPL"]))
        bar = pd.DataFrame({"Close": [150.0]}, index=pd.Index(["AAPL"]))
        assert strat.predict(bar, bar_date=pd.Timestamp("2019-01-02")) == {}

    def test_entry_on_positive_transcript(self):
        from autoalpha.strategies.earnings_nlp import EarningsNLPStrategy
        strat = EarningsNLPStrategy(fmp_api_key="TEST", score_threshold=0.001)

        positive_text = "record beat exceeded strong growth robust solid achieved outperformed " * 20
        with patch("autoalpha.strategies.earnings_nlp.get_transcripts",
                   return_value=positive_text):
            strat.fit(_mi_ohlcv(["AAPL"], n_bars=400))

        bar = pd.DataFrame({"Close": [150.0]}, index=pd.Index(["AAPL"]))
        # First trading day of Q1 2019 → evaluates Q4 2018 transcript
        result = strat.predict(bar, bar_date=pd.Timestamp("2019-01-02"))
        assert "AAPL" in result
        assert abs(result["AAPL"] - 0.02) < 1e-9

    def test_no_entry_on_negative_transcript(self):
        from autoalpha.strategies.earnings_nlp import EarningsNLPStrategy
        strat = EarningsNLPStrategy(fmp_api_key="TEST", score_threshold=0.001)

        negative_text = "declined missed below weak disappointing challenged difficult unfavorable " * 20
        with patch("autoalpha.strategies.earnings_nlp.get_transcripts",
                   return_value=negative_text):
            strat.fit(_mi_ohlcv(["AAPL"], n_bars=400))

        bar = pd.DataFrame({"Close": [150.0]}, index=pd.Index(["AAPL"]))
        result = strat.predict(bar, bar_date=pd.Timestamp("2019-01-02"))
        assert "AAPL" not in result

    def test_no_double_entry_same_quarter(self):
        from autoalpha.strategies.earnings_nlp import EarningsNLPStrategy
        strat = EarningsNLPStrategy(fmp_api_key="TEST", score_threshold=0.001, hold_bars=100)
        positive_text = "record beat exceeded strong growth " * 30

        with patch("autoalpha.strategies.earnings_nlp.get_transcripts",
                   return_value=positive_text):
            strat.fit(_mi_ohlcv(["AAPL"], n_bars=400))

        bar = pd.DataFrame({"Close": [150.0]}, index=pd.Index(["AAPL"]))
        r1 = strat.predict(bar, bar_date=pd.Timestamp("2019-01-02"))
        r2 = strat.predict(bar, bar_date=pd.Timestamp("2019-01-03"))  # same quarter
        assert r2.get("AAPL") != 0.02  # no re-entry within same quarter

    def test_exit_after_hold_bars(self):
        from autoalpha.strategies.earnings_nlp import EarningsNLPStrategy
        strat = EarningsNLPStrategy(fmp_api_key="TEST", score_threshold=0.001, hold_bars=2)
        positive_text = "record beat exceeded strong growth " * 30

        with patch("autoalpha.strategies.earnings_nlp.get_transcripts",
                   return_value=positive_text):
            strat.fit(_mi_ohlcv(["AAPL"], n_bars=400))

        bar = pd.DataFrame({"Close": [150.0]}, index=pd.Index(["AAPL"]))
        dates = pd.date_range("2019-01-02", periods=5, freq="B")

        strat.predict(bar, bar_date=dates[0])   # entry (quarter changed)
        strat.predict(bar, bar_date=dates[1])   # hold (bars_held=1)
        res = strat.predict(bar, bar_date=dates[2])  # exit (bars_held=2 >= 2)
        assert res.get("AAPL") == 0.0
