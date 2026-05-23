"""Earnings NLP strategy.

Scores earnings call transcripts using the Loughran-McDonald financial
sentiment lexicon (simplified subset). Enters a long position when tone
is positive and uncertainty is low.

Signal = pos_count/words - neg_count/words - uncertainty_weight * unc_count/words

Entry: next bar's open after the transcript release quarter boundary.
Hold:  HOLD_BARS trading days.

Returns {} on non-transcript bars.
"""
from __future__ import annotations

import logging
import math
import os
from datetime import date
from typing import Optional

import pandas as pd

from autoalpha.core.strategy import Strategy
from autoalpha.data.fetcher import get_transcripts, VaultLeakError

logger = logging.getLogger(__name__)

POSITION_SIZE = 0.02
SCORE_THRESHOLD = 0.005     # minimum net positive score to enter
HOLD_BARS = 63              # ~1 quarter hold
UNCERTAINTY_WEIGHT = 0.5    # uncertainty term discount

# Simplified Loughran-McDonald lexicon subsets (lowercased)
_LM_POSITIVE = frozenset({
    "beat", "exceeded", "record", "strong", "growth", "increased", "improved",
    "expanding", "raised", "robust", "solid", "confident", "optimistic",
    "outperformed", "delivered", "achieved", "momentum", "accelerating",
    "attractive", "exceeded", "exceptional", "favorable", "strength",
})
_LM_NEGATIVE = frozenset({
    "declined", "decreased", "loss", "below", "missed", "challenged", "difficult",
    "concern", "risk", "weak", "disappointing", "unfavorable",
    "reduced", "cut", "falling", "headwind", "impairment", "restructuring",
    "downturn", "adverse", "deteriorated", "underperformed",
})
_LM_UNCERTAIN = frozenset({
    "may", "might", "could", "approximately", "potentially", "uncertain", "estimates",
    "expects", "anticipates", "projected", "depends", "pending", "subject",
    "likely", "possible", "if", "perhaps", "sometime", "whether",
})


def _score_transcript(text: str) -> float:
    """Return net sentiment score in [-1, 1]. Positive = bullish."""
    if not text:
        return 0.0
    words = text.lower().split()
    n = len(words)
    if n == 0:
        return 0.0

    pos = sum(1 for w in words if w.strip(".,;:!?\"'()") in _LM_POSITIVE)
    neg = sum(1 for w in words if w.strip(".,;:!?\"'()") in _LM_NEGATIVE)
    unc = sum(1 for w in words if w.strip(".,;:!?\"'()") in _LM_UNCERTAIN)

    score = (pos - neg) / n - UNCERTAINTY_WEIGHT * unc / n
    return score


def _quarter_of(dt: pd.Timestamp) -> tuple[int, int]:
    return dt.year, (dt.month - 1) // 3 + 1


class EarningsNLPStrategy(Strategy):
    """Earnings call transcript NLP strategy (Loughran-McDonald lexicon)."""

    def __init__(
        self,
        position_size: float = POSITION_SIZE,
        score_threshold: float = SCORE_THRESHOLD,
        hold_bars: int = HOLD_BARS,
        fmp_api_key: Optional[str] = None,
    ):
        self._position_size = position_size
        self._score_threshold = score_threshold
        self._hold_bars = hold_bars
        self._api_key = fmp_api_key or os.environ.get("FMP_API_KEY", "")

        # {ticker: {(year, quarter): score}}
        self._scores: dict[str, dict[tuple, float]] = {}
        # {ticker: int} — bars held
        self._holdings: dict[str, int] = {}
        self._prev_quarter: Optional[tuple] = None

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> None:
        """Fetch and score transcripts for all tickers × quarters in-sample."""
        self._scores = {}
        self._holdings = {}
        self._prev_quarter = None

        if data.empty:
            return

        tickers = data.index.get_level_values("ticker").unique().tolist()
        dates = data.index.get_level_values("date")
        start_year = dates.min().year
        end_date = dates.max()

        # Transcripts are released ~45 days after quarter end. Compute the last
        # (year, quarter) whose transcript is available as of end_date.
        # Example: fold ends 2024-02-15 → 45-day cutoff = 2024-01-01 → Q4 2023 available.
        # Without this lag, a fold ending in January would set last_complete_quarter=0
        # and skip all of 2024, missing Q4 2023 transcripts for the end_year.
        cutoff = end_date - pd.Timedelta(days=45)
        _quarter_ends = [(3, 31), (6, 30), (9, 30), (12, 31)]
        avail_year, avail_q = cutoff.year, 0
        for qnum, (month, day) in enumerate(_quarter_ends, 1):
            if pd.Timestamp(cutoff.year, month, day) <= cutoff:
                avail_q = qnum
        if avail_q == 0:
            avail_year -= 1
            avail_q = 4

        if not self._api_key:
            logger.warning("FMP_API_KEY not set — EarningsNLP will produce no signals")
            return

        for ticker in tickers:
            ticker_scores: dict[tuple, float] = {}
            for year in range(start_year, avail_year + 1):
                max_q = avail_q if year == avail_year else 4
                for quarter in range(1, max_q + 1):
                    try:
                        text = get_transcripts(
                            ticker, year, quarter, api_key=self._api_key
                        )
                        if text:
                            ticker_scores[(year, quarter)] = _score_transcript(text)
                    except VaultLeakError:
                        raise
                    except Exception:
                        pass  # missing transcripts are common; skip silently
            if ticker_scores:
                self._scores[ticker] = ticker_scores

    def predict(
        self,
        bar_data: pd.DataFrame,
        bar_date: Optional[pd.Timestamp] = None,
    ) -> dict[str, float]:
        """Signal on first trading day of each new quarter using prior quarter's transcript."""
        result: dict[str, float] = {}

        if bar_date is None or bar_data.empty:
            return result

        # Manage holdings
        to_close: list[str] = []
        for ticker, bars_held in self._holdings.items():
            self._holdings[ticker] = bars_held + 1
            if self._holdings[ticker] >= self._hold_bars:
                result[ticker] = 0.0
                to_close.append(ticker)
        for t in to_close:
            del self._holdings[t]

        # Only evaluate on first trading day of a new quarter
        current_quarter = _quarter_of(bar_date)
        if current_quarter == self._prev_quarter:
            return result

        self._prev_quarter = current_quarter

        # Evaluate prior quarter's transcript score for each ticker
        prior_year, prior_q = current_quarter
        prior_q -= 1
        if prior_q == 0:
            prior_q = 4
            prior_year -= 1
        prior_key = (prior_year, prior_q)

        for ticker in bar_data.index:
            if ticker in self._holdings:
                continue
            score = self._scores.get(ticker, {}).get(prior_key, 0.0)
            if score >= self._score_threshold:
                result[ticker] = self._position_size
                self._holdings[ticker] = 0
                logger.debug(
                    "NLP entry %s on %s (score=%.4f)", ticker, bar_date.date(), score
                )

        return result
