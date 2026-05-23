"""Tests for Hypothesis dataclass and causal mechanism validation."""
from __future__ import annotations

import json

import pytest

from autoalpha.research.hypothesis import (
    Hypothesis,
    HypothesisValidationError,
    _validate_knowledge,
)

_VALID_KNOWLEDGE = (
    "When companies buy back shares they reduce the float, which mechanically "
    "increases EPS and signals management confidence in future cash flows, "
    "creating sustained buying pressure that supports the stock price."
)

_VALID_KWARGS = dict(
    hypothesis="Companies with high buyback yield outperform over the next quarter.",
    reason="Buybacks reduce float and signal management confidence.",
    concise_reason="buyback yield",
    knowledge=_VALID_KNOWLEDGE,
    predict_body="        return {}",
    cohort="value",
)


class TestKnowledgeValidation:
    def test_valid_knowledge_passes(self):
        _validate_knowledge(_VALID_KNOWLEDGE)  # should not raise

    def test_too_short_raises(self):
        with pytest.raises(HypothesisValidationError, match="too short"):
            _validate_knowledge("Stocks go up when people buy them.")

    def test_empty_raises(self):
        with pytest.raises(HypothesisValidationError):
            _validate_knowledge("")

    @pytest.mark.parametrize("phrase", [
        "historically correlated",
        "backtested well",
        "data showed",
        "in the past",
        "has worked",
        "tends to",
        "machine learning found",
        "empirically observed",
        "showed a pattern",
        "pattern in the data",
        "historical pattern",
    ])
    def test_curve_fitting_phrases_rejected(self, phrase):
        # Sentence is deliberately ≥20 words so the length check passes and
        # the phrase check is the one that fires.
        bad = (
            f"This factor {phrase} and therefore we believe it should continue "
            f"to predict future equity returns based on this economic reasoning "
            f"which is well established in the literature."
        )
        with pytest.raises(HypothesisValidationError, match="curve-fitting"):
            _validate_knowledge(bad)

    def test_case_insensitive_detection(self):
        bad = _VALID_KNOWLEDGE + " It Has Worked historically."
        with pytest.raises(HypothesisValidationError):
            _validate_knowledge(bad)

    def test_standalone_pattern_word_allowed(self):
        # 'pattern' alone was previously blacklisted, incorrectly blocking
        # legitimate causal references like 'behavioral pattern' or 'demand pattern'.
        valid = (
            "Investors exhibit a behavioral pattern of herding — buying into recent "
            "winners and selling recent losers — which creates persistent momentum "
            "as late adopters pile in after initial price moves have already begun."
        )
        _validate_knowledge(valid)  # should not raise


class TestHypothesisDataclass:
    def test_valid_construction(self):
        hyp = Hypothesis(**_VALID_KWARGS)
        assert hyp.cohort == "value"
        assert hyp.observation == ""

    def test_invalid_cohort_raises(self):
        with pytest.raises(HypothesisValidationError, match="cohort"):
            Hypothesis(**{**_VALID_KWARGS, "cohort": "unknown"})

    def test_all_cohorts_accepted(self):
        for cohort in ("momentum", "value", "quality", "macro"):
            Hypothesis(**{**_VALID_KWARGS, "cohort": cohort})

    def test_to_json_round_trip(self):
        hyp = Hypothesis(**_VALID_KWARGS)
        restored = Hypothesis.from_json(hyp.to_json())
        assert restored.hypothesis == hyp.hypothesis
        assert restored.knowledge == hyp.knowledge
        assert restored.cohort == hyp.cohort

    def test_from_dict_strips_valid_cohorts_field(self):
        d = {**_VALID_KWARGS, "_VALID_COHORTS": ["momentum"]}
        hyp = Hypothesis.from_dict(d)
        assert hyp.cohort == "value"

    def test_to_dict_excludes_valid_cohorts(self):
        hyp = Hypothesis(**_VALID_KWARGS)
        d = hyp.to_dict()
        assert "_VALID_COHORTS" not in d

    def test_observation_justification_roundtrip(self):
        hyp = Hypothesis(**_VALID_KWARGS)
        hyp.observation = "Sharpe was positive in all OOS folds."
        hyp.justification = "Accepted: DSR > 0.95."
        restored = Hypothesis.from_json(hyp.to_json())
        assert restored.observation == hyp.observation
        assert restored.justification == hyp.justification
