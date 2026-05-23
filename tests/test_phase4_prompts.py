"""Tests for prompts module — template construction and JSON parsing."""
from __future__ import annotations

import json

import pytest

from autoalpha.research.hypothesis import Hypothesis
from autoalpha.research.prompts import (
    SYSTEM_PROMPT,
    build_generation_prompt,
    build_interpretation_prompt,
    build_refinement_prompt,
    parse_llm_json,
)

_VALID_KNOWLEDGE = (
    "Short sellers face margin calls when prices rise sharply, forcing them to cover "
    "their positions by buying shares, which further accelerates the upward price move "
    "in a feedback loop known as a short squeeze driven by limited supply."
)

_HYP = Hypothesis(
    hypothesis="High short interest predicts positive returns via squeeze dynamics.",
    reason="Short squeezes create forced buying pressure.",
    concise_reason="short squeeze",
    knowledge=_VALID_KNOWLEDGE,
    predict_body="        return {}",
    cohort="momentum",
)


class TestSystemPrompt:
    def test_system_prompt_is_nonempty(self):
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_contains_contract(self):
        assert "predict" in SYSTEM_PROMPT
        assert "bar_data" in SYSTEM_PROMPT

    def test_system_prompt_contains_json_format(self):
        assert "hypothesis" in SYSTEM_PROMPT
        assert "predict_body" in SYSTEM_PROMPT
        assert "cohort" in SYSTEM_PROMPT


class TestBuildGenerationPrompt:
    def test_returns_two_strings(self):
        sys_p, user_p = build_generation_prompt([], {}, trial_number=1)
        assert isinstance(sys_p, str) and isinstance(user_p, str)

    def test_trial_number_in_user_prompt(self):
        _, user_p = build_generation_prompt([], {}, trial_number=7)
        assert "7" in user_p

    def test_history_included(self):
        history = [
            {
                "trial_number": 1,
                "status": "rejected",
                "sharpe": 0.3,
                "dsr": 0.4,
                "cohort": "value",
                "hypothesis_json": json.dumps({
                    "hypothesis": "PE ratio predicts returns.",
                    "reason": "Cheap stocks revert to fair value.",
                    "concise_reason": "pe ratio",
                    "knowledge": _VALID_KNOWLEDGE,
                    "predict_body": "        return {}",
                    "cohort": "value",
                }),
            }
        ]
        _, user_p = build_generation_prompt(history, {}, trial_number=2)
        assert "Trial 1" in user_p

    def test_empty_history_message(self):
        _, user_p = build_generation_prompt([], {}, trial_number=1)
        assert "no history" in user_p.lower()

    def test_cohort_weights_included(self):
        weights = {"momentum": 1.5, "value": 0.4}
        _, user_p = build_generation_prompt([], weights, trial_number=1)
        assert "momentum" in user_p
        assert "0.400" in user_p or "0.4" in user_p


class TestBuildRefinementPrompt:
    def test_returns_two_strings(self):
        sys_p, user_p = build_refinement_prompt(_HYP, {"sharpe": 0.3, "dsr": 0.4, "max_drawdown": 0.1}, trial_number=2)
        assert isinstance(sys_p, str) and isinstance(user_p, str)

    def test_hypothesis_name_in_prompt(self):
        _, user_p = build_refinement_prompt(_HYP, {"sharpe": 0.3, "dsr": 0.4, "max_drawdown": 0.1}, trial_number=2)
        assert "short squeeze" in user_p

    def test_backtest_metrics_in_prompt(self):
        _, user_p = build_refinement_prompt(_HYP, {"sharpe": 0.42, "dsr": 0.38, "max_drawdown": 0.18}, trial_number=2)
        assert "0.42" in user_p

    def test_predict_body_included_in_refinement(self):
        _, user_p = build_refinement_prompt(_HYP, {"sharpe": 0.3, "dsr": 0.4, "max_drawdown": 0.1}, trial_number=2)
        assert _HYP.predict_body.strip() in user_p


class TestBuildInterpretationPrompt:
    def test_returns_two_strings(self):
        sys_p, user_p = build_interpretation_prompt(_HYP, {"sharpe": 1.3, "dsr": 0.97, "max_drawdown": 0.09})
        assert isinstance(sys_p, str) and isinstance(user_p, str)

    def test_acceptance_thresholds_mentioned(self):
        _, user_p = build_interpretation_prompt(_HYP, {"sharpe": 1.3, "dsr": 0.97, "max_drawdown": 0.09})
        assert "0.95" in user_p
        assert "60" in user_p

    def test_system_contains_response_format(self):
        sys_p, _ = build_interpretation_prompt(_HYP, {})
        assert "observation" in sys_p
        assert "justification" in sys_p
        assert "status" in sys_p


class TestParseLlmJson:
    def test_plain_json(self):
        data = parse_llm_json('{"key": "value"}')
        assert data["key"] == "value"

    def test_strips_json_fence(self):
        text = '```json\n{"key": "value"}\n```'
        data = parse_llm_json(text)
        assert data["key"] == "value"

    def test_strips_plain_fence(self):
        text = '```\n{"key": "value"}\n```'
        data = parse_llm_json(text)
        assert data["key"] == "value"

    def test_raises_on_invalid_json(self):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            parse_llm_json("this is not json")

    def test_nested_object(self):
        text = '{"a": {"b": [1, 2, 3]}}'
        data = parse_llm_json(text)
        assert data["a"]["b"] == [1, 2, 3]
