"""Hypothesis dataclass with causal mechanism enforcement.

The 'knowledge' field must contain a genuine causal explanation of why the
factor should predict returns. Pure curve-fitting (e.g. "historically correlated")
is rejected at construction time.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

_MIN_KNOWLEDGE_WORDS: int = 20

_CURVE_FITTING_PHRASES: frozenset[str] = frozenset({
    "backtested well",
    "historically correlated",
    "showed a pattern",
    "pattern in the data",
    "historical pattern",
    "machine learning found",
    "data showed",
    "empirically observed",
    "in the past",
    "has worked",
    "tends to",
})


class HypothesisValidationError(ValueError):
    """Raised when a Hypothesis fails causal mechanism validation."""


def _validate_knowledge(knowledge: str) -> None:
    if not knowledge or len(knowledge.split()) < _MIN_KNOWLEDGE_WORDS:
        raise HypothesisValidationError(
            f"knowledge field too short — must explain the causal mechanism "
            f"in at least {_MIN_KNOWLEDGE_WORDS} words"
        )
    lower = knowledge.lower()
    for phrase in _CURVE_FITTING_PHRASES:
        if phrase in lower:
            raise HypothesisValidationError(
                f"knowledge field appears to describe curve-fitting rather than "
                f"a causal mechanism (contains: '{phrase}')"
            )


@dataclass
class Hypothesis:
    """A single research hypothesis produced by the LLM loop.

    Fields:
        hypothesis:     One-sentence description of the factor.
        reason:         Why this factor should predict returns.
        concise_reason: ≤5-word summary; used as the signal name in SignalLibrary.
        observation:    Filled after backtest — what the data showed.
        justification:  Filled after backtest — accept/reject rationale.
        knowledge:      Causal mechanism; validated non-empty and non-trivial.
        predict_body:   Raw LLM-generated body of the predict() method.
        cohort:         One of 'momentum', 'value', 'quality', 'macro'.
    """

    hypothesis: str
    reason: str
    concise_reason: str
    knowledge: str
    predict_body: str
    cohort: str = "value"
    observation: str = ""
    justification: str = ""

    _VALID_COHORTS: frozenset[str] = field(
        default=frozenset({"momentum", "value", "quality", "macro"}),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _validate_knowledge(self.knowledge)
        if self.cohort not in self._VALID_COHORTS:
            raise HypothesisValidationError(
                f"cohort must be one of {sorted(self._VALID_COHORTS)}, got '{self.cohort}'"
            )

    def to_dict(self) -> dict[str, str]:
        d = asdict(self)
        d.pop("_VALID_COHORTS", None)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        d = {k: v for k, v in d.items() if k != "_VALID_COHORTS"}
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "Hypothesis":
        return cls.from_dict(json.loads(s))
