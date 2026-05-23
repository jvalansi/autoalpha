"""Outer LLM hypothesis loop — Phase 4 entry point.

Runs iteratively until `max_iterations` or `max_cost_usd` is reached.
Each iteration either generates a fresh hypothesis or refines the most-recent
rejected one (up to `max_refinements` times).

Accepted hypotheses are:
  - Written to HypothesisMemory with status='active'
  - Registered in SignalLibrary
  - Their CPCV alpha returns stored in portfolio_alpha
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from anthropic import Anthropic

from autoalpha.evaluation.library import SignalLibrary
from autoalpha.research.code_validator import CodeValidationError, validate_predict_body, wrap_predict_body
from autoalpha.research.hypothesis import Hypothesis, HypothesisValidationError
from autoalpha.research.memory import HypothesisMemory
from autoalpha.research.prompts import (
    build_generation_prompt,
    build_interpretation_prompt,
    build_refinement_prompt,
    parse_llm_json,
)
from autoalpha.research.subprocess_runner import BacktestResult, run_strategy_subprocess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Acceptance thresholds
# ---------------------------------------------------------------------------

_DSR_THRESHOLD = 0.95
_MAX_DRAWDOWN_THRESHOLD = 0.25  # 25%

# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


class ResearchLoop:
    """Drives the LLM hypothesis generation / evaluate / accept cycle."""

    def __init__(
        self,
        data_path: str,
        db_path: Optional[Path] = None,
        model: str = "claude-opus-4-7",
        max_iterations: int = 20,
        max_cost_usd: float = 5.00,
        max_refinements: int = 3,
    ) -> None:
        self._data_path = data_path
        self._model = model
        self._max_iterations = max_iterations
        self._max_cost_usd = max_cost_usd
        self._max_refinements = max_refinements
        self._client = Anthropic()

        db_kwargs = {"db_path": db_path} if db_path else {}
        self._memory = HypothesisMemory(**db_kwargs)
        self._library = SignalLibrary(**({"db_path": db_path} if db_path else {}))

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the loop until budget or iteration cap is reached."""
        total_cost = 0.0

        for iteration in range(1, self._max_iterations + 1):
            if total_cost >= self._max_cost_usd:
                logger.info("Budget exhausted (%.2f USD). Stopping.", total_cost)
                break

            logger.info("=== Iteration %d (spent $%.3f) ===", iteration, total_cost)

            trial_number = self._memory.next_trial_number()

            # Decide: refine or generate
            pending = self._memory.get_pending_refinement(self._max_refinements)

            if pending:
                hyp, hyp_id, cost = self._refine(pending, trial_number)
            else:
                hyp, hyp_id, cost = self._generate(trial_number)

            total_cost += cost

            if hyp is None:
                logger.warning("Hypothesis construction failed on iteration %d", iteration)
                continue

            # Validate code
            try:
                validate_predict_body(hyp.predict_body)
            except CodeValidationError as exc:
                logger.warning("Code validation failed: %s", exc)
                self._memory.update_status(hyp_id, "error")
                continue

            # Run backtest
            source = wrap_predict_body(hyp.predict_body)
            n_trials = self._memory.current_trial_count()
            result = run_strategy_subprocess(source, self._data_path, n_trials)

            if not result.succeeded:
                logger.warning("Backtest failed: %s", result.error)
                self._memory.update_result(
                    hyp_id,
                    status="error",
                    sharpe=0.0,
                    dsr=0.0,
                    max_drawdown=0.0,
                    additional_cost_usd=0.0,
                    observation="Backtest raised an exception.",
                    justification=str(result.error)[:500],
                )
                continue

            # Interpret result
            interp, interp_cost = self._interpret(hyp, result)
            total_cost += interp_cost
            status = interp.get("status", "rejected")
            observation = interp.get("observation", "")
            justification = interp.get("justification", "")

            self._memory.update_result(
                hyp_id,
                status=status,
                sharpe=result.sharpe,
                dsr=result.dsr,
                max_drawdown=result.max_drawdown,
                additional_cost_usd=interp_cost,
                observation=observation,
                justification=justification,
            )

            if status == "accepted":
                self._accept(hyp, hyp_id, result)
            else:
                self._memory.increment_refinement_count(hyp_id)
                logger.info(
                    "Rejected — Sharpe=%.2f DSR=%.3f DD=%.1f%% — %s",
                    result.sharpe,
                    result.dsr,
                    result.max_drawdown * 100,
                    justification[:120],
                )

        logger.info("Loop complete after %d iterations. Total cost: $%.3f", iteration, total_cost)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate(self, trial_number: int) -> tuple[Optional[Hypothesis], int, float]:
        history = self._memory.get_all_for_trace()
        cohort_weights = self._memory.get_cohort_weight_summary(self._library.all_weights())
        system, user = build_generation_prompt(history, cohort_weights, trial_number)

        raw, cost = self._call_llm(system, user)
        return self._parse_and_store(raw, trial_number, cost)

    def _refine(self, pending: dict, trial_number: int) -> tuple[Optional[Hypothesis], int, float]:
        original_hyp = Hypothesis.from_json(pending["hypothesis_json"])
        backtest_result = {
            "sharpe": pending.get("sharpe") or 0.0,
            "dsr": pending.get("dsr") or 0.0,
            "max_drawdown": pending.get("max_drawdown") or 0.0,
        }
        system, user = build_refinement_prompt(original_hyp, backtest_result, trial_number)
        raw, cost = self._call_llm(system, user)
        result = self._parse_and_store(raw, trial_number, cost)
        # Always increment even on parse failure — a failed refinement attempt still counts
        # toward the max_refinements cap so the loop can't retry the same hypothesis forever.
        self._memory.increment_refinement_count(pending["id"])
        return result

    def _interpret(self, hyp: Hypothesis, result: BacktestResult) -> tuple[dict, float]:
        system, user = build_interpretation_prompt(hyp, result.to_dict())
        raw, cost = self._call_llm(system, user)
        try:
            data = parse_llm_json(raw)
        except (ValueError, json.JSONDecodeError):
            # Fall back to rule-based decision
            accepted = result.dsr > _DSR_THRESHOLD and result.max_drawdown < _MAX_DRAWDOWN_THRESHOLD
            data = {
                "status": "accepted" if accepted else "rejected",
                "observation": raw[:200],
                "justification": "Parsed from metrics (LLM JSON parse failed).",
            }
        return data, cost

    def _parse_and_store(
        self,
        raw: str,
        trial_number: int,
        cost: float,
    ) -> tuple[Optional[Hypothesis], int, float]:
        try:
            data = parse_llm_json(raw)
            hyp = Hypothesis.from_dict(data)
        except (ValueError, KeyError, HypothesisValidationError) as exc:
            logger.warning("Failed to construct Hypothesis from LLM output: %s", exc)
            return None, -1, cost

        hyp_id = self._memory.store_hypothesis(hyp, trial_number, cost_usd=cost)
        return hyp, hyp_id, cost

    def _accept(self, hyp: Hypothesis, hyp_id: int, result: BacktestResult) -> None:
        if result.returns:
            alpha = pd.Series(result.returns)
            self._memory.store_portfolio_alpha(hyp_id, alpha)

        # Mark active before add_signal so get_active_count() is accurate
        self._memory.update_status(hyp_id, "active")
        self._library.add_signal(hyp.concise_reason)
        n_active = self._memory.get_active_count()
        logger.info(
            "ACCEPTED — %s  Sharpe=%.2f DSR=%.3f DD=%.1f%%  active_signals=%d",
            hyp.concise_reason,
            result.sharpe,
            result.dsr,
            result.max_drawdown * 100,
            n_active,
        )

    def _call_llm(self, system: str, user: str) -> tuple[str, float]:
        """Call Claude with prompt caching and return (response_text, cost_usd)."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        text = response.content[0].text if response.content else ""
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cost = _estimate_cost(
            usage.input_tokens, usage.output_tokens, self._model,
            cache_read_tokens=cache_read, cache_write_tokens=cache_write,
        )
        logger.debug(
            "LLM call: %d in + %d out + %d cache_read + %d cache_write = $%.4f",
            usage.input_tokens, usage.output_tokens, cache_read, cache_write, cost,
        )
        return text, cost

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ResearchLoop":
        return self

    def __exit__(self, *_) -> None:
        self._memory.close()
        self._library.close()


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

_PRICING: dict[str, tuple[float, float, float, float]] = {
    # (input, output, cache_write, cache_read) $/1M tokens
    "claude-opus-4-7":   (15.00, 75.00, 18.75, 1.50),
    "claude-sonnet-4-6": (3.00,  15.00,  3.75, 0.30),
    "claude-haiku-4-5":  (0.80,   4.00,  1.00, 0.08),
}


def _estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    in_price, out_price, write_price, read_price = _PRICING.get(model, (15.00, 75.00, 18.75, 1.50))
    return (
        input_tokens * in_price
        + output_tokens * out_price
        + cache_write_tokens * write_price
        + cache_read_tokens * read_price
    ) / 1_000_000
