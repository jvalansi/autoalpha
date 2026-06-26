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
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

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
# Slack notifier
# ---------------------------------------------------------------------------


class _SlackNotifier:
    """Posts loop events to a Slack channel/thread via the bot token.

    Reads from env vars:
      SLACK_BOT_TOKEN      — required for any posting
      SLACK_LOOP_CHANNEL   — channel ID to post to
      SLACK_LOOP_THREAD_TS — thread timestamp (optional; posts in-thread if set)
    """

    def __init__(self) -> None:
        self._token = os.environ.get("SLACK_BOT_TOKEN", "")
        self._channel = os.environ.get("SLACK_LOOP_CHANNEL", "")
        self._thread_ts = os.environ.get("SLACK_LOOP_THREAD_TS", "")
        self._enabled = bool(self._token and self._channel)

    def post(self, text: str) -> None:
        if not self._enabled:
            return
        payload: dict = {"channel": self._channel, "text": text}
        if self._thread_ts:
            payload["thread_ts"] = self._thread_ts
        try:
            requests.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {self._token}"},
                json=payload,
                timeout=10,
            )
        except Exception as exc:
            logger.warning("Slack notification failed: %s", exc)


# ---------------------------------------------------------------------------
# Acceptance thresholds
# ---------------------------------------------------------------------------

_DSR_THRESHOLD = 0.65  # PSR vs market benchmark 0.62: ~65% confident true SR exceeds market (≈ Sharpe > 0.9)
_MAX_DRAWDOWN_THRESHOLD = 0.30  # 30% — diversified ~80-stock long-only universe
_MIN_ACTIVE_DAYS = 50           # need at least 50 invested OOS days for meaningful stats
_MIN_BASELINE_OVERLAP_DAYS = 30 # below this, regression is too noisy; defer to standalone gates


def _min_alpha_t(n_active: int) -> float:
    """Scale the marginal-α t-stat threshold by current book size.

    LOO-mean ≈ overall-mean as N grows, so every candidate looks more redundant by
    construction. Decay the bar linearly from 2.0 (sparse book, need clear winners)
    to a 0.5 floor (sign filter on α).
    """
    return max(0.5, 2.0 - 0.01 * n_active)

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
        self._slack = _SlackNotifier()

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

            # Hard-coded activity gate (LLM can't override this)
            active_days = int(result.activity_rate * len(result.returns)) if result.returns else 0
            if status == "accepted" and active_days < _MIN_ACTIVE_DAYS:
                status = "rejected"
                justification = (
                    f"Only {active_days} active OOS days < {_MIN_ACTIVE_DAYS} minimum — "
                    "too few observations for the Sharpe to be statistically meaningful."
                )
                self._memory.update_result(hyp_id, status="rejected", sharpe=result.sharpe,
                                           dsr=result.dsr, max_drawdown=result.max_drawdown,
                                           additional_cost_usd=0.0, observation=observation,
                                           justification=justification)

            # Marginal-alpha gate: candidate must add positive alpha vs current equal-weight book
            if status == "accepted":
                passes, alpha_note = self._baseline_alpha_passes(result)
                if not passes:
                    status = "rejected"
                    justification = alpha_note
                    self._memory.update_result(hyp_id, status="rejected", sharpe=result.sharpe,
                                               dsr=result.dsr, max_drawdown=result.max_drawdown,
                                               additional_cost_usd=0.0, observation=observation,
                                               justification=justification)

            if status == "accepted":
                self._accept(hyp, hyp_id, result)
            else:
                self._memory.increment_refinement_count(hyp_id)
                logger.info(
                    "Rejected — Sharpe=%.2f DSR=%.3f DD=%.1f%% Activity=%.0f%% — %s",
                    result.sharpe,
                    result.dsr,
                    result.max_drawdown * 100,
                    result.activity_rate * 100,
                    justification[:120],
                )

        logger.info("Loop complete after %d iterations. Total cost: $%.3f", iteration, total_cost)
        n_active = self._memory.get_active_count()
        self._slack.post(
            f"*autoalpha loop complete* — {iteration} iterations  |  {n_active} active signals total"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate(self, trial_number: int) -> tuple[Optional[Hypothesis], int, float]:
        history = self._memory.get_all_for_trace()
        cohort_weights = self._memory.get_cohort_weight_summary(self._library.all_weights())
        system, user = build_generation_prompt(history, cohort_weights, trial_number)

        raw, cost = self._call_llm_json(system, user)
        return self._parse_and_store(raw, trial_number, cost)

    def _refine(self, pending: dict, trial_number: int) -> tuple[Optional[Hypothesis], int, float]:
        original_hyp = Hypothesis.from_json(pending["hypothesis_json"])
        backtest_result = {
            "sharpe": pending.get("sharpe") or 0.0,
            "dsr": pending.get("dsr") or 0.0,
            "max_drawdown": pending.get("max_drawdown") or 0.0,
        }
        system, user = build_refinement_prompt(original_hyp, backtest_result, trial_number)
        raw, cost = self._call_llm_json(system, user)
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
            logger.warning("Raw LLM text was: %r", raw[:500])
            return None, -1, cost

        hyp_id = self._memory.store_hypothesis(hyp, trial_number, cost_usd=cost)
        return hyp, hyp_id, cost

    def _baseline_alpha_passes(self, result: BacktestResult) -> tuple[bool, str]:
        """OLS-regress candidate returns on the live equal-weight book and require α
        with t ≥ _min_alpha_t(n_active).

        Bootstraps an empty book and short overlaps — falls back to standalone metrics in both cases.
        Returns (passes, one-line justification).
        """
        if not result.returns or not result.return_dates:
            return True, "no candidate returns"

        baseline = self._memory.get_portfolio_alpha()
        if baseline is None or len(baseline) == 0:
            return True, "no baseline (bootstrap)"

        cand = pd.Series(result.returns, index=pd.to_datetime(result.return_dates))
        df = pd.concat([cand.rename("c"), baseline.rename("b")], axis=1, sort=True).dropna()
        n = len(df)
        if n < _MIN_BASELINE_OVERLAP_DAYS:
            return True, f"overlap {n}d < {_MIN_BASELINE_OVERLAP_DAYS}d (insufficient)"

        y = df["c"].to_numpy()
        x = df["b"].to_numpy()
        x_mean = x.mean()
        Sxx = float(((x - x_mean) ** 2).sum())
        if Sxx == 0.0:
            return True, "baseline has zero variance"

        beta = float(((x - x_mean) * (y - y.mean())).sum() / Sxx)
        alpha = float(y.mean() - beta * x_mean)
        resid = y - (alpha + beta * x)
        s2 = float((resid ** 2).sum() / (n - 2))
        var_alpha = s2 * (1.0 / n + x_mean ** 2 / Sxx)
        se_alpha = float(np.sqrt(var_alpha)) if var_alpha > 0 else float("inf")
        t = alpha / se_alpha if se_alpha > 0 else 0.0

        n_active = self._memory.get_active_count()
        t_min = _min_alpha_t(n_active)
        ann_alpha = alpha * 252
        if t >= t_min:
            return True, f"α={ann_alpha:.3f}/yr t={t:.2f} ≥ {t_min:.2f} (N={n_active})"
        return False, (
            f"α={ann_alpha:.3f}/yr t={t:.2f} < {t_min:.2f} (N={n_active}) — "
            f"redundant with active book (overlap {n}d)"
        )

    def _accept(self, hyp: Hypothesis, hyp_id: int, result: BacktestResult) -> None:
        if result.returns and result.return_dates:
            idx = pd.to_datetime(result.return_dates)
            alpha = pd.Series(result.returns, index=idx)
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
        self._slack.post(
            f":white_check_mark: *ACCEPTED* — _{hyp.concise_reason}_\n"
            f"Sharpe={result.sharpe:.2f}  DSR={result.dsr:.3f}  DD={result.max_drawdown * 100:.1f}%  "
            f"active_signals={n_active}"
        )

    def _call_llm_json(self, system: str, user: str) -> tuple[str, float]:
        """Call LLM and retry once if the response is not parseable JSON."""
        raw, cost = self._call_llm(system, user)
        try:
            parse_llm_json(raw)
            return raw, cost
        except (ValueError, json.JSONDecodeError):
            logger.warning("JSON parse failed on first attempt — retrying with explicit instruction")
            retry_user = user + "\n\nIMPORTANT: Respond with ONLY the raw JSON object. No prose, no markdown fences, no commentary."
            raw2, cost2 = self._call_llm(system, retry_user)
            return raw2, cost + cost2

    def _call_llm(self, system: str, user: str) -> tuple[str, float]:
        """Call Claude via CLI (uses subscription) and return (response_text, 0.0)."""
        result = subprocess.run(
            [
                "claude", "-p",
                "--system-prompt", system,
                "--tools", "",
                "--no-session-persistence",
                "--model", self._model,
                user,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"claude CLI error: rc={result.returncode} "
                f"stderr={result.stderr!r} stdout={result.stdout[:1000]!r}"
            )
        text = result.stdout.strip()
        logger.debug("LLM raw response (first 500 chars): %s", text[:500])
        return text, 0.0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ResearchLoop":
        return self

    def __exit__(self, *_) -> None:
        self._memory.close()
        self._library.close()


