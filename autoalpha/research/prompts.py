"""Prompt templates for the LLM hypothesis loop.

Each build_*_prompt function returns a (system, user) tuple.
The LLM must respond with a JSON object — no prose, no markdown fences.
"""
from __future__ import annotations

import json
import re

from autoalpha.research.hypothesis import Hypothesis

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a quantitative research assistant specializing in equity alpha factor research.
Your task is to generate or refine trading hypotheses grounded in genuine economic causation.

## Response format
Respond ONLY with a raw JSON object (no markdown fences, no commentary):
{
  "hypothesis":      "<one-sentence description of the factor>",
  "reason":          "<why this factor should predict returns>",
  "concise_reason":  "<≤5-word signal name, e.g. 'earnings quality beat'>",
  "knowledge":       "<causal mechanism — must be ≥20 words explaining the economic reason, NOT just empirical correlation>",
  "predict_body":    "<Python source for the body of predict(), indented 8 spaces>",
  "cohort":          "<one of: momentum, value, quality, macro>"
}

## predict() contract
The method signature is:
    def predict(self, bar_data: pd.DataFrame, bar_date: pd.Timestamp | None = None) -> dict[str, float]:

Arguments available inside predict():
- `bar_data` — DataFrame indexed by ticker with columns:
    Open, High, Low, Close, Volume   (current bar OHLCV)
    ret_1d, ret_5d, ret_21d, ret_63d, ret_252d  (lookback returns, pre-computed)
    pe_ratio, pb_ratio, ps_ratio, ev_ebitda      (valuation multiples)
    roe, net_margin                              (fundamental ratios — only these two are currently available)
    earnings_surprise, revenue_surprise          (most recent quarter vs estimate)
    analyst_revision_3m                          (3-month EPS estimate revision %)
    vix, yield_10y, yield_2y, credit_spread      (macro — same for all tickers)
    sentiment_score                              (news/earnings-call NLP score)
- `bar_date` — the current bar's date (may be None; guard with `if bar_date is None: return {}`)
- `self._price_history` — NOT available; do not reference it

## Returns
A dict `{ticker: weight}` for tickers to hold. Weights must be positive and sum to 1.0.
Return {} to hold cash.

## Causal mechanism rule
The `knowledge` field must explain the economic mechanism — why the factor CAUSES future returns.
Do NOT write statements like "historically correlated", "backtested well", "data showed", "in the past".
Write the supply-demand or behavioral-finance mechanism that links the factor to future price moves.

## Constraints
- predict() must return in < 1 second
- No I/O, no network calls, no imports inside predict()
- Use only `bar_data` columns listed above; do not assume other columns exist
- Handle missing data: guard with `.dropna()` or `.fillna()` before ranking
"""

# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------

def build_generation_prompt(
    history_summary: list[dict],
    cohort_weights: dict[str, float],
    trial_number: int,
) -> tuple[str, str]:
    """Return (system, user) for a fresh hypothesis generation request."""
    history_text = _format_history(history_summary)
    cohort_text = _format_cohort_weights(cohort_weights)

    user = f"""## Research history (trial 1 to {trial_number - 1})
{history_text}

## Current cohort performance (Darwinian weights)
{cohort_text}

## Task
Generate trial #{trial_number}.

Guidelines:
- Do NOT repeat a hypothesis already in the history.
- Favour cohorts with lower Darwinian weights (they need more exploration).
- Focus on a factor with a clear economic causal mechanism.
- The predict() body should rank tickers by the factor and return equal weights for the top quintile.
"""
    return SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# Refinement prompt
# ---------------------------------------------------------------------------

def build_refinement_prompt(
    original_hyp: Hypothesis,
    backtest_result: dict,
    trial_number: int,
) -> tuple[str, str]:
    """Return (system, user) to refine a rejected hypothesis."""
    user = f"""## Trial to refine (signal: {original_hyp.concise_reason})
Original hypothesis: {original_hyp.hypothesis}
Reason: {original_hyp.reason}
Knowledge: {original_hyp.knowledge}
Cohort: {original_hyp.cohort}

## Original predict() implementation
```python
{original_hyp.predict_body}
```

## Backtest result
Sharpe: {backtest_result.get('sharpe', 'N/A')}
DSR: {backtest_result.get('dsr', 'N/A')}
Max drawdown: {backtest_result.get('max_drawdown', 'N/A')}
Observation: {original_hyp.observation}
Rejection justification: {original_hyp.justification}

## Task
Produce trial #{trial_number} — a refined version of the above hypothesis.

Guidelines:
- Keep what worked; fix what the observation/justification identified as broken.
- Change the signal construction, lookback window, or ranking logic as needed.
- Do NOT simply re-submit the same predict() body — the improvement must be substantive.
- Update `concise_reason` only if the signal is meaningfully different.
"""
    return SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# Interpretation prompt
# ---------------------------------------------------------------------------

def build_interpretation_prompt(
    hypothesis: Hypothesis,
    backtest_result: dict,
) -> tuple[str, str]:
    """Return (system, user) to interpret a backtest result and decide accept/reject."""
    system = (
        "You are a quantitative research analyst reviewing a backtest result. "
        "Respond ONLY with a raw JSON object (no markdown fences):\n"
        '{"observation": "<what the data showed>", '
        '"justification": "<why to accept or reject>", '
        '"status": "<accepted or rejected>"}'
    )

    result_text = json.dumps(backtest_result, indent=2, default=str)
    user = f"""## Hypothesis
{hypothesis.hypothesis}

Reason: {hypothesis.reason}
Knowledge: {hypothesis.knowledge}
Cohort: {hypothesis.cohort}

## Backtest result
{result_text}

## Task
Acceptance criteria (ALL must hold):
- DSR > 0.95
- Max drawdown < 25 %

Write a concise observation (what the data showed) and justification (why accept or reject).
Set status to "accepted" only if both criteria are met.
"""
    return system, user


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def parse_llm_json(text: str) -> dict:
    """Strip optional markdown fences and parse JSON.

    Raises ValueError if the text is not valid JSON after stripping.
    """
    stripped = text.strip()
    # Remove ```json ... ``` or ``` ... ``` fences
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    stripped = stripped.strip()
    return json.loads(stripped)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _format_history(history: list[dict]) -> str:
    if not history:
        return "(no history yet)"
    lines = []
    for h in history:
        hyp_data = json.loads(h["hypothesis_json"]) if isinstance(h["hypothesis_json"], str) else h["hypothesis_json"]
        lines.append(
            f"  Trial {h['trial_number']} [{h['status']}] "
            f"Sharpe={h.get('sharpe') or 'N/A'} DSR={h.get('dsr') or 'N/A'} "
            f"cohort={h.get('cohort') or '?'} — {hyp_data.get('hypothesis', '')}"
        )
    return "\n".join(lines)


def _format_cohort_weights(weights: dict[str, float]) -> str:
    if not weights:
        return "(no active signals yet)"
    return "\n".join(f"  {cohort}: {w:.3f}" for cohort, w in sorted(weights.items()))
