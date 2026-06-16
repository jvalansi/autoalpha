# Generalization Plan: autoalpha → generic metric-optimizer

Sketch for refactoring autoalpha into a domain-agnostic "given data + metric, search for a function that optimizes the metric" framework. Alpha research becomes one `Task` among many.

## 1. Define the `Task` protocol

New file `autoalpha/research/task.py`. A `Task` bundles everything domain-specific:

```python
class Task(Protocol):
    def load_data(self) -> Any: ...
    def split(self, data) -> Iterator[tuple[Train, Test]]: ...
    def evaluate(self, candidate, test_data, state: SearchState) -> EvalResult: ...
    def acceptance(self, result: EvalResult, state: SearchState) -> bool: ...
    def prompt_context(self) -> PromptContext: ...
```

- `EvalResult` is a free-form dict the `Task` owns; the loop only reads what `acceptance()` tells it.
- `SearchState` carries `n_trials`, `accepted_candidates`, and any artifacts prior tasks produced (e.g. portfolio returns for marginal-alpha checks).

## 2. Generalize the candidate interface

Rename `Strategy` → `Candidate`. Drop the `dict[str, float]` portfolio-fraction return type. Keep `fit(train) → None` and `predict(input) → Any`. The `Task`'s `evaluate()` is what interprets `Any`.

## 3. Pluggable splitter

Move CPCV out of `backtest/` into `autoalpha/research/splitters/`. Ship `KFold`, `TimeSeriesSplit`, and `CPCV` as built-ins. `Task.split()` chooses one. CPCV stays the autoalpha default — it's a `Task` choice, not a framework choice.

## 4. Refactor the loop

`research/loop.py` currently hard-codes `_DSR_THRESHOLD`, `_MAX_DRAWDOWN_THRESHOLD`, and calls Sharpe/drawdown directly. Replace with:

```python
result = task.evaluate(candidate, test_data, state)
if task.acceptance(result, state):
    memory.accept(candidate, result)
    state.accepted.append(candidate)
```

The loop no longer knows what "good" means.

## 5. Generalize prompts

`research/prompts.py` hard-codes "generate a trading signal." Pull the domain language into `Task.prompt_context()`:

- candidate signature description
- available features / data schema
- examples of accepted candidates
- domain priors ("prefer economically motivated hypotheses" → task-specific)

The prompt builder takes `PromptContext` and assembles a generic skeleton around it.

## 6. Port autoalpha into the new shape

`autoalpha/strategies/alpha_task.py` implements `Task`:

- `load_data` → existing data pipeline
- `split` → CPCV with embargo
- `evaluate` → DSR + drawdown + marginal alpha (uses `state.accepted_candidates`)
- `acceptance` → DSR > 0.65, DD < 0.30
- `prompt_context` → current alpha-research prompt content

This is the compatibility test: if alpha-research still works after the refactor, the abstraction is correct.

## 7. What stays untouched

- `research/hypothesis.py` (Hypothesis struct is already generic)
- `research/memory.py` (history of trials)
- `research/code_validator.py` (sandboxed code execution)
- `research/subprocess_runner.py` (isolation)
- Slack/Discord notifier plumbing

## 8. Out of scope for v1

- Multi-modal data (images, text): the `Task` protocol allows it, but prompts need work
- Online/streaming tasks: current loop assumes batched evaluation
- Hyperparameter search inside a candidate: candidates are still LLM-emitted code

## Where the statistical hygiene lives

A clarification, since this is the easy thing to get wrong:

- **DSR / PSR** → metric. Lives inside `Task.evaluate()`. User-suppliable.
- **CPCV splitter with embargo** → not a metric. Lives in `Task.split()` via the splitter abstraction.
- **Deflation by trial count** → metric-shaped, but needs `state.n_trials`. Hence `evaluate(..., state)` rather than `evaluate(predictions, data)`.
- **Marginal contribution to existing portfolio** → depends on `state.accepted_candidates`. Same reason.

So most overfit-control collapses into "the user wrote a good `evaluate()`" — provided the signature carries search state, not just this candidate's output.

## Order of work

1. Write `Task` + `SearchState` + `EvalResult` types (no behavior change)
2. Extract CPCV behind splitter interface
3. Build `AlphaTask` adapter that wraps current behavior — verify nightly run is unchanged
4. Move thresholds out of loop into `AlphaTask.acceptance`
5. Generalize prompts via `PromptContext`
6. Add a second `Task` (e.g. tabular regression on a UCI dataset) as a smoke test that the framework is genuinely generic

Step 3 is the gate — if the existing alpha loop produces byte-identical accepted hypotheses before and after, the abstraction held.
