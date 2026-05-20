# Implementation Plan

## Prior Art & Design Decisions

We surveyed the two closest existing systems before building:

### RD-Agent (Microsoft)
Full R&D automation loop integrated with Qlib. Well-implemented core loop and structured `Hypothesis` object (fields: hypothesis, reason, observation, justification, knowledge). However: uses plain time bars (not dollar bars), standard train/val/test splits (not CPCV), IC/Rank-IC evaluation (not deflated Sharpe), no triple-barrier labeling, no meta-labeling. Heavy abstraction (~10 layers of ABC classes) and Docker-based factor execution.

**Borrow:** `Hypothesis` struct design and trace-fed prompt pattern (each round receives full history of prior hypotheses + feedback).  
**Skip:** Qlib entirely. Use vectorbt with our own data pipeline instead.

### ATLAS (General Intelligence Capital)
Applies Karpathy-style autoresearch to markets: LLM agents give LONG/SHORT/conviction calls, agent *prompts* are the weights, rolling Sharpe is the loss. Darwinian weighting gives high-performing agents up to 2.5x influence, poor ones silenced to 0.3x. Regime detection via relative performance of short-horizon vs. long-horizon agent cohorts.

Results after 378 days: Financials Sharpe -4.14 → 0.45, EM Sharpe -0.45 → -0.06 (still negative). No overfitting protection.

**Borrow:** Darwinian weighting for the signal library — validated signals that continue performing get higher allocation weight (floor 0.3, ceiling 2.5); decaying signals get reduced. Regime detection via relative weight shifts across signal cohorts (momentum vs. value vs. macro).  
**Skip:** The "LLMs as analysts" paradigm. We generate quantitative factor code, not qualitative opinions.

### Our Differentiator
Neither RD-Agent nor ATLAS has statistical rigor against overfitting. Our core differentiator is the CPCV + deflated Sharpe evaluation layer — this is what separates real edges from mined noise.

---

## Core Architecture

The central design principle is a **pure-function strategy** that runs identically in backtesting, paper trading, and live trading. There is no translation step between research and production.

```
Strategy(data) → action (target position, e.g. "2% of portfolio in AAPL")

DataProvider (abstract)
  ├── HistoricalProvider  — replays bars from disk
  └── LiveProvider        — streams from yfinance / Polygon

Executor (abstract)
  ├── SimExecutor         — fake fills, tracks P&L in memory
  └── LiveExecutor        — real broker API

Runner(strategy, data_provider, executor)
  — backtest:       Runner(strategy, Historical, Sim)
  — paper trading:  Runner(strategy, Live, Sim)
  — live trading:   Runner(strategy, Live, Live)
```

Action is always a **target position** (stateless, portfolio-relative), not an order. The executor converts target → order. This makes strategies composable and executor-agnostic.

Autoalpha's job: generate Strategy → evaluate via Runner(Historical, Sim) → update → repeat. Paper trading and live deployment reuse the same Runner and Strategy with no porting step.

### Strategy Modes

Strategies run in one of two modes, declared via `strategy.mode`:

- **`bar`** — `predict` is called on every bar with the full universe snapshot (e.g. momentum, quality). Input: `dict[ticker, pd.DataFrame]`. Used for cross-sectional ranking strategies.
- **`event`** — `predict` is called only when a qualifying event is detected (e.g. earnings release, transcript drop). Input: event object. Used for event-driven strategies (PEAD, earnings NLP, earnings revisions). The Runner passes no-ops on non-event bars.

### Portfolio Overlays

Regime/risk overlays (e.g. trend-following scale-down on bearish regime) are not `Strategy` subclasses — they are handled in the `Executor` as a portfolio-level multiplier. This keeps the Strategy interface clean and per-stock. The `SimExecutor` and `LiveExecutor` both accept an optional `overlay` that scales all position targets before execution.

### Seed Strategies (Research Validation Set)

Five strategies covering the breadth of the strategy space, used to validate the interface and evaluation pipeline before the LLM loop runs:

| Strategy | Mode | Edge mechanism | Data source |
|---|---|---|---|
| PEAD | event | Analyst underreaction to earnings beats | FMP earnings + yfinance |
| Momentum (12-1) | bar | Trend persistence across 150yr / 46 countries | yfinance |
| Earnings NLP | event | Tone/uncertainty in transcripts not priced in | FMP transcripts |
| Quality factor | bar | Cheap high-quality ignored by market | FMP fundamentals |
| Earnings revisions | event | Slow analyst estimate updating post-earnings | FMP estimates |

Trend-following implemented as an overlay on the Executor, not a strategy.

---

## Phase 1 — Core Abstractions + Data Foundation
Goal: establish the Strategy/DataProvider/Executor/Runner interfaces and wire up clean data.

- [ ] `core/strategy.py` — abstract `Strategy` base: `fit(data)`, `predict(data) → dict[str, float]`, `mode: "bar" | "event"`
- [ ] `core/runner.py` — `Runner` wiring strategy + provider + executor; dispatches bar vs. event mode
- [ ] `core/providers.py` — `HistoricalProvider` and `LiveProvider`; event detection for event-mode strategies
- [ ] `core/executors.py` — `SimExecutor` (P&L tracking) and `LiveExecutor` (broker API stub); both accept optional `overlay` scalar
- [ ] `data/fetcher.py` — wrap FMP + yfinance fetchers from earnings-trader; add FRED macro fetcher
- [ ] `data/bars.py` — dollar bar constructor (sample on cumulative dollar volume threshold)
- [ ] `data/features.py` — fractional differentiation (implement López de Prado Ch. 5)
- [ ] `data/universe.py` — define tradeable universe (S&P 500 constituents, survivorship-bias-aware)
- [ ] Tests: verify no look-ahead leakage at data join seams

## Phase 2 — Labeling & Evaluation Engine
Goal: correct labeling and statistically honest backtesting. This is the core differentiator vs. existing systems.

- [ ] `labeling/triple_barrier.py` — triple-barrier label generator (profit-take, stop-loss, time expiry via ATR)
- [ ] `labeling/meta_label.py` — secondary labeling layer (did the primary signal actually work?)
- [ ] `backtest/cpcv.py` — Combinatorial Purged Cross-Validation splits
- [ ] `evaluation/sharpe.py` — deflated Sharpe ratio (penalizes for number of trials tested)
- [ ] `evaluation/regime.py` — regime-conditional performance breakdown (bull/bear/sideways, vol regime)
- [ ] `evaluation/library.py` — signal library with Darwinian weights: signals start at 1.0, updated daily on rolling Sharpe (floor 0.3, ceiling 2.5); pairwise correlation tracking to penalize redundant signals

## Phase 3 — Seed Strategies
Goal: implement all 5 seed strategies as `Strategy` subclasses; validate pipeline end-to-end and confirm the interface handles both modes cleanly.

- [ ] `strategies/pead.py` — event mode; port earnings-trader signal logic
- [ ] `strategies/momentum.py` — bar mode; 12-1 month cross-sectional ranking
- [ ] `strategies/earnings_nlp.py` — event mode; FMP transcript tone/uncertainty scoring
- [ ] `strategies/quality.py` — bar mode; ROE + debt + margin composite, quarterly rebalance
- [ ] `strategies/earnings_revisions.py` — event mode; FMP estimate delta signal
- [ ] Run each through `Runner(strategy, Historical, Sim)` with CPCV; verify deflated Sharpe is positive for at least PEAD and momentum
- [ ] Establish vault holdout (last 2 years of data, never touched until final validation)

## Phase 4 — LLM Hypothesis Loop
Goal: automated hypothesis generation and refinement, modelled on RD-Agent's trace structure.

- [ ] `research/hypothesis.py` — `Hypothesis` dataclass: hypothesis, reason, concise_reason, observation, justification, knowledge (causal mechanism required — rejects pure curve-fitting)
- [ ] `research/prompts.py` — prompt templates for generation, result interpretation, refinement; each round receives full trace of prior hypotheses + feedback
- [ ] `research/loop.py` — outer loop: generate Strategy → Runner(Historical, Sim) → evaluate → store → refine
- [ ] `research/memory.py` — hypothesis library: scores, status (active/decayed/rejected), LLM reasoning chains
- [ ] Regime detection: monitor relative Darwinian weight shifts across signal types (momentum, value, quality, macro) as an emergent regime signal — inspired by ATLAS's cohort weight differential

## Phase 5 — Meta-Labeling & Deployment
Goal: improve precision via secondary model; graduate validated strategies to paper then live.

- [ ] `labeling/meta_model.py` — train meta-labeling classifier per strategy
- [ ] `execution/sizer.py` — fractional Kelly bet sizing weighted by Darwinian signal weights and meta-model confidence
- [ ] Paper trading: `Runner(strategy, Live, Sim)` — no new code, just a different Runner config; run for 30 days before live
- [ ] Live trading: `Runner(strategy, Live, Live)` — implement `LiveExecutor` with real broker API
- [ ] Promotion pipeline: backtest passes → paper 30 days → live (gated on continued Sharpe)

## Phase 6 — Productionization
Goal: scheduled runs, monitoring, alerting.

- [ ] Scheduled nightly research loop (new hypotheses + re-evaluate existing library)
- [ ] Slack notifications for new validated signals, decaying signals, regime shifts
- [ ] Dashboard: signal library Darwinian weights over time, regime tracker

---

## Non-Goals
- Crypto / non-equity markets (out of scope for now)
- HFT / intraday signals (target holding period: days to weeks)
- Qualitative LLM analyst opinions (ATLAS paradigm) — we generate quantitative factor code only

## Key Dependencies
- `vectorbt` for fast backtesting (not Qlib)
- `anthropic` SDK for LLM loop
- `pandas-market-calendars` for correct trading day math
- `mlfinlab` or manual implementations of AFML concepts (CPCV, triple-barrier, deflated Sharpe)
