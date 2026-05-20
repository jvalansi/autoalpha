# Implementation Plan

## Phase 1 — Data Foundation
Goal: reliable, point-in-time clean data with proper bar construction.

- [ ] `data/fetcher.py` — wrap FMP + yfinance fetchers from earnings-trader; add FRED macro fetcher
- [ ] `data/bars.py` — dollar bar constructor (sample on cumulative dollar volume threshold)
- [ ] `data/features.py` — fractional differentiation (implement López de Prado Ch. 5)
- [ ] `data/universe.py` — define tradeable universe (S&P 500 constituents, survivorship-bias-aware)
- [ ] Tests: verify no look-ahead leakage at data join seams

## Phase 2 — Labeling & Evaluation Engine
Goal: correct labeling and statistically honest backtesting.

- [ ] `labeling/triple_barrier.py` — triple-barrier label generator (profit-take, stop-loss, time expiry via ATR)
- [ ] `labeling/meta_label.py` — secondary labeling layer (did the primary signal actually work?)
- [ ] `backtest/cpcv.py` — Combinatorial Purged Cross-Validation splits
- [ ] `evaluation/sharpe.py` — deflated Sharpe ratio (accounts for number of trials)
- [ ] `evaluation/regime.py` — regime-conditional performance breakdown (bull/bear/sideways, vol regime)
- [ ] `evaluation/library.py` — signal library: store validated signals, compute pairwise correlations

## Phase 3 — Seed Strategy (PEAD)
Goal: validate pipeline end-to-end on a known working hypothesis.

- [ ] `strategies/pead.py` — port earnings-trader signal logic into the new framework
- [ ] Run Phase 1–2 pipeline on PEAD; verify it scores positively with deflated Sharpe
- [ ] Establish vault holdout (last 2 years of data, never touched until final validation)

## Phase 4 — LLM Hypothesis Loop
Goal: automated hypothesis generation and refinement.

- [ ] `research/prompts.py` — prompt templates for hypothesis generation, result interpretation, refinement
- [ ] `research/loop.py` — outer loop: generate → evaluate → store → refine
- [ ] `research/memory.py` — hypothesis library with causal rationales, scores, status
- [ ] Constraint: each hypothesis must include a written causal mechanism (reject pure curve-fitting)
- [ ] Logging: track all hypotheses tested, scores, LLM reasoning chains

## Phase 5 — Meta-Labeling & Execution
Goal: improve precision via secondary model; paper trading gate.

- [ ] `labeling/meta_model.py` — train meta-labeling classifier per strategy
- [ ] `execution/sizer.py` — fractional Kelly bet sizing based on meta-model confidence
- [ ] `execution/paper.py` — paper trading gate: promising signals → 30-day live sim before any capital
- [ ] Integration with earnings-trader execution layer (optional)

## Phase 6 — Productionization
Goal: scheduled runs, monitoring, alerting.

- [ ] Scheduled nightly research loop (new hypotheses + re-evaluate existing library)
- [ ] Slack notifications for new validated signals, decaying signals, regime changes
- [ ] Dashboard: signal library scores over time, regime tracker

---

## Non-Goals
- Live trading (earnings-trader handles that)
- Crypto / non-equity markets (out of scope for now)
- HFT / intraday signals (target holding period: days to weeks)

## Key Dependencies
- `mlfinlab` or manual implementations of AFML concepts
- `vectorbt` for fast backtesting
- `anthropic` SDK for LLM loop
- `pandas-market-calendars` for correct trading day math
