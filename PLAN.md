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

### Strategy Interface

`predict(bar_data) → dict[str, float]` is called on every bar for all strategies. Event detection is internal to the strategy — non-event bars return `{}`. No mode distinction in the Runner.

### Portfolio Overlays

Regime/risk overlays (e.g. trend-following scale-down on bearish regime) are not `Strategy` subclasses — they are handled in the `Executor` as a portfolio-level multiplier. This keeps the Strategy interface clean and per-stock. The `SimExecutor` and `LiveExecutor` both accept an optional `overlay` that scales all position targets before execution.

### Seed Strategies (Research Validation Set)

Five strategies covering the breadth of the strategy space, used to validate the interface and evaluation pipeline before the LLM loop runs:

| Strategy | Edge mechanism | Data source |
|---|---|---|
| PEAD | Analyst underreaction to earnings beats | FMP earnings + yfinance |
| Momentum (12-1) | Trend persistence across 150yr / 46 countries | yfinance |
| Earnings NLP | Tone/uncertainty in transcripts not priced in | FMP transcripts |
| Quality factor | Cheap high-quality ignored by market | FMP fundamentals |
| Earnings revisions | Slow analyst estimate updating post-earnings | FMP estimates |

Trend-following implemented as an overlay on the Executor, not a strategy.

---

## Phase 1 — Core Abstractions + Data Foundation ✓
Goal: establish the Strategy/DataProvider/Executor/Runner interfaces and wire up clean data.

- [x] `autoalpha/core/strategy.py` — abstract `Strategy` base; `fit(data: pd.DataFrame)` called once per CPCV fold on in-sample MultiIndex(date, ticker) data; `predict(bar_data: pd.DataFrame) → dict[str, float]` called on every bar where DataFrame index = ticker and columns = OHLCV + features; return values are portfolio fractions (0.02 = 2% long, -0.01 = 1% short, absent = flat); event strategies implement `fit` as no-op
- [x] `autoalpha/core/runner.py` — `Runner` wiring strategy + provider + executor; **fill model**: signals from bar N are executed at bar N+1's open (`prev_targets` pattern — no look-ahead); for CPCV backtests, fold returns sliced to each fold's OOS date range before accumulation (prevents duplicate dates across folds)
- [x] `autoalpha/core/providers.py` — `HistoricalProvider` (per-year Parquet cache at `data/cache/{ticker}/{year}.parquet`) and `LiveProvider`
- [x] `autoalpha/core/executors.py` — `SimExecutor` (P&L tracking) and `LiveExecutor` (broker API stub); both accept optional `overlay` scalar; `portfolio_value(prices)` takes current prices to value open positions correctly
- [x] `autoalpha/data/fetcher.py` — FMP + yfinance fetchers; vault holdout enforced on OHLCV, earnings, fundamentals, **and transcripts**; per-year Parquet cache; `get_fundamentals` returns `roe` and `net_margin` derived columns
- [x] `autoalpha/data/bars.py` — dollar bar constructor; fallback to daily bars logs `DataQualityWarning` with ADF p-value and lag-1 autocorrelation; **open: ADTV threshold not yet recomputed monthly** (issue #44); Polygon subscription required for real dollar bars
- [x] `autoalpha/data/features.py` — fractional differentiation (LdP Ch. 5); `MEMORY_CUTOFF=1e-3` (gives windows ≤74 bars for all d, consistent with LdP examples); `fracdiff` window determined by cutoff, not `len(series)`; `find_min_d` grid [0.0, 0.1, …, 1.0] refined to 0.01; d computed per ticker per CPCV fold
- [x] `autoalpha/data/universe.py` — Sharadar S&P 500 constituent history via Nasdaq Data Link; point-in-time join using effective entry/removal dates; handles tickers with multiple index memberships (removed and re-added)
- [x] **Lock vault holdout** — `vault_holdout.json` locked: 2024-05-21 to 2026-05-21; never evaluate any strategy against this period until final validation
- [x] Tests: 22 tests covering interfaces, vault enforcement, fill model correctness, fracdiff causality, dollar bar fallback, no look-ahead at data join seams

**Open items from Phase 1 review (non-blocking for Phase 2):**
- Issue #44 — Dollar bar ADTV threshold computed once per request; spec calls for monthly recomputation
- Issue #46 — `_aggregate_dollar_bars` uses `iterrows()`; vectorize before production use

> **Data dependency note:** Dollar bar construction requires historical minute OHLCV data. Polygon.io is the required source — confirm a subscription before starting `data/bars.py`. Daily dollar bars (using daily dollar volume) are an acceptable fallback if Polygon is unavailable. FRED macro fetcher is deferred to Phase 4+ when a hypothesis actually requires macro features.

## Phase 2 — Labeling & Evaluation Engine
Goal: correct labeling and statistically honest backtesting. This is the core differentiator vs. existing systems.

### Objective Function

A strategy is accepted if and only if:

1. **Marginal alpha Sharpe > 0** — Sharpe computed on Fama-French 5-factor residual returns (alpha), net of transaction costs (~5-10bps per trade baked into returns), evaluated as marginal contribution to the existing portfolio (standalone for the first strategy). Deflated for number of trials tested.
2. **Max drawdown < 25%** — hard constraint regardless of Sharpe; protects against undeployable loss profiles

No separate beta constraint (captured by FF5 alpha), no separate turnover penalty (baked into net returns), no separate correlation penalty (captured by marginal contribution).

### Tasks

- [ ] `autoalpha/labeling/triple_barrier.py` — triple-barrier label generator; **default parameters**: profit-take = 2×ATR(21), stop-loss = 1×ATR(21), time expiry = 20 trading days; ATR on daily close-to-close; all three configurable per strategy (PEAD: 10-day expiry, Momentum: 63-day); closes #33
- [ ] `autoalpha/labeling/meta_label.py` — secondary labeling layer (did the primary signal actually work?); **backtest inclusion**: meta-model must be trained on each CPCV fold's in-sample data and applied to that fold's out-of-sample data — never train on the full dataset and apply globally, which would introduce look-ahead; closes #11
- [ ] `autoalpha/backtest/cpcv.py` — Combinatorial Purged Cross-Validation splits; **purging gap = label horizon h** (max triple-barrier expiry, typically 20 trading days); add an optional embargo of 5 days after the gap to further reduce autocorrelation leakage; both parameters must be matched to each strategy's label horizon at evaluation time (closes #22)
- [ ] `autoalpha/evaluation/alpha.py` — Fama-French 5-factor regression; compute alpha return series and residual Sharpe; fetch FF5 factors from Kenneth French's data library via `pandas_datareader.famafrench.FamaFrenchReader("F-F_Research_Data_5_Factors_2x3_daily")` (daily granularity, free, no API key)
- [ ] `autoalpha/evaluation/costs.py` — transaction cost model; deduct from returns before Sharpe computation; **default parameters**: half-spread 2 bps, commission 0.5 bps, market impact 3 bps (assuming ~10% of daily ADV participation) → **~11 bps total round-trip**; all parameters must be configurable; stress-test seed strategies at 2× base cost; closes #26
- [ ] `autoalpha/evaluation/sharpe.py` — deflated Sharpe ratio applied to alpha returns; trial count `T` is loaded from `autoalpha/research/memory.py` (cumulative count of all strategies ever evaluated, including rejected ones); `T` must persist across sessions — never reset (closes #23)
- [ ] `autoalpha/evaluation/drawdown.py` — max drawdown computation and hard constraint check
- [ ] `autoalpha/evaluation/marginal.py` — marginal Sharpe contribution; **strategy 1**: use standalone alpha Sharpe (no portfolio to be marginal to); **strategy 2+**: regress new strategy's alpha returns against existing portfolio's alpha returns; marginal Sharpe = Sharpe of the residual; existing portfolio's daily alpha returns stored in `research/memory.db` alongside each validated hypothesis; only new strategies evaluated on marginal basis — strategy 1's standalone Sharpe is not retroactively recomputed; closes #39
- [ ] `autoalpha/evaluation/regime.py` — regime-conditional performance breakdown; **trend regime**: bull = SPY 63-day return > +5%, bear < -5%, sideways otherwise; **vol regime**: high = SPY 21-day realized vol > 20% annualized, low otherwise; VIX available via yfinance (`^VIX`) as alternative vol signal; closes #35
- [ ] `autoalpha/evaluation/library.py` — signal library with Darwinian weights: signals start at 1.0, updated daily on **63-trading-day rolling alpha Sharpe** (≈ 1 quarter; annualized before comparison across strategies); floor 0.3, ceiling 2.5; update trigger: end of each trading day; **decay**: weight at floor for ≥ 63 consecutive days → status = 'decayed' (still traded at floor weight); **death**: weight at floor for ≥ 126 consecutive days → status = 'dead', removed from live trading and from portfolio return series; all transitions logged in `research/memory.db`; closes #24, #42

## Phase 3 — Seed Strategies
Goal: implement all 5 seed strategies as `Strategy` subclasses; validate pipeline end-to-end and confirm the interface handles both modes cleanly.

- [ ] `autoalpha/strategies/pead.py` — earnings beat + AH confirmation; **beat definition**: EPS actual > EPS consensus AND revenue actual > revenue consensus (both must beat; source: FMP earnings endpoint); **AH confirmation**: after-hours close ≥ 1% above prior regular-session close; **entry**: next regular-session open; **hold**: 10 trading days; triple-barrier expiry = 10 days, loose barriers (3×ATR profit, 1.5×ATR stop); reuse earnings-trader fetcher logic; returns `{}` on non-earnings bars; closes #38
- [ ] `autoalpha/strategies/momentum.py` — 12-1 month cross-sectional ranking; signal = cumulative return from `t-252` to `t-21` trading days (skip most recent 21 days to avoid short-term reversal); rank cross-sectionally, long top quintile; rebalance monthly on first trading day; closes #37
- [ ] `autoalpha/strategies/earnings_nlp.py` — FMP transcript tone/uncertainty scoring; returns `{}` on non-transcript bars
- [ ] `autoalpha/strategies/quality.py` — quality factor composite; **signal** = z_score(ROE) - z_score(leverage) + z_score(net_margin), all cross-sectional z-scores at rebalance; leverage = net_debt / (net_debt + market_cap); **rebalance**: quarterly on first trading day after new FMP fundamentals; long top quintile; returns `{}` on non-rebalance bars; closes #41
- [ ] `autoalpha/strategies/earnings_revisions.py` — FMP estimate delta signal; returns `{}` on non-revision bars
- [ ] Run each through `Runner(strategy, Historical, Sim)` with CPCV; verify deflated Sharpe is positive for at least PEAD and momentum

## Phase 4 — LLM Hypothesis Loop
Goal: automated hypothesis generation and refinement, modelled on RD-Agent's trace structure.

- [ ] `autoalpha/research/hypothesis.py` — `Hypothesis` dataclass: hypothesis, reason, concise_reason, observation, justification, knowledge (causal mechanism required — rejects pure curve-fitting)
- [ ] `autoalpha/research/prompts.py` — prompt templates for generation, result interpretation, refinement; each round receives full trace of prior hypotheses + feedback; **LLM output format**: the LLM generates only the `predict(self, bar_data)` method body (not a full class); the harness wraps it in a `Strategy` subclass automatically — this simplifies the subprocess timeout, keeps the interface contract intact, and makes AST validation trivial; closes #31
- [ ] `autoalpha/research/loop.py` — outer loop: generate Strategy → Runner(Historical, Sim) → evaluate → store → refine; run generated code in a subprocess with a 60s timeout (EC2 instance is itself the sandbox); **budget controls**: `max_iterations=20` per run (default), `max_cost_usd=5.00` per run — track token usage from API responses and stop early if exceeded; log total cost per run; closes #32
- [ ] `autoalpha/research/memory.py` — hypothesis library stored in **SQLite** (`research/memory.db`); schema: `hypotheses(id, created_at, status TEXT, sharpe REAL, cost_usd REAL, trial_number INT, hypothesis_json TEXT)`; SQLite enables `SELECT WHERE status='active'` queries without loading full history; `trial_number` is the monotonic counter fed to `evaluation/sharpe.py` as `T`; closes #36
- [ ] Regime detection: monitor relative Darwinian weight shifts across signal types (momentum, value, quality, macro) as an emergent regime signal — inspired by ATLAS's cohort weight differential

## Phase 5 — Meta-Labeling & Deployment
Goal: improve precision via secondary model; graduate validated strategies to paper then live.

- [ ] `autoalpha/labeling/meta_model.py` — train meta-labeling classifier per strategy
- [ ] `autoalpha/execution/sizer.py` — fractional Kelly bet sizing weighted by Darwinian signal weights and meta-model confidence; **Kelly fraction = 0.25** (quarter-Kelly; reduces median drawdown ~75% vs full Kelly); final position = `0.25 × kelly_bet × darwinian_weight × meta_confidence`; hard cap: no single position > 5% of portfolio; closes #29
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
- `vectorbt==0.26.2` for fast backtesting (not Qlib); pinned because upstream is unmaintained — author moved to paid `vectorbtpro`
- `anthropic` SDK for LLM loop
- `pandas-market-calendars` for correct trading day math
- CPCV, triple-barrier, and deflated Sharpe implemented from scratch (López de Prado books/papers as reference); do not depend on `mlfinlab` — largely unmaintained with known bugs in CPCV
- **Polygon.io** (historical minute OHLCV) — hard dependency for real dollar bars; yfinance only provides ~60 days of minute data
- **Sharadar via Nasdaq Data Link** (~$40/month) — survivorship-bias-free S&P 500 constituent history for `data/universe.py`
