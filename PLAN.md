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

- [x] `autoalpha/labeling/triple_barrier.py` — triple-barrier label generator; **default parameters**: profit-take = 2×ATR(21), stop-loss = 1×ATR(21), time expiry = 20 trading days; ATR on daily close-to-close; all three configurable per strategy (PEAD: 10-day expiry, Momentum: 63-day); closes #33, #49; **fixed**: added optional `entry_prices` parameter — strategies that fill at next-bar open pass their actual fill prices; falls back to `close[event_date]` when absent
- [x] `autoalpha/labeling/meta_label.py` — secondary labeling layer (did the primary signal actually work?); **backtest inclusion**: meta-model must be trained on each CPCV fold's in-sample data and applied to that fold's out-of-sample data — never train on the full dataset and apply globally, which would introduce look-ahead; closes #11; **fixed**: degenerate fallback now returns 0.0 when all primary signals are wrong (was incorrectly returning 1.0)
- [x] `autoalpha/backtest/cpcv.py` — Combinatorial Purged Cross-Validation splits; **purging gap = label horizon h** (max triple-barrier expiry, typically 20 trading days); add an optional embargo of 5 days after the gap to further reduce autocorrelation leakage; both parameters must be matched to each strategy's label horizon at evaluation time (closes #22); **fixed**: purge/embargo gap now uses `pd.offsets.BDay` (trading days) instead of `pd.Timedelta` (calendar days) — the prior code only purged ~14 trading days for a 20-day label horizon; **open: #51** (`to_runner_folds` drops purge metadata, must be resolved before Phase 4 Runner integration)
- [x] `autoalpha/evaluation/alpha.py` — Fama-French 5-factor regression; compute alpha return series and residual Sharpe; fetch FF5 factors from Kenneth French's data library via `pandas_datareader.famafrench.FamaFrenchReader("F-F_Research_Data_5_Factors_2x3_daily")` (daily granularity, free, no API key)
- [x] `autoalpha/evaluation/costs.py` — transaction cost model; deduct from returns before Sharpe computation; **default parameters**: half-spread 2 bps, commission 0.5 bps, market impact 3 bps (assuming ~10% of daily ADV participation) → **~11 bps total round-trip**; all parameters must be configurable; stress-test seed strategies at 2× base cost; closes #26
- [x] `autoalpha/evaluation/sharpe.py` — deflated Sharpe ratio applied to alpha returns; trial count `T` is loaded from `autoalpha/research/memory.py` (cumulative count of all strategies ever evaluated, including rejected ones); `T` must persist across sessions — never reset (closes #23)
- [x] `autoalpha/evaluation/drawdown.py` — max drawdown computation and hard constraint check; **fixed**: NAV series now anchored at 1.0 before `cumprod()` so initial losses are captured — prior code missed drawdowns that begin on the first bar
- [x] `autoalpha/evaluation/marginal.py` — marginal Sharpe contribution; **strategy 1**: use standalone alpha Sharpe (no portfolio to be marginal to); **strategy 2+**: regress new strategy's alpha returns against existing portfolio's alpha returns; marginal Sharpe = Sharpe of the residual; existing portfolio's daily alpha returns stored in `research/memory.db` alongside each validated hypothesis; only new strategies evaluated on marginal basis — strategy 1's standalone Sharpe is not retroactively recomputed; closes #39; **open: #50** (`combine_portfolio_alpha` uses hardcoded weight=0.5; must use actual allocation weight in Phase 4)
- [x] `autoalpha/evaluation/regime.py` — regime-conditional performance breakdown; **trend regime**: bull = SPY 63-day return > +5%, bear < -5%, sideways otherwise; **vol regime**: high = SPY 21-day realized vol > 20% annualized, low otherwise; VIX available via yfinance (`^VIX`) as alternative vol signal; closes #35; **fixed**: warm-up bars (first 63 for trend, first 21 for vol) now set to None instead of being misclassified as 'sideways'/'low'
- [x] `autoalpha/evaluation/library.py` — signal library with Darwinian weights: signals start at 1.0, updated daily on **63-trading-day rolling alpha Sharpe** (≈ 1 quarter; annualized before comparison across strategies); floor 0.3, ceiling 2.5; update trigger: end of each trading day; **decay**: weight at floor for ≥ 63 consecutive days → status = 'decayed' (still traded at floor weight); **death**: weight at floor for ≥ 126 consecutive days → status = 'dead', removed from live trading and from portfolio return series; all transitions logged in `research/memory.db`; closes #24, #42; **open: #52** (relative default DB path and connection leak — must fix before Phase 4)

**Tests: 52 passing (phase 2 only); 82 passing (phases 1 + 2 combined)**

**Interface change (Phase 3):** `Strategy.predict(bar_data, bar_date=None)` — added optional `bar_date: pd.Timestamp` parameter; Runner passes it on every bar. All strategies must accept it. Backward-compatible (default=None).

## Phase 3 — Seed Strategies
Goal: implement all 5 seed strategies as `Strategy` subclasses; validate pipeline end-to-end and confirm the interface handles both modes cleanly.

- [x] `autoalpha/strategies/pead.py` — earnings beat + AH confirmation; **beat definition**: EPS actual > EPS consensus AND revenue actual > revenue consensus (both must beat; source: FMP earnings endpoint); **AH confirmation**: after-hours close ≥ 1% above prior regular-session close; **entry**: next regular-session open; **hold**: 10 trading days; triple-barrier expiry = 10 days, loose barriers (3×ATR profit, 1.5×ATR stop); reuse earnings-trader fetcher logic; returns `{}` on non-earnings bars; closes #38
- [x] `autoalpha/strategies/momentum.py` — 12-1 month cross-sectional ranking; signal = cumulative return from `t-252` to `t-21` trading days (skip most recent 21 days to avoid short-term reversal); rank cross-sectionally, long top quintile; rebalance monthly on first trading day; seeds price buffer from `fit()` for immediate warmup; closes #37
- [x] `autoalpha/strategies/earnings_nlp.py` — FMP transcript tone/uncertainty scoring (Loughran-McDonald lexicon subset); signal = (pos - neg)/n - 0.5 * unc/n; quarterly evaluation using prior quarter's transcript; 63-day hold; returns `{}` on non-transcript bars
- [x] `autoalpha/strategies/quality.py` — quality factor composite; **signal** = z_score(ROE) - z_score(leverage) + z_score(net_margin), all cross-sectional z-scores at rebalance; leverage proxy = net_debt / (net_debt.abs() + 1B) (no market cap in FMP fundamentals); **rebalance**: quarterly on first trading day after new FMP fundamentals; long top quintile; returns `{}` on non-rebalance bars; closes #41
- [x] `autoalpha/strategies/earnings_revisions.py` — FMP estimate delta signal; QoQ EPS estimate upward revision > 5%; 21-day hold; returns `{}` on non-revision bars
- [x] Run each through `Runner(strategy, Historical, Sim)` with CPCV (`scripts/validate_phase3.py`); confirmed positive OOS Ann.SR: PEAD=1.308, Quality=0.685, Momentum=0.226; EarningsRevisions/NLP rate-limited (see #54); DSR=0.000 for all due to unit bug in `deflated_sharpe` (see #53); fixes: executor DatetimeIndex and NLP quarter cap (vault guard)

**Tests: 32 new (phase 3: 10 momentum + 22 strategies); 114 passing total**

**Known bugs from validation (to fix in Phase 4):**
- #53 `deflated_sharpe` DSR always ≈ 0 — `expected_max_sr` unit mismatch (per-period vs annualized)
- #54 FMP fetchers lack caching — 250 API calls per strategy per run causes HTTP 429s

## Phase 4 — LLM Hypothesis Loop ✓
Goal: automated hypothesis generation and refinement, modelled on RD-Agent's trace structure.

- [x] `autoalpha/research/hypothesis.py` — `Hypothesis` dataclass with causal mechanism enforcement; `_CURVE_FITTING_PHRASES` rejects pure curve-fitting language; `_validate_knowledge` requires ≥20-word causal explanation; cohort validation (momentum / value / quality / macro)
- [x] `autoalpha/research/prompts.py` — generation / interpretation / refinement prompt templates; full hypothesis trace fed to each round; **LLM output format**: `predict(self, bar_data)` body only; harness wraps in Strategy subclass; closes #31; **fixed**: `parse_llm_json` extracts JSON from prose preamble when model ignores no-commentary instruction; prompt lists only non-NaN columns (roe, net_margin, earnings_surprise, revenue_surprise, vix); universe size (49 tickers) noted to prevent large-universe guards
- [x] `autoalpha/research/loop.py` — outer loop: generate → validate → subprocess backtest → interpret → accept/reject/refine; budget controls (`max_iterations`, `max_cost_usd`); prompt caching on system message; closes #32; **fixed**: max drawdown acceptance threshold raised to 60% (long-only concentrated universe); `_MAX_DRAWDOWN_THRESHOLD = 0.60`
- [x] `autoalpha/research/memory.py` — SQLite-backed hypothesis library; `get_pending_refinement` returns sharpe/dsr/max_drawdown so loop doesn't need raw DB access; closes #36
- [x] `autoalpha/research/code_validator.py` — AST validation rejects imports, exec, eval, network calls, file I/O; length cap 4000 chars; wraps predict body in Strategy subclass
- [x] `autoalpha/research/subprocess_runner.py` — executes LLM-generated code in isolated child process; **fixed**: injects project root into child PYTHONPATH; pre-builds date→bar lookup dict (was 27s baseline, now ~8s); averages duplicate CPCV dates before Sharpe/drawdown computation (each date appeared 5× in raw concat, causing phantom 96% drawdowns); timeout raised 60s→300s
- [x] `scripts/build_loop_dataset.py` — builds MultiIndex (date, ticker) parquet with OHLCV, ret_1d/5d/21d/63d/252d, roe, net_margin, earnings/revenue surprise, vix; vault-safe (all dates < 2024-05-21)
- [x] `scripts/run_loop.py` — CLI entry point; `--iterations`, `--budget`, `--model`, `--data` args
- [x] `scripts/fetch_universe.py` — idempotent fetcher for expanded 49-stock S&P 500 universe; skips already-cached tickers; FMP rate-limiting via 0.25s sleep
- [x] `scripts/build_vault_dataset.py` — builds vault holdout dataset (2024-05-21 → today) using full history for accurate 252d lookback returns; same schema as loop_data.parquet
- [x] `scripts/evaluate_vault.py` — runs each active hypothesis on vault data; reports per-signal and combined OOS Sharpe/drawdown/return vs equal-weight benchmark; IS→OOS decay table
- [x] DSR bug #53 fixed — `deflated_sharpe` unit mismatch resolved; DSR now correctly deflates for trial count
- [x] Regime detection: cohort weights tracked in SignalLibrary; `get_cohort_weight_summary` feeds Darwinian weight differential back into generation prompt

**Phase 4 results (49-stock universe, ~40 iterations total, ~$1.30 total cost):**
- 3 active signals: quality earnings beat combo (Sharpe 1.15, DSR 0.998, DD 43.7%), margin revenue surprise (1.12, 0.989, 43.5%), dual profitability efficiency (1.07, 0.967, 40.8%)
- Vault evaluation: all 3 signals improve OOS vs IS (decay 112–151%); combined portfolio OOS Sharpe 1.54, MaxDD 11.4%, vs benchmark 1.66 / 11.2%
- No overfitting detected: modest IS Sharpe (1.07–1.15) → modest but stable OOS (no decay)

**Known issues / open items:**
- Model frequently outputs prose before JSON despite no-commentary instruction (parse_llm_json recovers, but wastes tokens and occasionally produces truncated JSON)
- Model sometimes generates predict() bodies > 4000 chars; code_validator rejects them; model should be prompted to write concisely
- ~~`pandas_datareader` not installed → FF5 alpha unavailable~~ **resolved**: installed; `evaluation/alpha.py` now disk-caches the factor file at `data/cache/ff5_daily.parquet` (the research loop spawns one subprocess per backtest — an in-process cache alone would re-download the zip every iteration). **Caveat:** French publishes the daily FF5 file with a ~2-month lag (as of 2026-08-25 it ends 2026-06-30), so recent paper windows fall below the 30-day overlap floor and fall back to benchmark-relative alpha. Backtests (all pre-2024-05-21) are fully covered.
- 49-stock universe still skews large-cap tech; broader sector diversification would improve cross-sectional signal strength

## Phase 5 — Feature Enrichment & Signal Library Growth
Goal: fill in the NaN feature columns and run enough iterations to build a library large enough for meaningful Darwinian weighting.

**Priority 1 — Fill NaN columns (unlocks a new class of hypotheses):** ✓
- [x] `pe_ratio`, `pb_ratio`, `ps_ratio` — from FMP `/ratios`; `ev_ebitda` — from FMP `/key-metrics`; cached as `fmp_valuation.parquet`; 100% fill rate
- [x] `yield_10y`, `yield_2y`, `credit_spread` — DGS10, DGS2, BAA10Y (Moody's BAA minus 10yr Treasury) from FRED public CSV endpoint (no auth); 100% fill rate
- [x] `analyst_revision_3m` — recomputed from FMP `/analyst-estimates` (`epsAvg` QoQ change %); fixed stale cache bug (wrong column names); 71% fill rate (NaN for first quarter per ticker, expected)

**Priority 2 — Build larger signal library:**
- [ ] Run 3–5 more 20-iteration batches (`python scripts/run_loop.py --iterations 20 --budget 5.00`) to target 15–20 active signals; Darwinian weighting is meaningful only with ≥10 signals
- [ ] Fix model verbosity: add explicit "max 80 lines of Python" constraint to system prompt; reduces code validation failures
- [ ] Add `re`-retry on JSON parse failure: if `parse_llm_json` fails, send a follow-up message asking the model to output only the JSON object; reduces wasted iterations
- [x] **Marginal-alpha gate at admission** (`loop.py:_baseline_alpha_passes`): regress candidate returns on the live equal-weight book; require α with t ≥ `_min_alpha_t(n_active)` on ≥ `_MIN_BASELINE_OVERLAP_DAYS` (30) of overlap. Threshold scales with book size: `max(0.5, 2.0 − 0.01·N)` — strict on a sparse book (t≥2 at N=0), relaxes as crowdedness mechanically tightens the LOO bar, floors at 0.5 (sign filter on α). Bootstraps an empty book and short overlaps. Leave-one-out diagnostic on the current 104-signal book: 12/104 pass at the scaled t≥1.0 threshold (was 5/104 at the fixed t≥1.5), confirming the existing book is heavily over-redundant. `scripts/diagnose_baseline_alpha.py` runs the diagnostic read-only.

**Priority 2b — Admission gate rework (opened 2026-08-27):**

The stated differentiator — "CPCV + deflated Sharpe is what separates real edges from mined noise" — was not actually being enforced, and two bugs made the backtest record unusable:

- **The gate never deflated.** `subprocess_runner.py` computes the field named `dsr` as `probabilistic_sharpe(alpha_series, benchmark_sr=0.62)` — a PSR against a fixed market benchmark — and `loop.py:276` admits on `dsr > 0.65`. `n_trials` was threaded into the child process but used only as an RNG seed; it never reached a deflation formula. `deflated_sharpe()` was called from exactly one place: the Phase 3 one-off script. Re-scoring the 4 active signals with the real penalty (3,541 trials, 1,605 OOS bars → benchmark SR 1.43 annualized) gives **TrueDSR 0.000–0.069 against the 0.95 bar**. Their admission Sharpes (1.16–1.40) were already below 1.43, so a correct gate would have rejected all four at the time
- [x] **Backtest universe was non-reproducible** — `random.Random(n_trials).sample(tickers, 400)` gave every hypothesis a different 400-of-2,550 draw, so a signal's Sharpe was partly an artifact of its trial number, no two signals were compared on the same data, and no result could be reproduced. **Fixed**: seed is now the fixed constant `AUTOALPHA_UNIVERSE_SEED` (default 20240521). Universe coverage is now a deliberate act — change the seed and re-run the whole book — rather than silent per-trial drift
- [x] **Nightly research loop paused** (`run_loop_nightly.sh` step 2). Every batch of 50 hypotheses raises E[max SR] ≈ sqrt(2 ln N) for the *entire* library, so running it makes admission harder for signals already in the book while adding new ones held to no real standard. Steps 1, 3, 3b and 4 (vault update, paper, weights, report) still run
- [x] **Holdout and paper were not measured on the same universe** — `subprocess_runner` sampled 400 of ~2,700 tickers while `run_paper.py` uses the full slice. A signal whose filters require N survivors fires at a completely different rate on 400 names than on 2,700: "cheap quality revision QARP" recorded `activity_rate = 0.000` over the entire 2-year holdout (its 501 near-identical −2bp days were a cost bleed, producing a meaningless Sharpe of −80.6 and alpha t of −127) while being the *best* paper signal on the full universe. **Fixed**: `AUTOALPHA_MAX_BACKTEST_TICKERS=0` disables sampling, and `evaluate_vault.py` defaults to the full universe
- [x] **Promotion now rests on untainted evidence** (`scripts/promotion_status.py`). Backtest DSR is unusable as a gate at N=3,541, so a signal must clear benchmark-relative alpha `t ≥ 2.0` on *both* windows that were never used for selection: the vault holdout (2024-05-21 → 2026-05-21) and forward paper trading (2026-05-28 →). The two windows are disjoint, so they are independent evidence. In-sample numbers are reported for context and are explicitly not an input
- [x] **Gate pools the two untainted windows** rather than testing each separately — 501 holdout days + 62 forward days is 2.2 years of post-selection evidence, and demanding t ≥ 2 from each half independently throws away most of that power. Both windows must still be individually positive so pooling can't mask a failed one. Integrity confirmed: `prune_redundant_signals.py` reads `portfolio_alpha`, which spans 2018-01-03 → 2024-05-20 — strictly pre-vault — so no selection step ever touched the holdout
- **Status 2026-08-27** — combined book: holdout alpha +21.1%/yr (t=1.64, DD 20.2%), forward +36.9%/yr (t=1.55, DD 1.8%), **pooled +23.1%/yr, t=1.96, IR=1.32 over 562 days**. Gate needs 2.0; at the current IR that is ~16 more trading days. Per-signal pooled t: QARP 1.87, quality dip 1.79, EV/EBITDA 1.80, revision quality 1.27. IS→OOS Sharpe decay is 71–83% across all four — the edge is real and stable, it is simply not yet significant. **0/4 signals clear the gate; not ready for live trading**
- [ ] Decide the long-term admission rule. Options: enforce real DSR (admits ~nothing at current N); argue N down legitimately by counting *independent* trials — clustering the book by return correlation, since 3,541 variations on "quality × revision" are plainly not independent; or keep PSR as a cheap screen and let the holdout + forward stages carry the burden permanently

**Priority 3 — Paper trading:**
- [x] `autoalpha/execution/sizer.py` — fractional Kelly bet sizing; **Kelly fraction = 0.25** (quarter-Kelly; reduces median drawdown ~75% vs full Kelly); final position = `0.25 × kelly_bet × darwinian_weight`; hard cap: no single position > 5% of portfolio; meta_confidence multiplied in Phase 6 after meta-model is trained; closes #29; **implemented**: `kelly_bet` = continuous Kelly `mu/sigma²` estimated on the signal's trailing 63-day alpha returns, clipped to [0, 2.0] and neutral (1.0) below 20 observations — the estimator is unstable on short samples; `PositionSizer.combine()` sums overlapping names across signals, re-applies the 5% per-name cap, then scales the book to `MAX_GROSS=1.0`; `meta_confidence < 0.5` skips the trade (wired now, returns 1.0 until Phase 6)
- [ ] Paper trading mode: `Runner(strategy, Live, Sim)` — LiveProvider streams from yfinance; run nightly after market close; compare paper P&L to vault benchmark daily
- [ ] Implement `LiveProvider.bars()` streaming from yfinance for paper mode
- [ ] Run paper for ≥ 30 calendar days before advancing; gate: paper Sharpe > 0 over the period
  - **Gate tightened 2026-08-25**: Sharpe alone can be satisfied by a book that is just long the market, so `_go_no_go` in `run_paper.py` now also requires benchmark-relative alpha `t ≥ 2.0` alongside PSR > 0.65, DD < 30%, and ≥ 50 active days
  - **Status at day 61 (2026-05-28 → 2026-08-24)**: combined Sharpe 3.46, return +9.5%, DD −1.8%; alpha vs equal-weight book **+36.7%/yr, t=1.52, beta=0.07, IR=3.22** — the book is close to market-neutral (so the naive +0.5% return-difference badly understates it), but the alpha is **not yet statistically significant**. At the current IR, t=2.0 arrives around paper day ~97 (≈38 more trading days)

## Phase 6 — Meta-Labeling & Live Deployment
Goal: improve signal precision via a secondary filter, then graduate to live trading.

Meta-labeling is placed here deliberately — after paper trading — because: (a) the meta-model needs real live signal firings as positive/negative examples beyond backtest data, and (b) with a library of 15–20 signals at quarterly cadence the training set is still thin; 30+ days of paper gives additional labeled events.

- [ ] `autoalpha/labeling/meta_model.py` — per-strategy binary classifier ("given signal fired, will it work?"); trained on each CPCV fold's in-sample data only (no look-ahead); features: signal strength z-score, VIX regime, rolling 63d Sharpe, sector, market cap tier; model: LightGBM or logistic regression (not deep learning — too few samples); output: probability p ∈ [0,1]
- [ ] Wire meta-confidence into sizer: final position = `0.25 × kelly_bet × darwinian_weight × meta_confidence`; if `meta_confidence < 0.5`, skip trade entirely
- [ ] Live trading: `Runner(strategy, Live, Live)` — implement `LiveExecutor` with Alpaca API; order type: market-on-open (MOO) to match backtest fill model
  - [x] `LiveExecutor` rewritten as a broker-agnostic base holding the target-fraction → order reconciliation (sizes off live account equity, closes names absent from targets even when they have no quote, skips sub-one-share deltas, honours the overlay, `dry_run` mode); subclasses implement only `account_equity` / `current_positions` / `_place_order`
  - [x] `autoalpha/execution/alpaca.py` — `AlpacaExecutor`: MOO orders (`type=market`, `time_in_force=opg`, whole shares — Alpaca rejects fractional `opg`), retry-with-backoff on 5xx/timeouts, immediate raise on 4xx (a rejection does not become truer on retry), cancels open orders before each reconcile so stale `opg` orders can't double a position, defaults to the **paper** endpoint and warns loudly when pointed at real money
  - [x] Smoke-test `AlpacaExecutor` against a funded Alpaca paper account — **passed 2026-08-26** against account `PA3H1E1LFFIZ` (dedicated paper account, $100k, opened for autoalpha; credentials in `/home/ubuntu/.env`). Verified: paper endpoint resolution, account equity / positions / clock fetch, live quotes, dry-run reconciliation sizing correctly in whole shares off real equity (2% SPY @ 766.42 → 2 sh, 1% AAPL @ 313.48 → 3 sh, 3% MSFT @ 496.36 → 6 sh), a real `opg` order accepted then cancelled by ID (`filled_qty=0`), and a bad symbol raising `BrokerError` without retry. Account left flat with 0 open orders
  - **Do not point autoalpha at paper account `PA3M2K95NHOO`** — it holds 9 long/short positions from an unrelated automated system (last fill 2026-06-05, +$7k unrealized). Full-account reconciliation would liquidate them on the first pass
- [ ] Promotion pipeline: backtest passes → paper ≥ 30 days (Sharpe > 0) → meta-model trained → live (gated on continued paper Sharpe)
- [ ] Graceful degradation: if live broker API fails, fall back to `SimExecutor` and alert via Slack

## Phase 7 — Productionization
Goal: scheduled runs, monitoring, alerting.

- [ ] Scheduled nightly research loop (new hypotheses + re-evaluate existing library); run via cron or systemd timer after market close
- [x] Darwinian weight update job: runs daily after close, updates signal weights based on 63-day rolling alpha Sharpe; decays / kills underperforming signals — `scripts/update_weights.py`, wired into `run_loop_nightly.sh` as step 3b (after paper trading, which produces the per-signal daily returns it consumes). Residualizes against FF5 when coverage ≥ 30 days, else against the equal-weight benchmark. `SignalLibrary.sync_active()` reconciles membership with the hypothesis book (first run retired 32 stale entries left over from pruning); the `days_at_floor` counter is now idempotent per `as_of` date so a nightly retry can't age a signal toward death twice
  - **Fixed while wiring this up**: `compute_benchmark_alpha` returned raw OLS residuals, which are mean-zero by construction — every signal scored a rolling Sharpe of ~0 and pinned to the 0.3 floor. The series now adds the intercept back, matching the FF5 convention, so its mean is the alpha and its Sharpe is the IR
  - **Calibration note**: with `_TARGET_SHARPE = 0.5`, any signal above rolling Sharpe 1.25 pins to the 2.5 ceiling. All 4 active signals are currently at the ceiling, so the weighting is not yet discriminating between them and the Darwinian-weighted book is identical to the equal-weight book. Raising the target (or ranking cross-sectionally) would restore separation
- [ ] Slack notifications for: new validated signals, decaying signals, signal death, regime shifts, live trade executions, daily P&L vs benchmark
- [ ] Dashboard: signal library Darwinian weights over time, regime tracker, paper/live P&L vs vault benchmark
- [x] Vault unlock (2026-05-21): final evaluation of all strategies against the 2-year holdout; update `vault_holdout.json` lock date and run `evaluate_vault.py` — **run 2026-08-27**, 3 months after expiry. `evaluate_vault.py` now clamps to `holdout_end` from `vault_holdout.json` so the holdout window stays disjoint from the paper period (the vault parquet had grown to include it), reports benchmark-relative alpha per signal, and writes `data/vault_results.json` for the promotion gate to consume

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
