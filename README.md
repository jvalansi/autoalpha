# autoalpha

An automated alpha research system that uses LLMs to generate, backtest, and refine quantitative trading strategies — inspired by [autoresearch](https://github.com/karpathy/autoresearch) applied to financial markets.

## Concept

The system runs a continuous loop:

1. **Generate** — LLM proposes a signal hypothesis (feature engineering code + causal rationale)
2. **Build** — construct dollar bars, apply fractional differentiation, compute features
3. **Label** — triple-barrier labeling with ATR-based barriers
4. **Backtest** — walk-forward evaluation using Combinatorial Purged Cross-Validation (CPCV)
5. **Evaluate** — deflated Sharpe ratio, max drawdown, regime breakdown, correlation to library
6. **Refine** — LLM reasons on results, updates hypothesis library, proposes next iteration

Strategies are validated against a permanently held-out vault dataset the LLM never sees results from.

## Architecture

```
autoalpha/
├── data/           # Bar construction, data fetching, point-in-time alignment
├── features/       # Feature engineering (LLM-generated + curated)
├── labeling/       # Triple-barrier labeling, meta-labeling
├── backtest/       # CPCV engine, walk-forward splitter
├── evaluation/     # Deflated Sharpe, regime analysis, signal library scoring
├── research/       # LLM hypothesis loop, prompt templates, memory
├── execution/      # Paper trading gate, position sizing (Kelly)
└── strategies/     # Strategy definitions (PEAD seed + discovered)
```

## Data Sources

- **Prices/volume:** yfinance, Polygon.io
- **Earnings:** Financial Modeling Prep (FMP)
- **Macro:** FRED
- **News sentiment:** GDELT / finviz

Reuses data fetchers from [earnings-trader](https://github.com/jvalansi/earnings-trader).

## Key Design Decisions

- **Dollar bars** instead of time bars — makes returns closer to IID
- **Fractional differentiation** — stationarity without losing memory
- **CPCV** — prevents leakage in time-series cross-validation
- **Deflated Sharpe** — penalizes for number of trials tested
- **Meta-labeling** — secondary model decides whether to act on primary signal
- **Causal hypothesis requirement** — every signal must have a written mechanism; pure data mining is rejected

## Seed Strategy

The first validated hypothesis is Post-Earnings Announcement Drift (PEAD): stocks that beat EPS + revenue estimates with after-hours price confirmation tend to continue drifting for ~10 trading days. This is a well-documented anomaly with a clear causal mechanism (analyst underreaction).

## Setup

```bash
conda create -n autoalpha python=3.11
conda activate autoalpha
pip install -r requirements.txt
cp .env.example .env  # add API keys
```

## Status

Early development. See [PLAN.md](PLAN.md) for implementation roadmap.
