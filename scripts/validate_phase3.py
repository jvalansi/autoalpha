"""Phase 3 CPCV validation: runs each seed strategy and reports Sharpe stats."""
from __future__ import annotations

import os
import sys
import logging
from datetime import date

# Load FMP key from ~/.env before any autoalpha imports
_env_file = os.path.expanduser("~/.env")
if os.path.exists(_env_file):
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from autoalpha.backtest.cpcv import CPCV
from autoalpha.core.providers import HistoricalProvider
from autoalpha.core.executors import SimExecutor
from autoalpha.core.runner import Runner
from autoalpha.evaluation.sharpe import annualized_sharpe, deflated_sharpe

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("validate_phase3")

# Small liquid universe — avoids needing full Sharadar constituent data
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "JNJ", "XOM", "UNH"]

# Date range: 2018-01-01 → 2024-05-20 (strictly before vault holdout)
IN_START = date(2018, 1, 1)
IN_END   = date(2024, 5, 20)

N_TRIALS = 5  # 5 strategies being evaluated — DSR correction

def get_dates(provider: HistoricalProvider, tickers: list[str]) -> pd.DatetimeIndex:
    df = provider.history(tickers, IN_START, IN_END)
    if df.empty:
        return pd.DatetimeIndex([])
    return df.index.get_level_values("date").unique().sort_values()


def run_strategy(name: str, strategy, tickers: list[str]) -> None:
    provider = HistoricalProvider(cache_dir="data/cache")
    executor = SimExecutor()

    dates = get_dates(provider, tickers)
    if len(dates) < 50:
        print(f"[{name}] Not enough data — skipping")
        return

    cpcv = CPCV(n_splits=6, n_test_splits=2, purge_days=20, embargo_days=5)
    folds = cpcv.to_runner_folds(dates)
    print(f"[{name}] {len(folds)} CPCV folds")

    runner = Runner(strategy, provider, executor, tickers)
    try:
        returns = runner.run_backtest(folds)
    except Exception as exc:
        print(f"[{name}] ERROR: {exc}")
        return

    if returns.empty:
        print(f"[{name}] No OOS returns generated")
        return

    sr = annualized_sharpe(returns)
    dsr = deflated_sharpe(returns, n_trials=N_TRIALS)
    n_nonzero = (returns != 0).sum()
    print(
        f"[{name}] OOS bars={len(returns)}, active days={n_nonzero}, "
        f"Ann.SR={sr:.3f}, DSR(n={N_TRIALS})={dsr:.3f}"
    )


def main() -> None:
    fmp_key = os.environ.get("FMP_API_KEY", "")
    if not fmp_key:
        print("WARNING: FMP_API_KEY not found — FMP-dependent strategies will produce no signals")

    print(f"Universe: {UNIVERSE}")
    print(f"Period:   {IN_START} → {IN_END}")
    print()

    # 1. Momentum — no FMP needed
    from autoalpha.strategies.momentum import MomentumStrategy
    run_strategy("Momentum", MomentumStrategy(), UNIVERSE)

    # 2. PEAD
    from autoalpha.strategies.pead import PEADStrategy
    run_strategy("PEAD", PEADStrategy(fmp_api_key=fmp_key), UNIVERSE)

    # 3. Quality
    from autoalpha.strategies.quality import QualityStrategy
    run_strategy("Quality", QualityStrategy(fmp_api_key=fmp_key), UNIVERSE)

    # 4. Earnings Revisions
    from autoalpha.strategies.earnings_revisions import EarningsRevisionsStrategy
    run_strategy("EarningsRevisions", EarningsRevisionsStrategy(fmp_api_key=fmp_key), UNIVERSE)

    # 5. Earnings NLP
    from autoalpha.strategies.earnings_nlp import EarningsNLPStrategy
    run_strategy("EarningsNLP", EarningsNLPStrategy(fmp_api_key=fmp_key), UNIVERSE)


if __name__ == "__main__":
    main()
