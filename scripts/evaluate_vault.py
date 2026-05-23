"""Evaluate accepted signals on the vault holdout period.

For each hypothesis with status='active' in the DB, re-runs its
predict() on vault_data.parquet and reports per-signal and combined
portfolio statistics vs. an equal-weight benchmark.

Usage:
    python scripts/evaluate_vault.py [--db PATH] [--data PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoalpha.research.code_validator import wrap_predict_body
from autoalpha.research.subprocess_runner import run_strategy_subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _load_active_hypotheses(db_path: Path) -> list[dict]:
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, trial_number, hypothesis_json, sharpe, dsr, max_drawdown "
        "FROM hypotheses WHERE status = 'active' ORDER BY trial_number"
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["hyp"] = json.loads(d["hypothesis_json"])
        result.append(d)
    return result


def _benchmark_returns(vault_data: pd.DataFrame) -> pd.Series:
    """Equal-weight daily returns across all tickers."""
    prices = (
        vault_data["Close"]
        .groupby(level=["date", "ticker"]).last()  # deduplicate any duplicate index entries
        .unstack(level="ticker")
        .sort_index()
    )
    rets = prices.pct_change().dropna(how="all")
    return rets.mean(axis=1).rename("benchmark")


def _print_stats(label: str, returns: pd.Series) -> None:
    if returns.empty or returns.std() == 0:
        log.info("  %-40s  (no returns)", label)
        return
    sharpe = returns.mean() / returns.std() * (252 ** 0.5)
    nav = pd.concat([pd.Series([1.0]), (1 + returns).cumprod()])
    dd = ((nav - nav.cummax()) / nav.cummax()).min()
    total_ret = (1 + returns).prod() - 1
    log.info(
        "  %-40s  Sharpe=%+.2f  MaxDD=%.1f%%  TotalRet=%+.1f%%  n=%d",
        label, sharpe, abs(dd) * 100, total_ret * 100, len(returns),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Vault holdout evaluation")
    parser.add_argument("--db", default=None, help="Path to memory.db")
    parser.add_argument("--data", default="data/vault_data.parquet", help="Vault parquet")
    args = parser.parse_args()

    vault_path = Path(args.data).resolve()
    if not vault_path.exists():
        print(f"ERROR: vault data not found: {vault_path}")
        print("Run scripts/build_vault_dataset.py first.")
        sys.exit(1)

    db_path = Path(args.db).resolve() if args.db else Path("research/memory.db").resolve()
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}")
        sys.exit(1)

    hypotheses = _load_active_hypotheses(db_path)
    if not hypotheses:
        print("No active hypotheses found in DB.")
        sys.exit(0)

    vault_data = pd.read_parquet(vault_path)
    date_range = (
        vault_data.index.get_level_values("date").min().date(),
        vault_data.index.get_level_values("date").max().date(),
    )
    log.info("Vault period: %s → %s  (%d tickers, %d rows)",
             *date_range,
             vault_data.index.get_level_values("ticker").nunique(),
             len(vault_data))
    log.info("Active signals to evaluate: %d", len(hypotheses))
    print()

    # Benchmark
    benchmark = _benchmark_returns(vault_data)

    print("=" * 75)
    print("PER-SIGNAL VAULT PERFORMANCE  (IS = in-sample stats from training)")
    print("=" * 75)
    print(f"  {'Signal':<40}  {'IS Sharpe':>9}  {'IS DSR':>7}  {'IS DD':>7}")
    print("-" * 75)
    for h in hypotheses:
        print(
            f"  {h['hyp']['concise_reason']:<40}  "
            f"{h['sharpe']:>9.2f}  {h['dsr']:>7.3f}  {h['max_drawdown']*100:>6.1f}%"
        )
    print()

    # Run each signal on vault
    signal_returns: dict[str, pd.Series] = {}
    for h in hypotheses:
        name = h["hyp"]["concise_reason"]
        log.info("Running vault backtest: %s (trial %d)", name, h["trial_number"])
        source = wrap_predict_body(h["hyp"]["predict_body"])
        result = run_strategy_subprocess(source, str(vault_path), n_trials_so_far=len(hypotheses))
        if not result.succeeded:
            log.warning("  FAILED: %s", result.error[:120])
            continue
        if not result.returns:
            log.warning("  Empty returns for %s", name)
            continue
        idx = pd.to_datetime(result.return_dates)
        rets = pd.Series(result.returns, index=idx)
        signal_returns[name] = rets
        log.info("  done: %d bars", len(rets))

    if not signal_returns:
        print("All signals failed on vault data.")
        sys.exit(1)

    # Combine into equal-weight portfolio
    combined_df = pd.DataFrame(signal_returns).dropna(how="all")
    portfolio = combined_df.mean(axis=1).rename("portfolio")

    print()
    print("=" * 75)
    print("VAULT HOLDOUT RESULTS")
    print("=" * 75)
    _print_stats("Equal-weight benchmark (all tickers)", benchmark)
    for name, rets in signal_returns.items():
        _print_stats(name, rets)
    print("-" * 75)
    _print_stats("COMBINED PORTFOLIO (equal-weight signals)", portfolio)
    print("=" * 75)

    # IS → OOS Sharpe comparison
    print()
    print("IN-SAMPLE vs VAULT SHARPE")
    print("-" * 55)
    for h in hypotheses:
        name = h["hyp"]["concise_reason"]
        if name not in signal_returns:
            continue
        oos_rets = signal_returns[name]
        oos_sharpe = (
            oos_rets.mean() / oos_rets.std() * (252 ** 0.5)
            if oos_rets.std() > 0 else 0.0
        )
        decay = oos_sharpe / h["sharpe"] if h["sharpe"] != 0 else float("nan")
        print(
            f"  {name:<40}  IS={h['sharpe']:+.2f}  OOS={oos_sharpe:+.2f}  "
            f"decay={decay:.0%}"
        )
    print()


if __name__ == "__main__":
    main()
