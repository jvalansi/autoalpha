"""Subprocess execution of LLM-generated strategy code.

Writes the wrapped strategy source to a temp file, spawns a child Python
process that runs a backtest, and returns a BacktestResult.

The child process writes a JSON result to stdout; this module reads it.
Execution is capped at TIMEOUT_SECONDS to prevent runaway code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

TIMEOUT_SECONDS: int = 600

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    sharpe: float = 0.0
    dsr: float = 0.0
    max_drawdown: float = 0.0
    activity_rate: float = 0.0
    returns: list[float] = field(default_factory=list)
    return_dates: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "sharpe": self.sharpe,
            "dsr": self.dsr,
            "max_drawdown": self.max_drawdown,
            "activity_rate": self.activity_rate,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_strategy_subprocess(
    strategy_source: str,
    data_path: str,
    n_trials_so_far: int,
) -> BacktestResult:
    """Write strategy_source to a temp file, run it in a subprocess, return result.

    Args:
        strategy_source: Complete Python module produced by wrap_predict_body().
        data_path: Path to the Parquet file the child process will load.
        n_trials_so_far: Passed to the child for DSR calculation.

    Returns:
        BacktestResult with metrics or an error message.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="autoalpha_gen_",
        delete=False,
    ) as f:
        f.write(strategy_source)
        tmp_path = f.name

    try:
        result = _run_child(tmp_path, data_path, n_trials_so_far)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return result


# ---------------------------------------------------------------------------
# Child process harness (also callable as __main__)
# ---------------------------------------------------------------------------

_CHILD_HARNESS = """\
import json
import sys
import traceback

import pandas as pd

def _main():
    strategy_path = sys.argv[1]
    data_path = sys.argv[2]
    n_trials = int(sys.argv[3])

    # Load the generated strategy module
    import importlib.util
    spec = importlib.util.spec_from_file_location("_generated", strategy_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    strategy = mod.strategy

    import pyarrow.dataset as _ds
    import pyarrow as _pa
    import random

    from autoalpha.core.providers import HistoricalProvider
    from autoalpha.core.executors import SimExecutor
    from autoalpha.core.runner import Runner
    from autoalpha.backtest.cpcv import CPCV
    from autoalpha.evaluation.sharpe import probabilistic_sharpe

    # Read ticker list cheaply using pyarrow (avoids loading all row data)
    _dataset = _ds.dataset(data_path, format="parquet")
    _all_tickers = (
        _dataset.to_table(columns=["ticker"])
        .column("ticker")
        .to_pylist()
    )
    _all_tickers = sorted(set(_all_tickers))

    # Sample tickers if the universe is large. The seed is FIXED, not derived
    # from n_trials: seeding by trial count gave every hypothesis a different
    # 400-of-2550 draw, so a signal's Sharpe was partly an artifact of which
    # universe its trial number happened to pull, two signals were never
    # compared on the same data, and no backtest could be reproduced later.
    # Coverage across the full universe is a separate concern — vary
    # AUTOALPHA_UNIVERSE_SEED deliberately and re-run the whole book, rather
    # than letting it drift silently between trials.
    # The cap is a speed compromise for the research loop. Evaluation paths that
    # must match paper trading (which uses the full universe) set
    # AUTOALPHA_MAX_BACKTEST_TICKERS=0 to disable sampling entirely — a signal
    # whose filters need N survivors fires at a completely different rate on 400
    # names than on 2,700, so a subsampled holdout is not comparable to paper.
    import os as _os
    MAX_BACKTEST_TICKERS = int(_os.environ.get("AUTOALPHA_MAX_BACKTEST_TICKERS", "400"))
    UNIVERSE_SEED = int(_os.environ.get("AUTOALPHA_UNIVERSE_SEED", "20240521"))
    if MAX_BACKTEST_TICKERS > 0 and len(_all_tickers) > MAX_BACKTEST_TICKERS:
        rng = random.Random(UNIVERSE_SEED)
        _all_tickers = sorted(rng.sample(_all_tickers, MAX_BACKTEST_TICKERS))

    # Load only the sampled tickers via filter pushdown — row groups are
    # organized per ticker so this reads ~400/2550 of the file.
    _table = _dataset.to_table(
        filter=_ds.field("ticker").isin(_all_tickers)
    )
    # Restore pandas MultiIndex from pyarrow table
    mi = _table.to_pandas()
    if not isinstance(mi.index, pd.MultiIndex):
        idx_cols = [c for c in ["date", "ticker"] if c in mi.columns]
        if idx_cols:
            mi = mi.set_index(idx_cols)
    mi.index.names = ["date", "ticker"]
    mi.index = pd.MultiIndex.from_arrays([
        pd.to_datetime(mi.index.get_level_values("date")),
        mi.index.get_level_values("ticker"),
    ], names=["date", "ticker"])
    mi = mi.sort_index(level="date", sort_remaining=False)

    tickers = _all_tickers
    dates = mi.index.get_level_values("date").unique().sort_values()

    provider = HistoricalProvider.__new__(HistoricalProvider)
    provider._data = mi

    import unittest.mock as mock
    # Pre-build a sorted list of (date, bar_df) tuples once to avoid repeated groupby.
    _date_vals = mi.index.get_level_values("date")
    _bar_lookup = {
        bar_date: grp.droplevel("date")
        for bar_date, grp in mi.groupby(level="date")
    }
    _sorted_dates = sorted(_bar_lookup.keys())

    def _history(tkrs, s, e):
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        mask = (_date_vals >= s_ts) & (_date_vals <= e_ts)
        return mi[mask]

    def _bars(tkrs, s, e):
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        for d in _sorted_dates:
            if s_ts <= d <= e_ts:
                yield d, _bar_lookup[d]

    with mock.patch.object(provider, "history", side_effect=_history), \\
         mock.patch.object(provider, "bars", side_effect=_bars):

        executor = SimExecutor(initial_capital=100_000, cost_bps=11)
        runner = Runner(strategy, provider, executor, tickers)

        cpcv = CPCV(n_splits=4, n_test_splits=2)
        folds = cpcv.to_runner_folds(dates)

        returns = runner.run_backtest(folds)

    if returns.empty:
        print(json.dumps({"sharpe": 0.0, "dsr": 0.0, "max_drawdown": 0.0, "activity_rate": 0.0, "error": "empty returns"}))
        return

    # CPCV generates overlapping OOS windows: each date can appear in multiple folds.
    # Average across folds so each calendar date contributes exactly once.
    returns = returns.groupby(level=0).mean()

    # Active days: number of OOS bars where strategy held positions (non-zero return)
    active_days = int((returns != 0).sum())
    activity_rate = float((returns != 0).mean())

    import numpy as np
    from autoalpha.evaluation.alpha import compute_alpha_returns
    # Use FF5 alpha returns for Sharpe/DSR; fall back to raw returns if unavailable.
    alpha_series, _ = compute_alpha_returns(returns)

    daily = alpha_series.values
    sharpe = float(daily.mean() / daily.std() * (252 ** 0.5)) if daily.std() > 0 else 0.0
    # PSR vs market benchmark (SPY annualized Sharpe ~0.62): P(true SR > market SR)
    dsr = float(probabilistic_sharpe(alpha_series, benchmark_sr=0.62))
    # Anchor NAV at 1.0 so losses on the very first bar are captured.
    nav = pd.concat([pd.Series([1.0]), (1 + alpha_series).cumprod()])
    roll_max = nav.cummax()
    dd = (nav - roll_max) / roll_max
    max_drawdown = float(abs(dd.min()))

    print(json.dumps({
        "sharpe": sharpe,
        "dsr": dsr,
        "max_drawdown": max_drawdown,
        "activity_rate": activity_rate,
        "active_days": active_days,
        "returns": daily.tolist(),
        "return_dates": alpha_series.index.strftime("%Y-%m-%d").tolist(),
        "error": None,
    }))

try:
    _main()
except Exception as exc:
    import json, traceback
    print(json.dumps({"sharpe": 0.0, "dsr": 0.0, "max_drawdown": 0.0, "error": traceback.format_exc()}))
"""


def _run_child(
    strategy_path: str,
    data_path: str,
    n_trials: int,
) -> BacktestResult:
    """Write the harness to a temp file and spawn a child process."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="autoalpha_harness_",
        delete=False,
    ) as f:
        f.write(_CHILD_HARNESS)
        harness_path = f.name

    # Ensure the project root is on the child's PYTHONPATH so autoalpha is importable.
    _project_root = str(Path(__file__).resolve().parents[2])
    child_env = {**os.environ}
    existing_pp = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = f"{_project_root}:{existing_pp}" if existing_pp else _project_root

    try:
        proc = subprocess.run(
            [sys.executable, harness_path, strategy_path, data_path, str(n_trials)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        return BacktestResult(error=f"Subprocess timed out after {TIMEOUT_SECONDS}s")
    except Exception as exc:
        return BacktestResult(error=f"Subprocess launch failed: {exc}")
    finally:
        try:
            os.unlink(harness_path)
        except OSError:
            pass

    stdout = proc.stdout.strip()
    if not stdout:
        stderr_snippet = proc.stderr[-500:] if proc.stderr else "(no stderr)"
        return BacktestResult(error=f"No output from child process. stderr: {stderr_snippet}")

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return BacktestResult(error=f"Child output not JSON: {exc}. stdout: {stdout[:200]}")

    if data.get("error"):
        return BacktestResult(
            sharpe=data.get("sharpe", 0.0),
            dsr=data.get("dsr", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
            activity_rate=data.get("activity_rate", 0.0),
            error=data["error"],
        )

    return BacktestResult(
        sharpe=data.get("sharpe", 0.0),
        dsr=data.get("dsr", 0.0),
        max_drawdown=data.get("max_drawdown", 0.0),
        activity_rate=data.get("activity_rate", 0.0),
        returns=data.get("returns", []),
        return_dates=data.get("return_dates", []),
    )
