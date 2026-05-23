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

TIMEOUT_SECONDS: int = 60

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    sharpe: float = 0.0
    dsr: float = 0.0
    max_drawdown: float = 0.0
    returns: list[float] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "sharpe": self.sharpe,
            "dsr": self.dsr,
            "max_drawdown": self.max_drawdown,
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

    # Load market data
    mi = pd.read_parquet(data_path)
    dates = mi.index.get_level_values("date").unique().sort_values()

    from autoalpha.core.providers import HistoricalProvider
    from autoalpha.core.executors import SimExecutor
    from autoalpha.core.runner import Runner
    from autoalpha.backtest.cpcv import CPCV
    from autoalpha.evaluation.sharpe import deflated_sharpe

    tickers = mi.index.get_level_values("ticker").unique().tolist()

    provider = HistoricalProvider.__new__(HistoricalProvider)
    provider._data = mi

    import unittest.mock as mock
    def _history(tkrs, s, e):
        mask = (
            (mi.index.get_level_values("date") >= pd.Timestamp(s)) &
            (mi.index.get_level_values("date") <= pd.Timestamp(e))
        )
        return mi[mask]

    def _bars(tkrs, s, e):
        sub = _history(tkrs, s, e)
        for bar_date, grp in sub.groupby(level="date"):
            yield bar_date, grp.droplevel("date")

    with mock.patch.object(provider, "history", side_effect=_history), \\
         mock.patch.object(provider, "bars", side_effect=_bars):

        executor = SimExecutor(initial_capital=100_000, cost_bps=11)
        runner = Runner(strategy, provider, executor, tickers)

        cpcv = CPCV(n_splits=6, n_test_splits=2)
        folds = cpcv.to_runner_folds(dates)

        returns = runner.run_backtest(folds)

    if returns.empty:
        print(json.dumps({"sharpe": 0.0, "dsr": 0.0, "max_drawdown": 0.0, "error": "empty returns"}))
        return

    import numpy as np
    daily = returns.values
    sharpe = float(daily.mean() / daily.std() * (252 ** 0.5)) if daily.std() > 0 else 0.0
    dsr = float(deflated_sharpe(returns, n_trials=n_trials))
    nav = (1 + returns).cumprod()
    roll_max = nav.cummax()
    dd = (nav - roll_max) / roll_max
    max_drawdown = float(abs(dd.min()))

    print(json.dumps({
        "sharpe": sharpe,
        "dsr": dsr,
        "max_drawdown": max_drawdown,
        "returns": daily.tolist(),
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

    try:
        proc = subprocess.run(
            [sys.executable, harness_path, strategy_path, data_path, str(n_trials)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env={**os.environ},
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
            error=data["error"],
        )

    return BacktestResult(
        sharpe=data.get("sharpe", 0.0),
        dsr=data.get("dsr", 0.0),
        max_drawdown=data.get("max_drawdown", 0.0),
        returns=data.get("returns", []),
    )
