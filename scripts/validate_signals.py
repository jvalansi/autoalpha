"""Re-backtest all active signals on the current loop_data.parquet.

Compares new Sharpe/DSR against the stored values from the original
49-ticker universe to detect signals that overfit to the narrow universe.

Output: printed table + data/signal_validation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from autoalpha.evaluation.sharpe import deflated_sharpe, expected_max_sr
from autoalpha.research.code_validator import wrap_predict_body
from autoalpha.research.memory import HypothesisMemory
from autoalpha.research.subprocess_runner import run_strategy_subprocess

DATA_PATH = str(Path("data/loop_data.parquet").resolve())
OUT_PATH = Path("data/signal_validation.json")


def main() -> None:
    memory = HypothesisMemory()
    rows = memory._conn.execute(
        "SELECT id, sharpe, dsr, max_drawdown, hypothesis_json FROM hypotheses "
        "WHERE status='active' ORDER BY id"
    ).fetchall()
    n_trials = memory._conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
    memory.close()

    print(f"Validating {len(rows)} active signals on {DATA_PATH}")
    print(f"n_trials (for DSR): {n_trials}")
    print()
    print(f"{'ID':>4}  {'Name':<44}  {'Old Sh':>7}  {'New Sh':>7}  {'Old PSR':>7}  {'New PSR':>7}  "
          f"{'TrueDSR':>7}  {'Old DD':>7}  {'New DD':>7}  Status")
    print("-" * 140)

    results = []
    for row_id, old_sharpe, old_dsr, old_dd, hyp_json in rows:
        hyp = json.loads(hyp_json)
        name = hyp.get("concise_reason", f"signal_{row_id}")[:52]

        source = wrap_predict_body(hyp["predict_body"])
        result = run_strategy_subprocess(source, DATA_PATH, n_trials)

        if not result.succeeded:
            status = f"ERROR: {result.error[:40]}"
            new_sharpe = new_dsr = new_dd = 0.0
            true_dsr = 0.0
            n_bars = 0
        else:
            new_sharpe = result.sharpe
            new_dsr = result.dsr
            new_dd = result.max_drawdown
            # The field named `dsr` — and the loop's admission gate — is really a
            # PSR against a fixed market benchmark (SR*=0.62). It carries no
            # multiple-testing penalty. Recompute the actual Deflated Sharpe
            # against every trial ever run.
            n_bars = len(result.returns)
            if result.returns:
                rets = pd.Series(result.returns, index=pd.to_datetime(result.return_dates))
                true_dsr = float(deflated_sharpe(rets, n_trials=n_trials))
            else:
                true_dsr = 0.0
            degradation = (old_sharpe - new_sharpe) / max(abs(old_sharpe), 0.01)
            if new_dsr < 0.95:
                status = "FAIL (DSR)"
            elif degradation > 0.50:
                status = f"DEGRADED ({degradation:.0%})"
            else:
                status = "OK"

        print(f"{row_id:>4}  {name[:44]:<44}  {old_sharpe:>7.2f}  {new_sharpe:>7.2f}  "
              f"{old_dsr:>7.3f}  {new_dsr:>7.3f}  {true_dsr:>7.3f}  "
              f"{old_dd*100:>6.1f}%  {new_dd*100:>6.1f}%  {status}")

        results.append({
            "id": row_id,
            "name": name,
            "old_sharpe": old_sharpe, "new_sharpe": new_sharpe,
            "old_dsr": old_dsr, "new_dsr": new_dsr, "true_dsr": true_dsr,
            "n_bars": n_bars,
            "old_dd": old_dd, "new_dd": new_dd,
            "status": status,
        })

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT_PATH}")

    emax = expected_max_sr(n_trials)
    print(f"\nDeflation context: {n_trials} cumulative trials → E[max SR] = {emax:.2f} × SE(SR).")
    print("New PSR is the loop's actual admission gate (no trial penalty); TrueDSR applies it.")

    pass_count = sum(1 for r in results if r["status"] == "OK")
    fail_count = len(results) - pass_count
    print(f"\nSummary: {pass_count} OK, {fail_count} degraded/failed out of {len(results)}")


if __name__ == "__main__":
    main()
