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
    print(f"{'ID':>4}  {'Name':<52}  {'Old Sh':>7}  {'New Sh':>7}  {'Old DSR':>7}  {'New DSR':>7}  {'Old DD':>7}  {'New DD':>7}  Status")
    print("-" * 130)

    results = []
    for row_id, old_sharpe, old_dsr, old_dd, hyp_json in rows:
        hyp = json.loads(hyp_json)
        name = hyp.get("concise_reason", f"signal_{row_id}")[:52]

        source = wrap_predict_body(hyp["predict_body"])
        result = run_strategy_subprocess(source, DATA_PATH, n_trials)

        if not result.succeeded:
            status = f"ERROR: {result.error[:40]}"
            new_sharpe = new_dsr = new_dd = 0.0
        else:
            new_sharpe = result.sharpe
            new_dsr = result.dsr
            new_dd = result.max_drawdown
            degradation = (old_sharpe - new_sharpe) / max(abs(old_sharpe), 0.01)
            if new_dsr < 0.95:
                status = "FAIL (DSR)"
            elif degradation > 0.50:
                status = f"DEGRADED ({degradation:.0%})"
            else:
                status = "OK"

        print(f"{row_id:>4}  {name:<52}  {old_sharpe:>7.2f}  {new_sharpe:>7.2f}  "
              f"{old_dsr:>7.3f}  {new_dsr:>7.3f}  {old_dd*100:>6.1f}%  {new_dd*100:>6.1f}%  {status}")

        results.append({
            "id": row_id,
            "name": name,
            "old_sharpe": old_sharpe, "new_sharpe": new_sharpe,
            "old_dsr": old_dsr, "new_dsr": new_dsr,
            "old_dd": old_dd, "new_dd": new_dd,
            "status": status,
        })

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT_PATH}")

    pass_count = sum(1 for r in results if r["status"] == "OK")
    fail_count = len(results) - pass_count
    print(f"\nSummary: {pass_count} OK, {fail_count} degraded/failed out of {len(results)}")


if __name__ == "__main__":
    main()
