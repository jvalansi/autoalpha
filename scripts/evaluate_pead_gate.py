"""Run earnings-trader's PEAD rules through the promotion gate.

Same evidence as scripts/promotion_status.py: holdout (vault_holdout.json window)
and forward (paper_start → latest vault bar), each run with SimExecutor at 11 bps
against the equal-weight benchmark, pooled into one alpha regression and judged
by promotion_status.gate_verdict. Does not add PEAD to the signal library.

Usage:
    python scripts/evaluate_pead_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoalpha.core.executors import SimExecutor
from autoalpha.evaluation.alpha import compute_benchmark_alpha
from autoalpha.strategies.pead_et import EarningsTraderPEAD
from promotion_status import MIN_ALPHA_T, gate_verdict

VAULT = Path("data/vault_data.parquet")


def load_window(start: pd.Timestamp, end: pd.Timestamp) -> dict[pd.Timestamp, pd.DataFrame]:
    t = ds.dataset(VAULT).to_table(
        filter=(ds.field("date") >= start.to_pydatetime()) & (ds.field("date") <= end.to_pydatetime()))
    df = t.to_pandas().reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["date", "ticker"], keep="last")
    return {d: g.set_index("ticker") for d, g in df.groupby("date")}


def run(bars: dict[pd.Timestamp, pd.DataFrame]) -> tuple[pd.Series, pd.Series, int]:
    strategy, executor = EarningsTraderPEAD(), SimExecutor(initial_capital=100_000, cost_bps=11)
    prev: dict[str, float] = {}
    entries = 0
    for d in sorted(bars):
        b = bars[d]
        executor.execute(prev, d.date(), b["Open"].dropna().to_dict())
        targets = strategy.predict(b, bar_date=d)
        entries += len(set(targets) - set(prev))
        prev = targets
    closes = pd.DataFrame({d: b["Close"] for d, b in bars.items()}).T.sort_index()
    benchmark = closes.pct_change(fill_method=None).iloc[1:].mean(axis=1)
    return executor.returns(), benchmark, entries


def summarize(rets: pd.Series, bm: pd.Series, entries: int) -> dict:
    st = compute_benchmark_alpha(rets, bm)
    nav = pd.concat([pd.Series([1.0]), (1 + rets).cumprod()])
    return {
        "alpha_annualized": round(st["alpha_annualized"] * 100, 2) if st.get("available") else None,
        "alpha_t": round(st["alpha_t"], 2) if st.get("available") else None,
        "beta": round(st["beta"], 2) if st.get("available") else None,
        "max_drawdown": round(float(((nav - nav.cummax()) / nav.cummax()).min()) * 100, 2),
        "total_return": round(float((1 + rets).prod() - 1) * 100, 2),
        "n_bars": len(rets), "entries": entries,
    }


def main() -> None:
    lock = json.loads(Path("vault_holdout.json").read_text())
    paper_start = pd.Timestamp(json.loads(Path("data/paper_state.json").read_text())["paper_start"])
    windows = {"holdout": (pd.Timestamp(lock["holdout_start"]), pd.Timestamp(lock["holdout_end"])),
               "forward": (paper_start, pd.Timestamp.today().normalize())}

    results, series = {}, {}
    for name, (s, e) in windows.items():
        rets, bm, entries = run(load_window(s, e))
        results[name] = summarize(rets, bm, entries)
        series[name] = (rets, bm)
        print(f"{name:8} {s.date()} → {e.date()}: {results[name]}")

    pooled = compute_benchmark_alpha(pd.concat([series["holdout"][0], series["forward"][0]]),
                                     pd.concat([series["holdout"][1], series["forward"][1]]))
    print(f"pooled   alpha={pooled['alpha_annualized']*100:.1f}%/yr  t={pooled['alpha_t']:.2f}  "
          f"IR={pooled['information_ratio']:.2f}  n={pooled['n_overlap']}  (gate: t ≥ {MIN_ALPHA_T})")
    passes, reasons = gate_verdict(results["holdout"], results["forward"], pooled)
    print("PASS" if passes else f"HOLD — {', '.join(reasons)}")


if __name__ == "__main__":
    main()
