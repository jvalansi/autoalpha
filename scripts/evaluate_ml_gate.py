"""Run the cross-sectional ML ranker (autoalpha/strategies/ml_cross_section.py)
through the promotion gate.

1. Internal check on loop data only: fit 2018-01 → 2021-12, trade 2022-01 → loop end.
2. Gate: fit on all loop data (2018-01 → 2024-05-20, pre-vault), then trade the
   holdout and forward windows exactly as scripts/evaluate_pead_gate.py does —
   SimExecutor at 11 bps, pooled alpha regression,
   promotion_status.gate_verdict — except the equal-weight benchmark is open-to-open
   (see run()). Hyperparameters are fixed in the strategy module
   and were not tuned on either window.

Usage:
    python scripts/evaluate_ml_gate.py
"""
from __future__ import annotations

import gc
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoalpha.core.executors import SimExecutor
from autoalpha.evaluation.alpha import compute_benchmark_alpha
from autoalpha.strategies.ml_cross_section import FEATURES, MLCrossSection
from evaluate_pead_gate import load_window, summarize
from promotion_status import MIN_ALPHA_T, gate_verdict

LOOP = Path("data/loop_data.parquet")
INTERNAL_SPLIT = pd.Timestamp("2022-01-01")


def run(strategy: MLCrossSection, bars: Iterable[tuple[pd.Timestamp, pd.DataFrame]]) -> tuple[pd.Series, pd.Series, int]:
    """bars: (date, per-ticker frame) in date order — a generator is fine."""
    executor = SimExecutor(initial_capital=100_000, cost_bps=11, hold_unchanged=True)
    prev: dict[str, float] = {}
    entries = 0
    opens = {}
    for d, b in bars:
        executor.execute(prev, d.date(), b["Open"].dropna().to_dict())
        targets = strategy.predict(b, bar_date=d)
        entries += len(set(targets) - set(prev))
        prev = targets
        opens[d] = b["Open"].where(b["Open"] > 0)  # loop data has zero prices
    # Open-to-open, matching SimExecutor, which marks NAV at each bar's open. A
    # close-to-close benchmark on the same date label overlaps only overnight, so
    # beta collapses toward 0 and the market's return shows up as alpha.
    benchmark = pd.DataFrame(opens).T.sort_index().pct_change(fill_method=None).iloc[1:].mean(axis=1)
    return executor.returns(), benchmark, entries


def load_loop() -> pd.DataFrame:
    """Only the columns the strategy uses, as float32 — the full frame doesn't fit in RAM."""
    df = pd.read_parquet(LOOP, columns=["Open", "Close"] + FEATURES).astype("float32")
    return df[~df.index.duplicated(keep="last")].sort_index()


def main() -> None:
    loop = load_loop()
    dates = loop.index.get_level_values("date")
    s = MLCrossSection()
    s.fit(loop[dates < INTERNAL_SPLIT])
    test = loop[dates >= INTERNAL_SPLIT]
    days = ((d, g.droplevel("date")) for d, g in test.groupby(level="date"))
    rets, bm, entries = run(s, days)
    st = compute_benchmark_alpha(rets, bm)
    print(f"internal {INTERNAL_SPLIT.date()} → {rets.index.max()}: {summarize(rets, bm, entries)} "
          f"IR={st['information_ratio']:.2f}", flush=True)
    del test

    s = MLCrossSection()
    s.fit(loop)
    del loop, dates
    gc.collect()
    lock = json.loads(Path("vault_holdout.json").read_text())
    paper_start = pd.Timestamp(json.loads(Path("data/paper_state.json").read_text())["paper_start"])
    windows = {"holdout": (pd.Timestamp(lock["holdout_start"]), pd.Timestamp(lock["holdout_end"])),
               "forward": (paper_start, pd.Timestamp.today().normalize())}

    results, series = {}, {}
    for name, (start, end) in windows.items():
        rets, bm, entries = run(s, sorted(load_window(start, end).items()))
        results[name] = summarize(rets, bm, entries)
        series[name] = (rets, bm)
        print(f"{name:8} {start.date()} → {end.date()}: {results[name]}", flush=True)

    pooled = compute_benchmark_alpha(pd.concat([series["holdout"][0], series["forward"][0]]),
                                     pd.concat([series["holdout"][1], series["forward"][1]]))
    print(f"pooled   alpha={pooled['alpha_annualized']*100:.1f}%/yr  t={pooled['alpha_t']:.2f}  "
          f"IR={pooled['information_ratio']:.2f}  n={pooled['n_overlap']}  (gate: t ≥ {MIN_ALPHA_T})")
    passes, reasons = gate_verdict(results["holdout"], results["forward"], pooled)
    print("PASS" if passes else f"HOLD — {', '.join(reasons)}")


if __name__ == "__main__":
    main()
