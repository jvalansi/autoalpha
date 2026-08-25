#!/usr/bin/env python
"""Nightly Darwinian weight update.

Reads the paper book's per-signal daily returns, residualizes them (FF5 where
the factor data covers the window, otherwise against the equal-weight
benchmark), and updates each signal's Darwinian weight from its 63-day rolling
alpha Sharpe. Also reconciles library membership with the hypothesis book so
pruned/retired signals stop drawing weight.

Usage:
    python scripts/update_weights.py [--pnl data/paper_pnl.json] [--dry-run]

Output: data/signal_weights.json  (overwritten each run)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoalpha.evaluation.alpha import (
    compute_alpha_returns,
    compute_benchmark_alpha,
    ff5_coverage,
)
from autoalpha.evaluation.library import SignalLibrary, _WEIGHT_CEILING, _WEIGHT_FLOOR
from autoalpha.research.memory import HypothesisMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PNL_FILE = Path("data/paper_pnl.json")
OUT_FILE = Path("data/signal_weights.json")
_MIN_FF5_OVERLAP = 30


def _series(mapping: dict) -> pd.Series:
    s = pd.Series(mapping, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def alpha_series_for(returns: pd.Series, benchmark: pd.Series | None) -> tuple[pd.Series, str]:
    """Residualize a signal's returns. Returns (alpha_series, method_used).

    FF5 is preferred but publishes ~2 months late, so recent paper windows fall
    back to the equal-weight benchmark — which is the book's actual investable
    alternative anyway.
    """
    covered, _ = ff5_coverage(returns.index)
    if covered >= _MIN_FF5_OVERLAP:
        alpha, _ = compute_alpha_returns(returns)
        return alpha, "ff5"

    if benchmark is not None and not benchmark.empty:
        stats = compute_benchmark_alpha(returns, benchmark)
        if stats.get("available"):
            return stats["residuals"], "benchmark"

    return returns, "raw"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnl", default=str(PNL_FILE))
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the weights that would be written, change nothing")
    args = parser.parse_args()

    pnl_path = Path(args.pnl)
    if not pnl_path.exists():
        log.error("%s not found — run scripts/run_paper.py first", pnl_path)
        sys.exit(1)

    pnl = json.loads(pnl_path.read_text())
    as_of = pd.Timestamp(pnl["paper_end"]).date()

    bm_raw = pnl.get("benchmark", {}).get("daily_returns")
    benchmark = _series(bm_raw) if bm_raw else None
    if benchmark is None:
        log.warning("No benchmark daily returns in %s — FF5 or raw returns only", pnl_path)

    # --- membership sync -------------------------------------------------
    memory = HypothesisMemory()
    active_names = [
        json.loads(r[0]).get("concise_reason", f"signal_{r[1]}")
        for r in memory._conn.execute(
            "SELECT hypothesis_json, id FROM hypotheses WHERE status='active' ORDER BY id"
        ).fetchall()
    ]
    memory.close()

    library = SignalLibrary()
    if args.dry_run:
        log.info("DRY RUN — no database writes")
    else:
        sync = library.sync_active(active_names)
        log.info("Library sync: %d added, %d retired", len(sync["added"]), len(sync["retired"]))

    # --- weight update ---------------------------------------------------
    alpha_returns: dict[str, pd.Series] = {}
    methods: dict[str, str] = {}
    for sig in pnl.get("signals", []):
        rets = _series(sig["daily_returns"])
        if rets.empty:
            continue
        alpha, method = alpha_series_for(rets, benchmark)
        alpha_returns[sig["name"]] = alpha
        methods[sig["name"]] = method

    if not alpha_returns:
        log.warning("No signal returns in %s — nothing to update", pnl_path)
        return

    before = library.all_weights()
    if not args.dry_run:
        library.update_weights(alpha_returns, as_of=as_of)
    after = library.all_weights()

    rows = library._conn.execute(
        "SELECT name, weight, status, days_at_floor FROM signal_library "
        "WHERE name IN (%s) ORDER BY weight DESC" % ",".join("?" * len(alpha_returns)),
        list(alpha_returns),
    ).fetchall()

    print(f"\nDarwinian weights as of {as_of}  (floor {_WEIGHT_FLOOR}, ceiling {_WEIGHT_CEILING})")
    print(f"{'Signal':<45} {'Weight':>7} {'Prev':>7} {'Status':>9} {'Floor days':>11} {'Alpha':>10}")
    print("-" * 95)
    summary = []
    for name, weight, status, floor_days in rows:
        prev = before.get(name, 1.0)
        print(f"{name[:45]:<45} {weight:>7.2f} {prev:>7.2f} {status:>9} {floor_days:>11} "
              f"{methods.get(name, '-'):>10}")
        summary.append({
            "name": name,
            "weight": round(float(weight), 4),
            "previous_weight": round(float(prev), 4),
            "status": status,
            "days_at_floor": int(floor_days),
            "alpha_method": methods.get(name, "-"),
        })

    if not args.dry_run:
        OUT_FILE.write_text(json.dumps(
            {"as_of": str(as_of), "signals": summary}, indent=2
        ))
        log.info("Saved %s", OUT_FILE)
    library.close()


if __name__ == "__main__":
    main()
