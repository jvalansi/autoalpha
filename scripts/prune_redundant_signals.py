"""Iterative backward elimination against the marginal-alpha admission gate.

For each active signal s, computes t-stat of alpha regressed on the equal-weight
mean of `book \\ {s}`. Drops the worst-failing signal, rebuilds the baseline,
repeats until every survivor passes the gate.

Two gate modes:
  --gate=anchor (default): gate = _min_alpha_t(N_initial). Answers "which signals
      fail the current admission criterion, iteratively?" Prune reduces collinearity
      and re-exposes redundancies that one-shot LOO misses.
  --gate=adaptive: gate = _min_alpha_t(N_current). Enforces self-consistency —
      every survivor could be admitted into the pruned book — but the gate ratchets
      up as N shrinks, so this typically collapses to a tiny survivor set.

Read-only unless --apply is passed. With --apply, dropped signals are marked
status='pruned' in the hypotheses table and 'dead' in the signal_library table.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoalpha.research.loop import _MIN_BASELINE_OVERLAP_DAYS, _min_alpha_t

DB = Path("research/memory.db")


def _loo_stat(cand: pd.Series, others: pd.DataFrame) -> tuple[int, float | None, float | None]:
    baseline = others.mean(axis=1, skipna=True).dropna()
    merged = pd.concat([cand.rename("c"), baseline.rename("b")], axis=1, sort=True).dropna()
    n = len(merged)
    if n < _MIN_BASELINE_OVERLAP_DAYS:
        return n, None, None
    y = merged["c"].to_numpy()
    x = merged["b"].to_numpy()
    x_mean = x.mean()
    Sxx = float(((x - x_mean) ** 2).sum())
    if Sxx == 0.0:
        return n, None, None
    beta = float(((x - x_mean) * (y - y.mean())).sum() / Sxx)
    alpha = float(y.mean() - beta * x_mean)
    resid = y - (alpha + beta * x)
    s2 = float((resid ** 2).sum() / (n - 2))
    var_alpha = s2 * (1.0 / n + x_mean ** 2 / Sxx)
    if var_alpha <= 0:
        return n, None, None
    se_alpha = float(np.sqrt(var_alpha))
    t = alpha / se_alpha if se_alpha > 0 else 0.0
    return n, alpha * 252, t


def _load(conn: sqlite3.Connection) -> tuple[dict[int, str], pd.DataFrame]:
    active = conn.execute(
        "SELECT id, hypothesis_json FROM hypotheses WHERE status='active' ORDER BY id"
    ).fetchall()
    names: dict[int, str] = {}
    for row in active:
        try:
            names[row["id"]] = json.loads(row["hypothesis_json"])["concise_reason"]
        except Exception:
            names[row["id"]] = f"hyp_{row['id']}"

    ids = list(names)
    if not ids:
        return names, pd.DataFrame()

    rows = conn.execute(
        f"SELECT as_of_date, alpha_return, hypothesis_id FROM portfolio_alpha "
        f"WHERE hypothesis_id IN ({','.join('?' * len(ids))})",
        ids,
    ).fetchall()
    df = pd.DataFrame(rows, columns=["date", "alpha_return", "hypothesis_id"])
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="hypothesis_id", values="alpha_return")
    return names, pivot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Mark dropped signals as 'pruned' in hypotheses and 'dead' in signal_library")
    ap.add_argument("--gate", choices=("anchor", "adaptive"), default="anchor",
                    help="anchor: gate fixed at _min_alpha_t(N_initial). "
                         "adaptive: gate recomputed at each step from N_current.")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    names, pivot = _load(conn)
    if pivot.empty:
        print("no active signals")
        return

    book = list(pivot.columns)
    n_initial = len(book)
    anchor_gate = _min_alpha_t(n_initial)
    print(f"# Iterative backward elimination — starting book: {n_initial} signals")
    if args.gate == "anchor":
        print(f"# Gate mode: anchor — fixed at _min_alpha_t({n_initial}) = {anchor_gate:.2f}")
    else:
        print(f"# Gate mode: adaptive — _min_alpha_t(N_current), starts at {anchor_gate:.2f}")
    print(f"# Overlap requirement: ≥ {_MIN_BASELINE_OVERLAP_DAYS} days")
    print()

    dropped: list[tuple[int, int, float | None, float | None, float, str]] = []
    step = 0
    while True:
        step += 1
        n = len(book)
        gate = anchor_gate if args.gate == "anchor" else _min_alpha_t(n)

        # Compute LOO stats for every signal in the current book.
        stats: dict[int, tuple[int, float | None, float | None]] = {}
        for hid in book:
            cand = pivot[hid].dropna()
            others = pivot[[c for c in book if c != hid]]
            stats[hid] = _loo_stat(cand, others)

        # Failures = evaluable signals whose t is below the gate.
        failures = [
            (hid, n_over, ann, t)
            for hid, (n_over, ann, t) in stats.items()
            if t is not None and t < gate
        ]

        if not failures:
            print(f"# Step {step}: N={n}, gate={gate:.2f} — converged, all survivors pass")
            break

        # Drop the single worst — the one whose t-stat is lowest below the gate.
        failures.sort(key=lambda r: r[3])
        worst_id, worst_n, worst_ann, worst_t = failures[0]
        reason = f"pruned N={n} t={worst_t:.2f} < {gate:.2f} α={worst_ann:.3f}/yr overlap={worst_n}d"
        print(f"step {step:>3}  N={n:>3}  gate={gate:.2f}  drop id={worst_id:>4}  "
              f"t={worst_t:>6.2f}  α/yr={worst_ann:>7.3f}  overlap={worst_n:>4}d  "
              f"({len(failures)} failing)  {names[worst_id][:60]}")
        dropped.append((worst_id, worst_n, worst_ann, worst_t, gate, names[worst_id]))
        book.remove(worst_id)

        if len(book) <= 1:
            print(f"# Step {step}: N={len(book)} — book too small to continue")
            break

    print()
    print(f"# Result: {len(book)}/{len(pivot.columns)} signals survive, {len(dropped)} dropped")
    print()

    if book:
        print("# Surviving signals — final marginal-α stats:")
        print(f"{'id':>4}  {'overlap':>7}  {'α/yr':>8}  {'t':>7}  name")
        print("-" * 90)
        final_gate = anchor_gate if args.gate == "anchor" else _min_alpha_t(len(book))
        for hid in book:
            n_over, ann, t = _loo_stat(pivot[hid].dropna(),
                                        pivot[[c for c in book if c != hid]])
            if t is None:
                print(f"{hid:>4}  {n_over:>7}  {'—':>8}  {'skip':>7}  {names[hid][:70]}")
            else:
                print(f"{hid:>4}  {n_over:>7}  {ann:>8.3f}  {t:>7.2f}  {names[hid][:70]}")
        print()
        print(f"# Final gate t ≥ {final_gate:.2f}")

    if not args.apply:
        print()
        print("# Dry run — re-run with --apply to mutate memory.db")
        return

    print()
    print(f"# Applying: marking {len(dropped)} signals as 'pruned' in hypotheses "
          f"and 'dead' in signal_library")
    for hid, n_over, ann, t, gate, name in dropped:
        justification = (
            f"Backward-elimination pruned: marginal α t={t:.2f} < gate {gate:.2f} "
            f"(α={ann:.3f}/yr, overlap={n_over}d) — redundant with surviving book."
        )
        row = conn.execute(
            "SELECT hypothesis_json FROM hypotheses WHERE id = ?", (hid,)
        ).fetchone()
        if row:
            hyp = json.loads(row["hypothesis_json"])
            hyp["justification"] = justification
            conn.execute(
                "UPDATE hypotheses SET status='pruned', hypothesis_json=? WHERE id=?",
                (json.dumps(hyp), hid),
            )
        else:
            conn.execute(
                "UPDATE hypotheses SET status='pruned' WHERE id=?", (hid,)
            )
        conn.execute(
            "UPDATE signal_library SET status='dead' WHERE name=?", (name,)
        )
    conn.commit()
    print("# Done.")


if __name__ == "__main__":
    main()
