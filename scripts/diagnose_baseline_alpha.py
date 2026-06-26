"""Leave-one-out diagnostic: regress each active signal's alpha on the equal-weight
mean of the others; report α/yr, t-stat, overlap, pass/fail at the gate threshold.

Read-only. Does not mutate memory.db.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from autoalpha.research.loop import _min_alpha_t, _MIN_BASELINE_OVERLAP_DAYS

DB = Path("research/memory.db")


def main() -> None:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

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
        print("no active signals")
        return

    rows = conn.execute(
        f"SELECT as_of_date, alpha_return, hypothesis_id FROM portfolio_alpha "
        f"WHERE hypothesis_id IN ({','.join('?' * len(ids))})",
        ids,
    ).fetchall()
    df = pd.DataFrame(rows, columns=["date", "alpha_return", "hypothesis_id"])
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="hypothesis_id", values="alpha_return")

    n_active = len(ids)
    t_min = _min_alpha_t(n_active)
    print(f"# Leave-one-out marginal-alpha diagnostic ({n_active} active signals)")
    print(f"# Gate: t ≥ {t_min:.2f} (scaled by N={n_active}), overlap ≥ {_MIN_BASELINE_OVERLAP_DAYS} days")
    print()
    print(f"{'id':>4}  {'overlap':>7}  {'α/yr':>8}  {'t':>7}  {'pass':>5}  name")
    print("-" * 80)

    results = []
    for hid in ids:
        cand = pivot[hid].dropna()
        others = pivot.drop(columns=[hid])
        baseline = others.mean(axis=1, skipna=True).dropna()
        merged = pd.concat([cand.rename("c"), baseline.rename("b")], axis=1, sort=True).dropna()
        n = len(merged)
        if n < _MIN_BASELINE_OVERLAP_DAYS:
            print(f"{hid:>4}  {n:>7}  {'—':>8}  {'—':>7}  {'skip':>5}  {names[hid]}")
            results.append({"id": hid, "n": n, "alpha_yr": None, "t": None, "passes": True, "skipped": True})
            continue
        y = merged["c"].to_numpy()
        x = merged["b"].to_numpy()
        x_mean = x.mean()
        Sxx = float(((x - x_mean) ** 2).sum())
        if Sxx == 0.0:
            print(f"{hid:>4}  {n:>7}  {'zero-var':>8}  {'—':>7}  {'skip':>5}  {names[hid]}")
            continue
        beta = float(((x - x_mean) * (y - y.mean())).sum() / Sxx)
        alpha = float(y.mean() - beta * x_mean)
        resid = y - (alpha + beta * x)
        s2 = float((resid ** 2).sum() / (n - 2))
        var_alpha = s2 * (1.0 / n + x_mean ** 2 / Sxx)
        se_alpha = float(np.sqrt(var_alpha)) if var_alpha > 0 else float("inf")
        t = alpha / se_alpha if se_alpha > 0 else 0.0
        passes = t >= t_min
        ann = alpha * 252
        mark = "✓" if passes else "✗"
        print(f"{hid:>4}  {n:>7}  {ann:>8.3f}  {t:>7.2f}  {mark:>5}  {names[hid]}")
        results.append({"id": hid, "n": n, "alpha_yr": ann, "t": t, "passes": passes, "skipped": False})

    # Summary
    evaluated = [r for r in results if not r.get("skipped")]
    n_pass = sum(1 for r in evaluated if r["passes"])
    n_fail = len(evaluated) - n_pass
    n_skip = sum(1 for r in results if r.get("skipped"))
    print()
    print(f"# Summary: {n_pass}/{len(evaluated)} pass ({100*n_pass/len(evaluated):.1f}%)  "
          f"{n_fail} fail  {n_skip} skipped (overlap < {_MIN_BASELINE_OVERLAP_DAYS}d)")
    if evaluated:
        ts = np.array([r["t"] for r in evaluated])
        print(f"# t-stat percentiles: p10={np.percentile(ts,10):.2f}  "
              f"p50={np.percentile(ts,50):.2f}  p90={np.percentile(ts,90):.2f}  "
              f"max={ts.max():.2f}")


if __name__ == "__main__":
    main()
