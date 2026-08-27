#!/usr/bin/env python
"""Promotion gate: can a signal go live?

Backtest evidence cannot carry this decision. The research loop has run 3,541
trials, and the Deflated Sharpe benchmark grows with the trial count
(E[max SR] ~ sqrt(2 ln N)) — at N=3,541 and ~1,600 OOS bars it sits near an
annualized Sharpe of 1.4, above anything the book has produced. Worse, the loop
never applied that penalty at admission (it gates on PSR vs a fixed market
benchmark), so in-sample numbers describe a selection process, not an edge.

So promotion rests on the two windows that were never used for selection:

  Stage 1  SCREEN   in-sample — reported for context, NOT a gate
  Stage 2  HOLDOUT  vault period, locked before any signal was generated
  Stage 3  FORWARD  paper trading, dates after every backtest

The two untainted windows are disjoint and contiguous, so the gate POOLS them
into a single regression rather than testing each separately. Two underpowered
tests are strictly worse than one powered test on the same data: 501 holdout
days plus ~60 paper days is 2.2 years of post-selection evidence, and requiring
t >= 2 from each half independently would discard most of that power. Both
windows must still be individually positive — pooling should not let a strong
holdout paper over a forward window that turned negative.

Usage:
    python scripts/promotion_status.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VALIDATION = Path("data/signal_validation.json")
VAULT = Path("data/vault_results.json")
PAPER = Path("data/paper_pnl.json")

MIN_ALPHA_T = 2.0      # on the POOLED untainted window
MAX_DD_PCT = 30.0      # hard drawdown constraint (PLAN.md Phase 2)
MIN_FORWARD_BARS = 50  # minimum paper observations


def pooled_alpha(signal_name: str, vault: dict, paper: dict) -> dict:
    """Regress a signal on its benchmark across holdout + forward as one series."""
    from autoalpha.evaluation.alpha import compute_benchmark_alpha

    v_sig = (vault.get("signals", {}).get(signal_name) or {}).get("daily_returns", {})
    v_bm = vault.get("benchmark_daily_returns", {})
    p_sig = next((s["daily_returns"] for s in paper.get("signals", [])
                  if s["name"] == signal_name), {})
    p_bm = (paper.get("benchmark", {}) or {}).get("daily_returns", {})

    if not (v_sig and p_sig and v_bm and p_bm):
        return {"available": False, "reason": "missing one of the two windows"}

    sig = pd.Series({**v_sig, **p_sig}, dtype=float)
    bm = pd.Series({**v_bm, **p_bm}, dtype=float)
    sig.index = pd.to_datetime(sig.index)
    bm.index = pd.to_datetime(bm.index)
    stats = compute_benchmark_alpha(sig.sort_index(), bm.sort_index())
    stats.pop("residuals", None)
    return stats


def gate_verdict(holdout: dict, forward: dict, pooled: dict) -> tuple[bool, list[str]]:
    """Apply the promotion gate to one signal's untainted evidence.

    Returns (passes, reasons_it_failed). In-sample numbers are deliberately not
    an input — they describe the selection process, not an edge.
    """
    reasons: list[str] = []

    p_t = pooled.get("alpha_t") if pooled.get("available") else None
    if p_t is None:
        reasons.append("no pooled result")
    elif p_t < MIN_ALPHA_T:
        reasons.append(f"pooled t={p_t:.2f}")

    # Sign consistency: pooling must not let one window mask a failure in the other.
    for label, window in (("holdout", holdout), ("forward", forward)):
        a = window.get("alpha_annualized")
        if a is None:
            reasons.append(f"no {label} result")
        elif a <= 0:
            reasons.append(f"{label} alpha {a:+.1f}%")

    v_dd = abs(holdout["max_drawdown"]) if holdout.get("max_drawdown") is not None else None
    if v_dd is not None and v_dd >= MAX_DD_PCT:
        reasons.append(f"holdout DD={v_dd:.0f}%")

    f_n = forward.get("n_bars")
    if f_n is not None and f_n < MIN_FORWARD_BARS:
        reasons.append(f"only {f_n} forward bars")

    return (not reasons), reasons


def _load(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fmt(v, spec=">7.2f", missing="   n/a"):
    return missing if v is None else format(v, spec)


def main() -> None:
    validation = _load(VALIDATION) or []
    vault = _load(VAULT) or {}
    paper = _load(PAPER) or {}

    screen = {r["name"]: r for r in validation}
    holdout = vault.get("signals", {})
    forward = {s["name"]: s for s in paper.get("signals", [])}

    names = sorted(set(screen) | set(forward) | {k for k in holdout if k != "COMBINED PORTFOLIO"})
    if not names:
        print("No signal data found. Run validate_signals.py, evaluate_vault.py, run_paper.py.")
        sys.exit(1)

    if vault:
        print(f"Holdout window : {vault.get('holdout_start')} → {vault.get('holdout_end')}")
    if paper:
        print(f"Forward window : {paper.get('paper_start')} → {paper.get('paper_end')} "
              f"({paper.get('n_trading_days')} trading days)")
    print(f"Gate           : POOLED alpha t ≥ {MIN_ALPHA_T} across both untainted windows, "
          f"each window positive, DD < {MAX_DD_PCT}%, ≥ {MIN_FORWARD_BARS} forward bars\n")

    print(f"{'Signal':<34} {'ISsh':>5} {'DSR':>6} | {'VaultA%':>7} {'t':>5} | "
          f"{'PaperA%':>7} {'t':>5} | {'PoolA%':>7} {'t':>5} {'n':>4} | Verdict")
    print("-" * 125)

    verdicts = {}
    for name in names:
        sc, hv, fw = screen.get(name, {}), holdout.get(name, {}), forward.get(name, {})

        v_t = hv.get("alpha_t")
        v_dd = abs(hv["max_drawdown"]) if hv.get("max_drawdown") is not None else None
        f_t = fw.get("alpha_t")
        f_n = fw.get("n_bars")

        pl = pooled_alpha(name, vault, paper)
        passed, reasons = gate_verdict(hv, fw, pl)
        verdict = "PROMOTE" if passed else "HOLD — " + ", ".join(reasons)
        verdicts[name] = verdict

        pl_a = pl.get("alpha_annualized")
        print(f"{name[:34]:<34} {_fmt(sc.get('new_sharpe'), '>5.2f', '  n/a')} "
              f"{_fmt(sc.get('true_dsr'), '>6.3f')} | "
              f"{_fmt(hv.get('alpha_annualized'), '>7.1f')} {_fmt(v_t, '>5.2f', '  n/a')} | "
              f"{_fmt(fw.get('alpha_annualized'), '>7.1f')} {_fmt(f_t, '>5.2f', '  n/a')} | "
              f"{_fmt(pl_a * 100 if pl_a is not None else None, '>7.1f')} "
              f"{_fmt(pl.get('alpha_t'), '>5.2f', '  n/a')} "
              f"{_fmt(pl.get('n_overlap'), '>4d', '  na')} | {verdict}")

    # Book level
    print()
    book_v = holdout.get("COMBINED PORTFOLIO", {})
    book_f = paper.get("alpha_vs_benchmark", {})
    book_f_t = book_f.get("alpha_t") if book_f.get("available") else None
    print("COMBINED BOOK")
    print(f"  holdout : alpha={_fmt(book_v.get('alpha_annualized'), '>6.1f')}%/yr  "
          f"t={_fmt(book_v.get('alpha_t'), '>5.2f')}  "
          f"DD={_fmt(abs(book_v['max_drawdown']) if book_v.get('max_drawdown') is not None else None, '>5.1f')}%")
    print(f"  forward : alpha={_fmt((book_f.get('alpha_annualized') or 0) * 100 if book_f.get('available') else None, '>6.1f')}%/yr  "
          f"t={_fmt(book_f_t, '>5.2f')}  "
          f"DD={_fmt(abs(paper.get('combined', {}).get('max_drawdown', 0)) or None, '>5.1f')}%")

    book_pool = pooled_alpha("COMBINED PORTFOLIO", vault, paper)
    if not book_pool.get("available"):
        # The paper file stores the book under `combined`, not as a named signal.
        import copy
        p2 = copy.deepcopy(paper)
        combo = pd.DataFrame({s["name"]: pd.Series(s["daily_returns"])
                              for s in paper.get("signals", [])}).mean(axis=1)
        p2["signals"] = [{"name": "COMBINED PORTFOLIO",
                          "daily_returns": {str(k): float(v) for k, v in combo.items()}}]
        book_pool = pooled_alpha("COMBINED PORTFOLIO", vault, p2)
    if book_pool.get("available"):
        print(f"  pooled  : alpha={book_pool['alpha_annualized']*100:>6.1f}%/yr  "
              f"t={book_pool['alpha_t']:>5.2f}  IR={book_pool['information_ratio']:>5.2f}  "
              f"n={book_pool['n_overlap']}")

        # At a stable IR, t grows as IR x sqrt(years) — so the remaining wait is
        # a function of the edge's quality, not a fixed calendar period.
        ir, t = book_pool["information_ratio"], book_pool["alpha_t"]
        if 0 < t < MIN_ALPHA_T and ir > 0:
            need = int((MIN_ALPHA_T / ir) ** 2 * 252)
            print(f"            → t={MIN_ALPHA_T} needs ~{need} pooled days at this IR "
                  f"({max(need - book_pool['n_overlap'], 0)} more)")

    n_promote = sum(1 for v in verdicts.values() if v == "PROMOTE")
    print(f"\n{n_promote}/{len(verdicts)} signals clear the gate.")
    if n_promote == 0:
        print("NOT READY FOR LIVE TRADING.")


if __name__ == "__main__":
    main()
