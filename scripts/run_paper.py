"""Nightly paper trading run for all active signals.

vault_data.parquet is the enriched live feed (2024-05-21 → today).
Only the paper-period slice is loaded via PyArrow filter pushdown to keep
memory usage low on the 2 GB instance.

P&L is measured from PAPER_START (stored in data/paper_state.json on
first run) so only forward-looking bars count toward the 30-day gate.

Usage:
    python scripts/run_paper.py [--rebuild-vault] [--slack-channel ID] [--slack-thread-ts TS]

Output: data/paper_pnl.json  (overwritten each run)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoalpha.core.executors import SimExecutor
from autoalpha.core.providers import HistoricalProvider
from autoalpha.core.runner import Runner
from autoalpha.evaluation.alpha import compute_benchmark_alpha, ff5_alpha_stats
from autoalpha.evaluation.library import SignalLibrary
from autoalpha.research.code_validator import wrap_predict_body
from autoalpha.research.memory import HypothesisMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

VAULT_DATA = Path("data/vault_data.parquet")
STATE_FILE = Path("data/paper_state.json")
PNL_FILE = Path("data/paper_pnl.json")


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------

def _slack_post(text: str) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    channel = os.environ.get("SLACK_LOOP_CHANNEL", "")
    thread_ts = os.environ.get("SLACK_LOOP_THREAD_TS", "")
    if not (token and channel):
        return
    payload: dict = {"channel": channel, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    try:
        requests.post("https://slack.com/api/chat.postMessage",
                      headers={"Authorization": f"Bearer {token}"},
                      json=payload, timeout=10)
    except Exception as exc:
        log.warning("Slack post failed: %s", exc)


# ---------------------------------------------------------------------------
# Strategy loading
# ---------------------------------------------------------------------------

def load_strategy(predict_body: str):
    """Instantiate a Strategy from a predict() body string."""
    source = wrap_predict_body(predict_body)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(source)
        path = f.name
    spec = importlib.util.spec_from_file_location("_strat", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.strategy


# ---------------------------------------------------------------------------
# Provider backed by a pre-loaded parquet
# ---------------------------------------------------------------------------

def make_parquet_provider(mi: pd.DataFrame) -> HistoricalProvider:
    """Return a HistoricalProvider whose history/bars methods read from mi."""
    date_vals = mi.index.get_level_values("date")
    bar_lookup = {d: grp.droplevel("date") for d, grp in mi.groupby(level="date")}
    sorted_dates = sorted(bar_lookup.keys())

    def _history(tkrs, s, e):
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        return mi[(date_vals >= s_ts) & (date_vals <= e_ts)]

    def _bars(tkrs, s, e):
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        for d in sorted_dates:
            if s_ts <= d <= e_ts:
                yield d, bar_lookup[d]

    provider = HistoricalProvider.__new__(HistoricalProvider)
    provider._history_fn = _history
    provider._bars_fn = _bars
    provider.history = _history
    provider.bars = _bars
    return provider


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _report_level(today: pd.Timestamp) -> str:
    """Return the highest-priority report level for today."""
    m, d = today.month, today.day
    # Quarterly: first calendar day of Jan/Apr/Jul/Oct (or first weekday if weekend)
    if m in (1, 4, 7, 10) and d <= 3 and today.weekday() < 5:
        # Check no earlier weekday existed in month (i.e. this is the first weekday)
        if all((today - pd.Timedelta(days=i)).month != m or
               (today - pd.Timedelta(days=i)).weekday() >= 5
               for i in range(1, d)):
            return "quarterly"
    # Monthly: first weekday of the month
    if d <= 3 and today.weekday() < 5:
        if all((today - pd.Timedelta(days=i)).month != m or
               (today - pd.Timedelta(days=i)).weekday() >= 5
               for i in range(1, d)):
            return "monthly"
    # Weekly: Friday
    if today.weekday() == 4:
        return "weekly"
    return "daily"


def _fmt_signal_table(signal_results: list[dict]) -> str:
    lines = ["*Per-signal attribution:*"]
    for s in signal_results:
        alpha = s["total_return"] - 0.0  # will be relative to benchmark below
        alpha_txt = (f"  α={s['alpha_annualized']:+.1f}%/yr t={s['alpha_t']:.2f}"
                     if s.get("alpha_t") is not None else "")
        lines.append(
            f"  • {s['name']}: Ret={s['total_return']:+.1f}%  "
            f"Sharpe={s['sharpe']:.2f}  DD={s['max_drawdown']:.1f}%  n={s['n_bars']}{alpha_txt}"
        )
    return "\n".join(lines)


_MIN_ALPHA_T = 2.0  # benchmark-relative alpha must be statistically distinguishable from 0


def _go_no_go(combo: pd.Series, combo_dd: float, n_paper_days: int,
              alpha_stats: dict | None = None) -> tuple[bool, str]:
    """Apply the live-deployment gate to the paper combined portfolio.

    Sharpe and drawdown alone can be satisfied by a book that is just long the
    market. The alpha t-stat is the term that asks whether the book beat the
    universe it could have held instead.
    """
    from autoalpha.evaluation.sharpe import probabilistic_sharpe
    if len(combo) < 50:
        return False, f"Only {len(combo)} paper return observations — need ≥50"
    psr = float(probabilistic_sharpe(combo, benchmark_sr=0.62))
    dd_pct = abs(combo_dd) * 100
    active = int((combo != 0).sum())

    alpha_stats = alpha_stats or {}
    alpha_t = float(alpha_stats.get("alpha_t", 0.0)) if alpha_stats.get("available") else 0.0
    alpha_ok = alpha_t >= _MIN_ALPHA_T

    passed = psr > 0.65 and dd_pct < 30.0 and active >= 50 and alpha_ok
    verdict = ":white_check_mark: GO LIVE" if passed else ":x: NOT YET"
    detail = (
        f"PSR={psr:.3f} {'✓' if psr > 0.65 else '✗'} (need >0.65)  |  "
        f"DD={dd_pct:.1f}% {'✓' if dd_pct < 30 else '✗'} (need <30%)  |  "
        f"Active days={active} {'✓' if active >= 50 else '✗'} (need ≥50)  |  "
        f"Alpha t={alpha_t:.2f} {'✓' if alpha_ok else '✗'} (need ≥{_MIN_ALPHA_T})"
    )
    return passed, f"{verdict}\n{detail}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-vault", action="store_true",
                        help="Rebuild vault_data.parquet before running")
    parser.add_argument("--slack-channel", default=None)
    parser.add_argument("--slack-thread-ts", default=None)
    parser.add_argument("--report-level", default=None,
                        choices=["daily", "weekly", "monthly", "quarterly"],
                        help="Override auto-detected report level")
    args = parser.parse_args()

    if args.slack_channel:
        os.environ["SLACK_LOOP_CHANNEL"] = args.slack_channel
    if args.slack_thread_ts:
        os.environ["SLACK_LOOP_THREAD_TS"] = args.slack_thread_ts

    today = pd.Timestamp.today().normalize()
    level = args.report_level or _report_level(today)

    if args.rebuild_vault:
        log.info("Rebuilding vault_data.parquet...")
        subprocess.run([sys.executable, "scripts/build_vault_dataset.py"], check=True)

    if not VAULT_DATA.exists():
        log.error("vault_data.parquet not found — run scripts/build_vault_dataset.py first")
        sys.exit(1)

    # Determine paper start date before loading data (so we can filter the parquet)
    state: dict = {}
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
    if "paper_start" not in state:
        state["paper_start"] = today.strftime("%Y-%m-%d")
        STATE_FILE.write_text(json.dumps(state))
        log.info("Paper trading started — paper_start=%s", state["paper_start"])
    paper_start = pd.Timestamp(state["paper_start"])

    # Load only the paper-period slice from vault using PyArrow date filter pushdown.
    # This avoids loading the full 1.3M-row dataset (~1 GB) when we only need a few weeks.
    log.info("Loading vault data from %s onwards...", paper_start.date())
    import pyarrow.dataset as _ds
    _vault_ds = _ds.dataset(str(VAULT_DATA), format="parquet")
    _vault_table = _vault_ds.to_table(
        filter=_ds.field("date") >= paper_start.to_pydatetime()
    )
    vault_mi = _vault_table.to_pandas()
    if not isinstance(vault_mi.index, pd.MultiIndex):
        idx_cols = [c for c in ["date", "ticker"] if c in vault_mi.columns]
        if idx_cols:
            vault_mi = vault_mi.set_index(idx_cols)
    vault_mi.index.names = ["date", "ticker"]
    vault_mi.index = pd.MultiIndex.from_arrays([
        pd.to_datetime(vault_mi.index.get_level_values("date")),
        vault_mi.index.get_level_values("ticker"),
    ], names=["date", "ticker"])
    vault_mi = vault_mi.sort_index(level="date", sort_remaining=False)
    if vault_mi.index.duplicated().any():
        n_dupes = vault_mi.index.duplicated().sum()
        log.warning("Dropping %d duplicate (date, ticker) rows from vault", n_dupes)
        vault_mi = vault_mi[~vault_mi.index.duplicated(keep="last")]

    tickers = vault_mi.index.get_level_values("ticker").unique().tolist()
    vault_dates = vault_mi.index.get_level_values("date").unique().sort_values()
    paper_dates = vault_dates[vault_dates >= paper_start]

    if paper_dates.empty:
        log.info("No paper bars yet (paper_start=%s, latest vault bar=%s)",
                 paper_start.date(), vault_dates.max().date() if len(vault_dates) else "none")
        return

    paper_start_date = paper_dates.min().to_pydatetime().date()
    paper_end_date = paper_dates.max().to_pydatetime().date()
    n_paper_days = len(paper_dates)
    log.info("Paper period: %s → %s  (%d trading days, %d tickers)",
             paper_start_date, paper_end_date, n_paper_days, len(tickers))

    # Load active signals + any accepted in the last 12 hours
    memory = HypothesisMemory()
    rows = memory._conn.execute(
        "SELECT id, hypothesis_json FROM hypotheses WHERE status='active' ORDER BY id"
    ).fetchall()
    since = (today - pd.Timedelta(hours=12)).isoformat()
    new_signal_rows = memory._conn.execute(
        "SELECT id, hypothesis_json, sharpe, dsr, max_drawdown FROM hypotheses "
        "WHERE status='active' AND created_at >= ? ORDER BY id",
        (since,),
    ).fetchall()
    memory.close()
    log.info("Active signals: %d", len(rows))

    # fit() is a no-op for all generated strategies — no need to load loop_data
    vault_provider = make_parquet_provider(vault_mi)

    # Benchmark: equal-weight all tickers, buy-and-hold over paper period
    bm_close = pd.concat([
        df["Close"].rename(d)
        for (d, df) in vault_provider.bars(tickers, paper_start_date, paper_end_date)
    ], axis=1).T  # shape: dates × tickers
    if len(bm_close) >= 2:
        bm_rets = bm_close.pct_change().dropna(how="all").mean(axis=1)  # skipna=True by default
    else:
        bm_rets = pd.Series(dtype=float)

    signal_results = []

    for row_id, hyp_json in rows:
        hyp = json.loads(hyp_json)
        name = hyp.get("concise_reason", f"signal_{row_id}")
        log.info("Running paper: %s", name)

        try:
            strategy = load_strategy(hyp["predict_body"])
        except Exception as exc:
            log.warning("  Failed to load strategy: %s", exc)
            continue

        # fit() is a no-op for all generated strategies — skip

        # Run on paper period
        executor = SimExecutor(initial_capital=100_000, cost_bps=11)
        runner = Runner(strategy, vault_provider, executor, tickers)

        prev_targets: dict = {}
        for bar_date, bar_df in vault_provider.bars(tickers, paper_start_date, paper_end_date):
            open_prices = bar_df["Open"].to_dict() if "Open" in bar_df.columns else {}
            if prev_targets and open_prices:
                executor.execute(prev_targets, bar_date, open_prices)
            prev_targets = strategy.predict(bar_df, bar_date=pd.Timestamp(bar_date))

        rets = executor.returns()
        if rets.empty:
            log.info("  %s: no trades", name)
            continue

        ann_sharpe = (rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0.0
        total_ret = (1 + rets).prod() - 1
        nav = executor.nav_series()
        drawdown = ((nav / nav.cummax()) - 1).min()

        log.info("  %s: Sharpe=%.2f  TotalRet=%.1f%%  DD=%.1f%%  n=%d",
                 name, ann_sharpe, total_ret * 100, drawdown * 100, len(rets))

        sig_alpha = compute_benchmark_alpha(rets, bm_rets) if len(bm_rets) > 1 else {"available": False}

        signal_results.append({
            "name": name,
            "sharpe": round(ann_sharpe, 3),
            "total_return": round(total_ret * 100, 2),
            "max_drawdown": round(drawdown * 100, 2),
            "n_bars": len(rets),
            "alpha_annualized": round(sig_alpha["alpha_annualized"] * 100, 2) if sig_alpha.get("available") else None,
            "alpha_t": round(sig_alpha["alpha_t"], 2) if sig_alpha.get("available") else None,
            "beta": round(sig_alpha["beta"], 2) if sig_alpha.get("available") else None,
            "daily_returns": {str(d.date()): round(r, 6) for d, r in rets.items()},
        })

    # Combined portfolio: equal-weight signals, plus a Darwinian-weighted variant
    combo = pd.Series(dtype=float)
    wcombo = pd.Series(dtype=float)
    darwin_weights: dict[str, float] = {}
    if signal_results:
        all_rets = pd.DataFrame({s["name"]: pd.Series(s["daily_returns"]) for s in signal_results})
        all_rets.index = pd.to_datetime(all_rets.index)
        combo = all_rets.mean(axis=1).dropna()
        combo_sharpe = (combo.mean() / combo.std() * (252 ** 0.5)) if combo.std() > 0 else 0.0
        combo_ret = (1 + combo).prod() - 1
        combo_dd = ((1 + combo).cumprod() / (1 + combo).cumprod().cummax() - 1).min()

        # Darwinian-weighted book. Weights are maintained nightly by
        # scripts/update_weights.py; missing names default to 1.0.
        with SignalLibrary() as _lib:
            stored = _lib.all_weights()
        darwin_weights = {s["name"]: float(stored.get(s["name"], 1.0)) for s in signal_results}
        w = pd.Series(darwin_weights)
        if w.sum() > 0:
            wcombo = (all_rets[w.index] * w).sum(axis=1).div(w.sum()).dropna()
            wcombo_sharpe = (wcombo.mean() / wcombo.std() * (252 ** 0.5)) if wcombo.std() > 0 else 0.0
            wcombo_ret = (1 + wcombo).prod() - 1
            wcombo_dd = ((1 + wcombo).cumprod() / (1 + wcombo).cumprod().cummax() - 1).min()
        else:
            wcombo_sharpe = wcombo_ret = wcombo_dd = 0.0
    else:
        combo_sharpe = combo_ret = combo_dd = 0.0
        wcombo_sharpe = wcombo_ret = wcombo_dd = 0.0

    bm_sharpe = (bm_rets.mean() / bm_rets.std() * (252 ** 0.5)) if len(bm_rets) > 1 and bm_rets.std() > 0 else 0.0
    bm_ret = (1 + bm_rets).prod() - 1

    # Alpha of the combined book. Benchmark alpha covers the whole window;
    # FF5 lags ~2 months so it is reported separately with its coverage.
    bench_alpha = compute_benchmark_alpha(combo, bm_rets) if len(combo) > 1 and len(bm_rets) > 1 \
        else {"available": False, "reason": "insufficient overlap"}
    ff5 = ff5_alpha_stats(combo) if len(combo) > 1 else {"available": False}

    def _alpha_payload(d: dict) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in d.items() if k != "residuals"}

    output = {
        "paper_start": state["paper_start"],
        "paper_end": paper_end_date.isoformat(),
        "n_trading_days": n_paper_days,
        "benchmark": {
            "sharpe": round(bm_sharpe, 3),
            "total_return": round(bm_ret * 100, 2),
            "daily_returns": {str(d.date()): round(r, 6) for d, r in bm_rets.items()},
        },
        "combined": {
            "sharpe": round(combo_sharpe, 3),
            "total_return": round(combo_ret * 100, 2),
            "max_drawdown": round(combo_dd * 100, 2),
        },
        "combined_darwinian": {
            "sharpe": round(wcombo_sharpe, 3),
            "total_return": round(wcombo_ret * 100, 2),
            "max_drawdown": round(wcombo_dd * 100, 2),
            "weights": {k: round(v, 3) for k, v in darwin_weights.items()},
        },
        "alpha_vs_benchmark": _alpha_payload(bench_alpha),
        "alpha_ff5": _alpha_payload(ff5),
        "signals": signal_results,
    }

    PNL_FILE.write_text(json.dumps(output, indent=2))
    log.info("Saved %s", PNL_FILE)

    # Build Slack message based on report level
    header = (
        f"*autoalpha paper {level} report* — day {n_paper_days} "
        f"({paper_start.date()} → {paper_end_date})"
    )
    summary = (
        f"Combined: Sharpe={combo_sharpe:.2f}  Ret={combo_ret*100:+.1f}%  DD={combo_dd*100:.1f}%\n"
        f"Darwinian-weighted: Sharpe={wcombo_sharpe:.2f}  Ret={wcombo_ret*100:+.1f}%  "
        f"DD={wcombo_dd*100:.1f}%\n"
        f"Benchmark: Sharpe={bm_sharpe:.2f}  Ret={bm_ret*100:+.1f}%  "
        f"Alpha={combo_ret*100 - bm_ret*100:+.1f}%"
    )

    if bench_alpha.get("available"):
        alpha_line = (
            f"Alpha vs equal-weight book: {bench_alpha['alpha_annualized']*100:+.1f}%/yr  "
            f"t={bench_alpha['alpha_t']:.2f}  beta={bench_alpha['beta']:.2f}  "
            f"IR={bench_alpha['information_ratio']:.2f}  (n={bench_alpha['n_overlap']})"
        )
        # At a stable IR, the alpha t-stat grows as IR × sqrt(years). Report how
        # much more paper history the current edge needs to clear the gate.
        ir = bench_alpha["information_ratio"]
        if 0 < bench_alpha["alpha_t"] < _MIN_ALPHA_T and ir > 0:
            days_needed = int((_MIN_ALPHA_T / ir) ** 2 * 252)
            alpha_line += (
                f"\n  → t={_MIN_ALPHA_T} needs ~{days_needed} paper days at this IR "
                f"({max(days_needed - bench_alpha['n_overlap'], 0)} more)"
            )
    else:
        alpha_line = f"Alpha vs equal-weight book: unavailable — {bench_alpha.get('reason', '?')}"

    if ff5.get("available"):
        alpha_line += (
            f"\nFF5 alpha: {ff5['alpha_annualized']*100:+.1f}%/yr  t={ff5['alpha_t']:.2f}  "
            f"(n={ff5['n_overlap']}/{ff5['n_total']} days covered)"
        )
    else:
        alpha_line += f"\nFF5 alpha: n/a — {ff5.get('reason', 'unavailable')}"

    lines = [header, summary, alpha_line]

    if new_signal_rows:
        new_lines = ["*New signals tonight:*"]
        for r in new_signal_rows:
            h = json.loads(r["hypothesis_json"])
            name = h.get("concise_reason", f"signal_{r['id']}")
            sharpe = r["sharpe"] or 0.0
            dsr = r["dsr"] or 0.0
            dd = abs(r["max_drawdown"] or 0.0) * 100
            new_lines.append(f"  • {name}  Sharpe={sharpe:.2f}  DSR={dsr:.3f}  DD={dd:.1f}%")
        lines.append("\n".join(new_lines))
    else:
        lines.append("*New signals tonight:* none")

    if level in ("weekly", "monthly", "quarterly") and signal_results:
        lines.append(_fmt_signal_table(signal_results))

    if level == "quarterly":
        if signal_results:
            all_rets_df = pd.DataFrame({s["name"]: pd.Series(s["daily_returns"]) for s in signal_results})
            all_rets_df.index = pd.to_datetime(all_rets_df.index)
            combo_series = all_rets_df.mean(axis=1).dropna()
        else:
            combo_series = pd.Series(dtype=float)
        _, verdict = _go_no_go(combo_series, combo_dd, n_paper_days, bench_alpha)
        lines.append(f"\n*Go/No-Go for live trading:*\n{verdict}")

    report_text = "\n".join(lines)
    Path("data/last_paper_report.txt").write_text(report_text)
    log.info("Posting %s report to Slack", level)
    _slack_post(report_text)


if __name__ == "__main__":
    main()
