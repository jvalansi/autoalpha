"""Trade one paper-book signal on the $10k "autoalpha - small" Alpaca account.

Rehearsal for real money at small size: the same targets the $100k simulated paper
book trades (scripts/run_paper.py), sized off this account's equity, with
fractional remainders (AlpacaExecutor(fractional_remainder=True): whole shares
market-on-open, the rest as a day market order) so ~140 names fit in $10k. The
nightly report compares this account with the simulated book for the same signal.

  - Runs pre-market (08:00-09:25 ET) on trading days, once per day, after the
    nightly job has appended the previous session's bar to the vault. The signal
    on bar N fills at bar N+1's open, the same timing as the simulation.
  - Refuses to trade on a stale vault (last bar is not the previous session).
  - Account: ALPACA_10K_API_KEY / ALPACA_10K_SECRET_KEY (ALPACA_10K_BASE_URL,
    default paper). Refuses the real-money endpoint unless LIVE_TRADING_CONFIRMED=yes.
  - The account is dedicated: names it holds that are not in the targets are sold.

Usage:
    python scripts/run_live_small.py [--dry-run] [--ignore-clock]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoalpha.core.runner import _normalize_targets
from autoalpha.execution.alpaca import AlpacaExecutor
from autoalpha.research.memory import HypothesisMemory
from run_paper import load_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
SIGNAL = "revision quality value drift"  # the only signal with positive holdout and paper alpha (2026-11-01)
LIVE_DIR = Path("data/live")
STATE_PATH = LIVE_DIR / "small_state.json"
ORDERS_LOG = LIVE_DIR / "small_orders.jsonl"
VAULT = Path("data/vault_data.parquet")
WINDOW = (8 * 60, 9 * 60 + 25)  # ET minutes; opg orders are rejected after 9:28


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(state: dict) -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(STATE_PATH)


def executor_from_env(dry_run: bool = False) -> AlpacaExecutor:
    return AlpacaExecutor(api_key=os.environ.get("ALPACA_10K_API_KEY", ""),
                          secret_key=os.environ.get("ALPACA_10K_SECRET_KEY", ""),
                          base_url=os.environ.get("ALPACA_10K_BASE_URL") or None,
                          fractional_remainder=True, dry_run=dry_run)


def signal_strategy(name: str):
    memory = HypothesisMemory()
    rows = memory._conn.execute("SELECT hypothesis_json FROM hypotheses WHERE status='active'").fetchall()
    memory.close()
    for (hj,) in rows:
        h = json.loads(hj)
        if h.get("concise_reason") == name:
            return load_strategy(h["predict_body"])
    raise SystemExit(f"signal {name!r} is not active in the hypothesis book")


def latest_bar() -> tuple[pd.Timestamp, pd.DataFrame]:
    dates = ds.dataset(VAULT).to_table(columns=["date"]).column("date").to_pandas()
    last = pd.Timestamp(dates.max())
    t = ds.dataset(VAULT).to_table(filter=ds.field("date") == last.to_pydatetime()).to_pandas()
    df = t.reset_index() if "ticker" not in t.columns else t
    return last, df.drop_duplicates("ticker", keep="last").set_index("ticker")


def report_line(signal_results: list[dict]) -> str | None:
    """Nightly-report line: this account vs the simulated book for the same signal.

    Simulated returns are dated d for open(d-1) -> open(d) (SimExecutor marks at the
    open); the account trades from start_date's open, so the comparison compounds
    simulated returns dated after start_date. Account equity is as of the last close."""
    state = load_state()
    if "start_date" not in state:
        return None
    try:
        equity = executor_from_env().account_equity()
    except Exception as exc:  # the report must not fail because the broker is down
        return f"*Small account:* unavailable — {type(exc).__name__}"
    acct = equity / state["start_equity"] - 1
    sim = next((s for s in signal_results if s["name"] == state["signal"]), None)
    sim_txt = "n/a"
    if sim:
        rets = pd.Series(sim["daily_returns"], dtype=float)
        rets = rets[pd.to_datetime(rets.index) > pd.Timestamp(state["start_date"])]
        sim_txt = f"{(1 + rets).prod() - 1:+.2%}"
    kind = "paper" if state.get("paper", True) else "REAL MONEY"
    return (f"*Small account ($10k {kind}, {state['signal']}):* equity ${equity:,.0f} ({acct:+.2%}) "
            f"vs simulated {sim_txt} since {state['start_date']} · last run {state.get('last_run')}: "
            f"{state.get('last_targets', 0)} names, {state.get('last_orders', 0)} orders, {state.get('last_rejected', 0)} rejected legs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="compute and log orders without placing them")
    parser.add_argument("--ignore-clock", action="store_true", help="skip the pre-market/once-a-day guard")
    args = parser.parse_args()

    now = datetime.now(ET)
    today = pd.Timestamp(now.date())
    executor = executor_from_env(args.dry_run)
    if not executor.is_paper and os.getenv("LIVE_TRADING_CONFIRMED", "").lower() != "yes":
        raise SystemExit("Refusing the real-money endpoint without LIVE_TRADING_CONFIRMED=yes")

    state = load_state()
    if not args.ignore_clock:
        if not WINDOW[0] <= now.hour * 60 + now.minute < WINDOW[1]:
            log.info("Outside the 08:00-09:25 ET window — nothing to do")
            return
        if state.get("last_run") == str(today.date()):
            log.info("Already ran today")
            return

    cal = executor._request("GET", "/v2/calendar", params={
        "start": (today - pd.Timedelta(days=12)).strftime("%Y-%m-%d"), "end": today.strftime("%Y-%m-%d")})
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime([d["date"] for d in cal])))
    if not args.ignore_clock and sessions[-1] != today:
        log.info("%s is not a trading day", today.date())
        return
    prev_session = sessions[sessions < today][-1]

    bar_date, bar = latest_bar()
    if bar_date != prev_session:
        raise SystemExit(f"vault ends {bar_date.date()}, expected {prev_session.date()} — not trading on stale data")

    targets = _normalize_targets(signal_strategy(SIGNAL).predict(bar, bar_date=bar_date))
    prices = {t: float(p) for t, p in bar["Close"].items() if pd.notna(p) and p > 0}
    equity = executor.account_equity()
    log.info("%s: %d names, gross %.2f, equity $%.2f", SIGNAL, len(targets), sum(targets.values()), equity)

    if not args.dry_run:
        # Mark the attempt before the first order: a run that dies partway must not be
        # repeated by the second cron fire, which could stack orders on top of ones
        # whose cancellation hasn't settled. A partial book is fixed by the next day's run.
        # Counts reset with it, so a dead run can't report the previous day's as its own.
        state.update(last_run=str(today.date()), last_targets=0, last_orders=0, last_rejected=0)
        save_state(state)
    executor.execute(targets, today.date(), prices)
    # orders whose split dropped every leg (sub-share delta on an unfractionable name) never reached the broker
    orders = [o for o in executor.orders() if o.get("dry_run") or o.get("response")]
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    with (Path("/tmp/small_orders_dry_run.jsonl") if args.dry_run else ORDERS_LOG).open("a") as f:
        for o in orders:
            f.write(json.dumps({"date": str(today.date()), "signal_bar": str(bar_date.date()), **o}, default=str) + "\n")
    rejected = sum(1 for o in orders for leg in (o.get("response") or {}).values()
                   if isinstance(leg, dict) and "error" in leg)
    log.info("%d orders, %d rejected legs", len(orders), rejected)

    if not args.dry_run:
        state.setdefault("start_date", str(today.date()))
        state.setdefault("start_equity", equity)
        state["signal"] = SIGNAL
        state["paper"] = executor.is_paper
        state["last_targets"] = len(targets)
        state["last_orders"] = len(orders)
        state["last_rejected"] = rejected
        save_state(state)
        # Routine runs are covered by the nightly report; rejected legs or an empty book alert now.
        msg = (f"**autoalpha $10k small account ({'paper' if executor.is_paper else 'REAL MONEY'}) — {today.date()}**\n"
               f"{SIGNAL}: {len(targets)} names, equity ${equity:,.0f} · {len(orders)} orders, {rejected} rejected legs")
        if (rejected or not targets) and os.environ.get("DISCORD_BOT_TOKEN"):
            subprocess.run([sys.executable, str(Path(__file__).with_name("post_to_discord.py"))], input=msg, text=True, check=False)


if __name__ == "__main__":
    main()
