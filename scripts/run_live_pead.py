"""Trade earnings-trader's PEAD rules live at the open, on autoalpha's Alpaca account.

Runs once per trading day in the first half hour after the open, like
earnings-trader's 9:30 ET scan: exits/stop updates for open positions, then
entries on today's earnings reactions (AMC reports from the previous session,
BMO reports today), filled as same-session market orders.

  - Account: ALPACA_* from the environment; refuses the real-money endpoint
    unless LIVE_TRADING_CONFIRMED=yes.
  - Risk: RiskGuard (4% daily loss, 15% drawdown, data/live/HALT) blocks entries,
    never exits.
  - Reconciliation: the broker is the system of record for holdings. Positions in
    state but not at the broker are dropped; broker positions the strategy doesn't
    hold are closed (the account is dedicated to this runner).
  - Universe: tickers in data/vault_data.parquet (the universe the gate evaluated).
  - Held positions keep their share count (no daily rebalance), like earnings-trader.

Usage:
    python scripts/run_live_pead.py [--dry-run] [--ignore-clock]
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

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoalpha.data.event_features import FALLBACK_ETF, SECTOR_ETF_MAP, fetch_earnings_calendar
from autoalpha.execution.alpaca import AlpacaExecutor
from autoalpha.execution.risk import RiskGuard
from autoalpha.strategies.pead_et import ATR_PERIOD, MAX_POSITIONS, EarningsTraderPEAD

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
LIVE_DIR = Path("data/live")
STATE_PATH = LIVE_DIR / "pead_state.json"
ORDERS_LOG = LIVE_DIR / "pead_orders.jsonl"
VAULT = Path("data/vault_data.parquet")
COMPARE_LOG = LIVE_DIR / "pead_compare.jsonl"
ET_DATA = Path("/home/ubuntu/earnings-trader/data")
FMP_EARNINGS = "https://financialmodelingprep.com/stable/earnings"


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {"positions": {}, "last_run": None}


def save_state(state: dict) -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(STATE_PATH)


def earnings_trader_day(today: pd.Timestamp) -> dict:
    """earnings-trader's book, today's fills and halt state, read from its data files."""
    def read(name, default):
        path = ET_DATA / name
        return json.loads(path.read_text()) if path.exists() else default
    trades = []
    if (ET_DATA / "trades_log.jsonl").exists():
        for line in (ET_DATA / "trades_log.jsonl").read_text().splitlines():
            t = json.loads(line)
            if pd.Timestamp(t["timestamp"]).tz_convert(ET).date() == today.date():
                trades.append({k: t.get(k) for k in ("ticker", "action", "quantity", "fill_price", "status")})
    risk = read("risk_state.json", {})
    return {"holding": sorted(p["ticker"] for p in read("positions.json", [])), "trades": trades,
            "halted_since": risk.get("halted_since"), "halt_reasons": risk.get("halt_reasons", [])}


def universe_sectors() -> dict[str, str]:
    t = pq.read_table(VAULT, columns=["date", "ticker", "sector"]).to_pandas().reset_index()
    return t.sort_values("date").drop_duplicates("ticker", keep="last").set_index("ticker")["sector"].to_dict()


def reacting_today(today: pd.Timestamp, trading_days: pd.DatetimeIndex, api_key: str) -> pd.DataFrame:
    """Calendar rows whose first reaction bar is today (bmo → report date, else next session)."""
    cal = fetch_earnings_calendar((today - pd.Timedelta(days=6)).strftime("%Y-%m-%d"),
                                  today.strftime("%Y-%m-%d"), api_key)
    days = trading_days.to_numpy()
    pos = np.where(cal["time"] == "bmo", np.searchsorted(days, cal["date"].to_numpy(), side="left"),
                   np.searchsorted(days, cal["date"].to_numpy(), side="right"))
    cal = cal[pos < len(days)]
    return cal[days[pos[pos < len(days)]] == today.to_datetime64()]


def surprise(ticker: str, report_date: pd.Timestamp, api_key: str) -> tuple[float, float]:
    """EPS and revenue surprise from FMP /stable/earnings — earnings-trader's source."""
    resp = requests.get(FMP_EARNINGS, params={"symbol": ticker, "limit": 5, "apikey": api_key}, timeout=15)
    resp.raise_for_status()
    for r in resp.json() or []:
        if r.get("date") == report_date.strftime("%Y-%m-%d") and r.get("epsActual") is not None:
            def pct(a, e):
                return (float(a) - float(e)) / abs(float(e)) if a is not None and e not in (None, 0) else np.nan
            return pct(r["epsActual"], r.get("epsEstimated")), pct(r.get("revenueActual"), r.get("revenueEstimated"))
    return np.nan, np.nan


def daily_bars(symbols: list[str], today: pd.Timestamp, executor: AlpacaExecutor) -> dict[str, pd.DataFrame]:
    """Completed daily bars from yfinance plus today's bar from Alpaca snapshots.

    yfinance's same-day row is unreliable in the first minutes of the session: on
    2026-09-29 at 9:31 its KMX row still carried 9/28's open (57.17 vs a 60.41 open),
    so the gap filter saw +1.1% instead of +6.8%. Today's bar comes from Alpaca's
    snapshot instead: dailyBar once it is dated today, else the last trade."""
    raw = yf.download(" ".join(symbols), period="3mo", interval="1d", auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    snaps = executor.snapshots(symbols)
    out = {}
    for s in symbols:
        try:
            df = (raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna(how="all")
        except KeyError:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        now = today_bar(snaps.get(s) or {}, today)
        if now is None or len(df[df.index < today]) <= ATR_PERIOD + 11:
            continue
        out[s] = pd.concat([df.loc[df.index < today, ["Open", "High", "Low", "Close"]],
                            pd.DataFrame([now], index=[today])])
    return out


def today_bar(snap: dict, today: pd.Timestamp) -> dict | None:
    """Today's OHLC from an Alpaca snapshot; Close is the last trade."""
    last = (snap.get("latestTrade") or {}).get("p")
    if not last:
        return None
    bar = snap.get("dailyBar") or {}
    if bar.get("t") and pd.Timestamp(bar["t"]).tz_convert(ET).date() == today.date():
        return {"Open": bar["o"], "High": max(bar["h"], last), "Low": min(bar["l"], last), "Close": last}
    return {"Open": last, "High": last, "Low": last, "Close": last}


def features(tickers: list[str], bars: dict[str, pd.DataFrame], sectors: dict[str, str]) -> pd.DataFrame:
    rows = {}
    for t in tickers:
        df = bars.get(t)
        if df is None:
            continue
        hist, now = df.iloc[:-1], df.iloc[-1]
        pc = hist["Close"].shift(1)
        tr = pd.concat([hist["High"] - hist["Low"], (hist["High"] - pc).abs(), (hist["Low"] - pc).abs()],
                       axis=1).max(axis=1)
        etf = bars.get(SECTOR_ETF_MAP.get(sectors.get(t, ""), FALLBACK_ETF))
        rows[t] = {
            "Open": now["Open"], "High": now["High"], "Low": now["Low"], "Close": now["Close"],
            "gap_1d": now["Open"] / hist["Close"].iloc[-1] - 1,
            "ret_1d": now["Close"] / hist["Close"].iloc[-1] - 1,
            "ret_10d": now["Close"] / hist["Close"].iloc[-10] - 1,
            "sector_ret_1d": (etf["Close"].iloc[-1] / etf["Close"].iloc[-2] - 1) if etf is not None else np.nan,
            "atr_14": float(tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean().iloc[-1]),
        }
    cols = ["Open", "High", "Low", "Close", "gap_1d", "ret_1d", "ret_10d", "sector_ret_1d", "atr_14"]
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)  # keep columns on no-candidate days


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="compute and log orders without placing them")
    parser.add_argument("--ignore-clock", action="store_true", help="skip the market-hours/once-a-day guard")
    args = parser.parse_args()

    now = datetime.now(ET)
    today = pd.Timestamp(now.date())
    executor = AlpacaExecutor(time_in_force="day", dry_run=args.dry_run)
    if not executor.is_paper and os.getenv("LIVE_TRADING_CONFIRMED", "").lower() != "yes":
        raise SystemExit("Refusing the real-money endpoint without LIVE_TRADING_CONFIRMED=yes")

    state = load_state()
    if not args.ignore_clock:
        if not (9 * 60 + 30 <= now.hour * 60 + now.minute < 10 * 60) or not executor.market_is_open():
            log.info("Outside the 9:30-10:00 ET window or market closed — nothing to do")
            return
        if state.get("last_run") == str(today.date()):
            log.info("Already ran today")
            return

    api_key = os.environ["FMP_API_KEY"]
    cal = executor._request("GET", "/v2/calendar", params={
        "start": (today - pd.Timedelta(days=12)).strftime("%Y-%m-%d"), "end": today.strftime("%Y-%m-%d")})
    trading_days = pd.DatetimeIndex(sorted(pd.to_datetime([d["date"] for d in cal])))
    if trading_days[-1] != today:
        log.info("%s is not a trading day", today.date())
        return

    # Reconcile strategy state against the broker (system of record)
    broker = executor.current_positions()
    notes = []
    for t in [t for t in state["positions"] if t not in broker]:
        notes.append(f"{t}: in state but not at broker — dropped")
        del state["positions"][t]
    for t in [t for t in broker if t not in state["positions"]]:
        notes.append(f"{t}: at broker but not in state — will be closed")

    equity = executor.account_equity()
    guard_dir = Path(os.environ.get("TMPDIR", "/tmp")) if args.dry_run else LIVE_DIR
    risk = RiskGuard(guard_dir / "pead_risk_state.json", LIVE_DIR / "HALT").evaluate(equity, str(today.date()))

    sectors = universe_sectors()
    rx = reacting_today(today, trading_days, api_key)
    rx = rx[rx["ticker"].isin(sectors)]
    held = list(state["positions"])
    etfs = sorted(set(SECTOR_ETF_MAP.values()) | {FALLBACK_ETF})
    bars = daily_bars(sorted(set(rx["ticker"]) | set(held) | set(broker) | set(etfs)), today, executor)
    feat = features(sorted(set(rx["ticker"]) | set(held)), bars, sectors)

    feat[["days_since_earnings", "earnings_surprise", "revenue_surprise"]] = np.nan
    for _, r in rx.iterrows():
        if r["ticker"] in feat.index:
            eps, rev = surprise(r["ticker"], r["date"], api_key)
            feat.loc[r["ticker"], ["days_since_earnings", "earnings_surprise", "revenue_surprise"]] = [0, eps, rev]
    log.info("Candidates reacting today: %d in universe, %d with data", len(rx), int((feat["days_since_earnings"] == 0).sum()))

    cols = ["earnings_surprise", "revenue_surprise", "gap_1d", "ret_10d", "ret_1d", "sector_ret_1d"]
    cand = feat[feat["days_since_earnings"] == 0].reindex(columns=cols)
    if not cand.empty:
        log.info("Candidate inputs:\n%s", cand.round(3).to_string())

    strategy = EarningsTraderPEAD()
    strategy.load_state({t: {"stop": p["stop"], "days": p["days"]} for t, p in state["positions"].items()})
    targets = strategy.predict(feat, bar_date=today)
    new = [t for t in targets if t not in state["positions"]]
    if new and not risk.entries_allowed:
        notes.append(f"entries blocked: {', '.join(new)}")
        for t in new:
            targets.pop(t)
        strategy.load_state({t: p for t, p in strategy.state().items() if t not in new})
        new = []

    # Held positions keep their shares: target their current weight so no order is sent
    prices = {t: float(bars[t]["Close"].iloc[-1]) for t in set(targets) | set(broker) if t in bars}
    for t in targets:
        if t in broker and t in prices:
            targets[t] = broker[t] * prices[t] / equity
    exits = [t for t in state["positions"] if t not in targets]
    executor.execute(targets, today.date(), prices)

    et = earnings_trader_day(today)
    kept = strategy.state()
    state["positions"] = {
        t: {**state["positions"].get(t, {"entry_date": str(today.date()), "entry_price": prices.get(t)}),
            "stop": kept[t]["stop"], "days": kept[t]["days"]}
        for t in targets}
    if not args.dry_run:
        state["last_run"] = str(today.date())
        save_state(state)
        with ORDERS_LOG.open("a") as f:
            for o in executor.orders():
                f.write(json.dumps({"date": str(today.date()), **{k: o.get(k) for k in ("symbol", "qty", "side", "price")},
                                    "order_id": (o.get("response") or {}).get("id")}) + "\n")
        with COMPARE_LOG.open("a") as f:
            f.write(json.dumps({
                "date": str(today.date()),
                "autoalpha": {"candidates": json.loads(cand.to_json(orient="index")), "entries": new,
                              "exits": exits, "holding": sorted(targets), "fills": [
                                  {k: o.get(k) for k in ("symbol", "qty", "side", "price")} for o in executor.orders()]},
                "earnings_trader": et}) + "\n")

    et_trades = ", ".join(f"{t['action']} {t['ticker']}" for t in et["trades"]) or "none"
    et_halt = f" | ⛔ halted since {et['halted_since'][:10]}: {'; '.join(et['halt_reasons'])}" if et["halted_since"] else ""
    lines = [f"**autoalpha live PEAD — {today.date()}**{' (dry run)' if args.dry_run else ''}", risk.line(),
             f"Candidates reacting today: {len(rx)} | entries: {', '.join(new) or 'none'} | "
             f"exits: {', '.join(exits) or 'none'}",
             f"Holding ({len(targets)}/{MAX_POSITIONS}): {', '.join(sorted(targets)) or 'none'}",
             f"earnings-trader holding: {', '.join(et['holding']) or 'none'} | "
             f"trades today: {et_trades}{et_halt}",
             *notes]
    msg = "\n".join(lines)
    log.info("\n%s", msg)
    if not args.dry_run and os.environ.get("DISCORD_BOT_TOKEN"):
        subprocess.run([sys.executable, "scripts/post_to_discord.py"], input=msg, text=True, check=False)


if __name__ == "__main__":
    main()
