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
  - Universe: every actively traded NYSE/NASDAQ/AMEX stock (FMP screener, no ETFs or
    funds), like earnings-trader's scan; the backtest universe (data/vault_data.parquet)
    missed names it traded (WASH 2026-10-20, SPCX 2026-10-21).
  - Held positions keep their share count (no daily rebalance), like earnings-trader.

Usage:
    python scripts/run_live_pead.py [--dry-run] [--ignore-clock]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
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
COMPARE_LOG = LIVE_DIR / "pead_compare.jsonl"
UNIVERSE_CACHE = LIVE_DIR / "universe_sectors.json"
ET_DATA = Path("/home/ubuntu/earnings-trader/data")
FMP_EARNINGS = "https://financialmodelingprep.com/stable/earnings"
FMP_SCREENER = "https://financialmodelingprep.com/stable/company-screener"
FMP_QUOTES = "https://financialmodelingprep.com/stable/batch-quote"


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


def fmp_get(url: str, params: dict, timeout: float = 30, tries: int = 3) -> list:
    """GET an FMP endpoint that returns a JSON list, retrying 5xx/429/network errors.

    Errors name the endpoint only — requests' own messages include the apikey."""
    for attempt in range(tries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            err = f"{type(e).__name__}"
        else:
            if resp.status_code < 400:
                body = resp.json()
                if isinstance(body, list):
                    return body
                raise RuntimeError(f"FMP {url}: unexpected response {str(body)[:200]}")
            err = f"HTTP {resp.status_code}"
            if resp.status_code < 500 and resp.status_code != 429:
                break
        if attempt + 1 < tries:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"FMP {url}: {err}")


def redact(e: Exception) -> str:
    """Exception text safe for logs/Discord — requests errors embed the FMP apikey."""
    return re.sub(r"apikey=[^&\s]+", "apikey=***", str(e))


def universe_sectors(api_key: str) -> tuple[dict[str, str], str | None]:
    """Ticker -> sector for actively traded US-listed stocks (no ETFs/funds), plus a
    note when the last good screener result had to be used instead."""
    limit = 20000
    try:
        rows = fmp_get(FMP_SCREENER, timeout=60, params={
            "exchange": "NYSE,NASDAQ,AMEX", "isEtf": "false", "isFund": "false",
            "isActivelyTrading": "true", "limit": limit, "apikey": api_key})
        sectors = {r["symbol"]: r.get("sector") or "" for r in rows}
        if not 1000 <= len(sectors) < limit:  # truncated or capped → universe silently wrong
            raise RuntimeError(f"FMP screener returned {len(sectors)} stocks")
    except Exception as e:
        if not UNIVERSE_CACHE.exists():
            raise
        age = (datetime.now().timestamp() - UNIVERSE_CACHE.stat().st_mtime) / 86400
        return json.loads(UNIVERSE_CACHE.read_text()), f"screener failed ({redact(e)}) — using {age:.0f}-day-old universe"
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = UNIVERSE_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(sectors))
    tmp.rename(UNIVERSE_CACHE)
    return sectors, None


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


def fmp_quotes(symbols: list[str], api_key: str) -> dict[str, dict]:
    out = {}
    for i in range(0, len(symbols), 100):
        out.update({q["symbol"]: q for q in fmp_get(FMP_QUOTES, {"symbols": ",".join(symbols[i:i + 100]),
                                                                 "apikey": api_key}) or []})
    return out


def daily_bars(symbols: list[str], today: pd.Timestamp, quotes: dict[str, dict],
               price_only: set[str] = frozenset()) -> dict[str, pd.DataFrame]:
    """Completed daily bars from yfinance plus today's bar from FMP's consolidated quote.

    yfinance's same-day row can still carry the previous session's open at 9:31
    (2026-09-29 KMX: +1.1% gap read vs +6.8% real), and Alpaca's free IEX snapshot can
    have no trade yet for thinner names (2026-10-22 LNN: -0.3% read vs +4.9% real).
    History is dividend-adjusted (as in the backtest and earnings-trader); the quote is
    checked against the unadjusted closes, which is what FMP's previousClose reports.

    `price_only` symbols (held positions) whose quote is on today's session but has no
    usable open still get a bar at the last price, so stops and the 10-day exit run."""
    raw = yf.download(" ".join(symbols), period="3mo", interval="1d", auto_adjust=False,
                      progress=False, group_by="ticker", threads=True)
    out = {}
    for s in symbols:
        try:
            df = (raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna(how="all")
        except KeyError:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[df.index < today].dropna(subset=["Open", "High", "Low", "Close", "Adj Close"])
        if len(df) <= ATR_PERIOD + 11:
            continue
        q = quotes.get(s)
        now = today_bar(q, df["Close"].iloc[-2:].tolist(), float(df["Open"].iloc[-1]), today)
        if now is None and s in price_only and on_today_session(q, df["Close"].iloc[-2:].tolist(),
                                                                 float(df["Open"].iloc[-1]), today):
            now = dict.fromkeys(["Open", "High", "Low", "Close"], float(q["price"]))
        if now is None:
            log.warning("%s: no usable quote for today's session — skipped", s)
            continue
        hist = df[["Open", "High", "Low", "Close"]].mul(df["Adj Close"] / df["Close"], axis=0)
        out[s] = pd.concat([hist, pd.DataFrame([now], index=[today])])
    return out


def on_today_session(quote: dict | None, last_closes: list[float], last_open: float,
                     today: pd.Timestamp) -> bool:
    """Is this FMP quote from today's session?

    `last_closes` = unadjusted [day-before-yesterday's, yesterday's] close, `last_open` =
    yesterday's open. A quote still on yesterday's session is stamped today at 9:31 but
    reports the day before as previousClose and yesterday's open as its open."""
    if not quote or not quote.get("price") or not quote.get("previousClose") or not quote.get("timestamp"):
        return False
    if pd.Timestamp(quote["timestamp"], unit="s", tz="UTC").tz_convert(ET).date() != today.date():
        return False
    pc, (c2, c1) = float(quote["previousClose"]), last_closes
    if c1 != c2:
        return abs(pc - c1) < abs(pc - c2)
    return not quote.get("open") or float(quote["open"]) != last_open  # flat two days: fall back on the open


def today_bar(quote: dict | None, last_closes: list[float], last_open: float,
              today: pd.Timestamp) -> dict | None:
    """Today's OHLC from an FMP quote (Close = last price), or None if it isn't usable."""
    if not on_today_session(quote, last_closes, last_open, today) or not quote.get("open"):
        return None
    if abs(float(quote["previousClose"]) / last_closes[-1] - 1) > 0.05:  # bad print / corporate action
        return None
    price, open_ = float(quote["price"]), float(quote["open"])
    return {"Open": open_, "High": max(float(quote.get("dayHigh") or price), price, open_),
            "Low": min(float(quote.get("dayLow") or price), price, open_), "Close": price}


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

    # FMP failures cost the day's entries, never the exits on what's already held
    try:
        sectors, note = universe_sectors(api_key)
        notes += [note] if note else []
        rx = reacting_today(today, trading_days, api_key)
        rx = rx[rx["ticker"].isin(sectors)]
    except Exception as e:
        notes.append(f"no entries today — earnings scan failed: {redact(e)}")
        sectors, rx = {}, pd.DataFrame(columns=["ticker", "date", "time"])
    held = list(state["positions"])
    etfs = sorted(set(SECTOR_ETF_MAP.values()) | {FALLBACK_ETF})
    symbols = sorted(set(rx["ticker"]) | set(held) | set(broker) | set(etfs))
    try:
        quotes = fmp_quotes(symbols, api_key)
    except Exception as e:
        notes.append(f"FMP quotes failed: {redact(e)}")
        quotes = {}
    bars = daily_bars(symbols, today, quotes, price_only=set(held))
    notes += [f"{t}: no price today — stop and 10-day exit not checked" for t in held if t not in bars]
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
    # No candidates and nothing held → empty frame without bar columns (crashed 2026-10-05)
    targets = {} if feat.empty and not state["positions"] else strategy.predict(feat, bar_date=today)
    new = [t for t in targets if t not in state["positions"]]
    if new and not risk.entries_allowed:
        notes.append(f"entries blocked: {', '.join(new)}")
        for t in new:
            targets.pop(t)
        strategy.load_state({t: p for t, p in strategy.state().items() if t not in new})
        new = []

    # Held positions keep their shares: the executor sends no order for them. Re-sizing
    # through a weight resized them whenever equity moved between reads (2026-10-12 sold
    # 1 LW and 1 NEOG, 2026-10-20 bought 1 NEOG).
    prices = {t: float(bars[t]["Close"].iloc[-1]) for t in set(targets) | set(broker) if t in bars}
    exits = [t for t in state["positions"] if t not in targets]
    executor.execute(targets, today.date(), prices, hold=frozenset(t for t in targets if t in state["positions"]))

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
