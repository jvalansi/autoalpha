"""Check that autoalpha's dataset columns reproduce earnings-trader's PEAD entry inputs.

For every earnings event in earnings-trader's backtest cache (calendar + surprise
already on disk, so no API calls), compute the entry inputs with earnings-trader's
own code — the same timing branch as src/backtest/runner.py, then
get_prior_runup_as_of / get_sector_move_on_date / evaluate_entry — on autoalpha's
bars, and compare with what a strategy sees in autoalpha's columns on the reaction bar.

Both sides use the same OHLCV, so disagreements are definition/timing/source bugs,
not vendor noise.

Usage:
    python scripts/check_pead_parity.py [--earnings-trader PATH] [--since 2022-01-01]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoalpha.data.event_features import SECTOR_ETF_MAP, FALLBACK_ETF, fetch_sector_closes

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

COLS = ["date", "ticker", "Open", "Close", "ret_1d", "ret_10d", "gap_1d", "sector_ret_1d",
        "days_since_earnings", "earnings_surprise", "revenue_surprise", "sector"]
TOL = 1e-6


def load_bars(since: pd.Timestamp, tickers: set[str]) -> dict[str, pd.DataFrame]:
    frames = []
    for path in ["data/loop_data.parquet", "data/vault_data.parquet"]:
        t = ds.dataset(path).to_table(
            columns=COLS,
            filter=(ds.field("date") >= since.to_pydatetime()) & ds.field("ticker").isin(sorted(tickers)),
        )
        frames.append(t.to_pandas().reset_index())
    df = pd.concat(frames, ignore_index=True)[COLS]
    df["date"] = pd.to_datetime(df["date"])
    return {t: g.set_index("date").sort_index() for t, g in df.groupby("ticker")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--earnings-trader", default="/home/ubuntu/earnings-trader")
    parser.add_argument("--since", default="2022-01-01")
    args = parser.parse_args()

    et_root = Path(args.earnings_trader)
    sys.path.insert(0, str(et_root / "src"))
    from backtest.data import CACHE_DIR, get_close_on_date, get_prior_runup_as_of, get_sector_move_on_date
    from config import MAX_PRIOR_RUNUP_PCT, MIN_AH_MOVE_PCT, MIN_EPS_BEAT_PCT, SECTOR_ETF_MIN
    from data.earnings import EarningsSurprise, _beat_pct
    from decision import evaluate_entry

    # Events earnings-trader has fully cached: calendar entry + surprise + sector ETF
    events = []
    for cal_path in sorted(CACHE_DIR.glob("earnings_v3_*.json")):
        date = cal_path.stem.removeprefix("earnings_v3_")
        if date < args.since:
            continue
        for rec in json.loads(cal_path.read_text()):
            t = rec.get("symbol", "")
            s_path = CACHE_DIR / f"surprise_{t}_{date}.json"
            e_path = CACHE_DIR / f"sector_{t}.json"
            if s_path.exists() and e_path.exists():
                events.append((t, date, (rec.get("time") or "").lower(), s_path, e_path))
    log.info("%d cached events since %s", len(events), args.since)

    since = pd.Timestamp(args.since) - pd.Timedelta(days=30)
    bars = load_bars(since, {e[0] for e in events})
    closes = fetch_sector_closes(since.strftime("%Y-%m-%d"), pd.Timestamp.today().strftime("%Y-%m-%d"))
    etf_dfs = {c: closes[[c]].rename(columns={c: "Close"}).dropna() for c in closes.columns}

    rows = []
    for ticker, date, timing, s_path, e_path in events:
        df = bars.get(ticker)
        if df is None:
            continue
        # --- earnings-trader side (timing branch copied from src/backtest/runner.py) ---
        try:
            if timing == "bmo":
                day_rows = df[df.index.strftime("%Y-%m-%d") == date]
                prev_rows = df[df.index.strftime("%Y-%m-%d") < date]
                if day_rows.empty or prev_rows.empty:
                    continue
                entry = day_rows.index[0]
                ah_move = float(day_rows["Open"].iloc[0]) / float(prev_rows["Close"].iloc[-1]) - 1.0
                signal_asof = prev_rows.index[-1].strftime("%Y-%m-%d")
            else:
                reg_close = get_close_on_date(df, date)
                next_rows = df[df.index.strftime("%Y-%m-%d") > date]
                if next_rows.empty:
                    continue
                entry = next_rows.index[0]
                ah_move = float(next_rows["Open"].iloc[0]) / reg_close - 1.0
                signal_asof = date
            runup = get_prior_runup_as_of(df, signal_asof)
            et_etf = json.loads(e_path.read_text()).get("etf", FALLBACK_ETF)
            sector_move = get_sector_move_on_date(et_etf, etf_dfs[et_etf], signal_asof)
        except Exception:
            continue  # earnings-trader's backtest skips these events too
        r = json.loads(s_path.read_text())
        eps_a, eps_e = float(r.get("epsActual") or 0.0), float(r.get("epsEstimated") or 0.0)
        rev_a, rev_e = float(r.get("revenueActual") or 0.0), float(r.get("revenueEstimated") or 0.0)
        surprise = EarningsSurprise(ticker=ticker, eps_actual=eps_a, eps_estimate=eps_e,
                                    eps_beat_pct=_beat_pct(eps_a, eps_e), rev_actual=rev_a,
                                    rev_estimate=rev_e, rev_beat_pct=_beat_pct(rev_a, rev_e),
                                    guidance_weak=None)
        et_enter = evaluate_entry(ticker=ticker, surprise=surprise, ah_move=ah_move, prior_runup=runup,
                                  sector_move=sector_move, atr=1.0, current_price=1.0,
                                  open_positions=[]).should_enter

        # --- autoalpha side: columns on the reaction bar (and the bar before it) ---
        row = df.loc[entry]
        prev = df.iloc[df.index.get_loc(entry) - 1]
        aa_runup = (1 + row["ret_10d"]) / (1 + row["ret_1d"]) - 1
        aa_etf = SECTOR_ETF_MAP.get(row["sector"], FALLBACK_ETF)
        aa_enter = bool(
            row["earnings_surprise"] >= MIN_EPS_BEAT_PCT and row["revenue_surprise"] > 0
            and row["gap_1d"] >= MIN_AH_MOVE_PCT and aa_runup <= MAX_PRIOR_RUNUP_PCT
            and prev["sector_ret_1d"] > SECTOR_ETF_MIN
        )
        rows.append({
            "ticker": ticker, "date": date, "timing": timing or "amc",
            "reaction_day": row["days_since_earnings"] == 0,
            "gap": abs(row["gap_1d"] - ah_move) < TOL,
            "runup": abs(aa_runup - runup) < TOL,
            "sector_etf": aa_etf == et_etf,
            "sector_move": abs(prev["sector_ret_1d"] - sector_move) < TOL,
            "eps_surprise": abs(row["earnings_surprise"] - surprise.eps_beat_pct) < 1e-4,
            "rev_surprise": abs(row["revenue_surprise"] - surprise.rev_beat_pct) < 1e-4,
            "et_enter": et_enter, "aa_enter": aa_enter,
        })

    res = pd.DataFrame(rows)
    checks = ["reaction_day", "gap", "runup", "sector_etf", "sector_move", "eps_surprise", "rev_surprise"]
    res["period"] = np.where(res["date"] < "2024-05-21", "loop", "vault")
    print(f"\n{len(res)} comparable events")
    print("\nShare of events where autoalpha matches earnings-trader:")
    print(res.groupby("period")[checks].mean().round(3).T.to_string())
    print("\nEntry decision (rows: earnings-trader, cols: autoalpha):")
    print(pd.crosstab(res["et_enter"], res["aa_enter"]).to_string())
    bad = res[res["et_enter"] != res["aa_enter"]]
    if len(bad):
        print("\nSample decision mismatches:")
        print(bad.head(15)[["ticker", "date", "timing", *checks]].to_string(index=False))


if __name__ == "__main__":
    main()
