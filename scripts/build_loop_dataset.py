"""Build a feature-enriched MultiIndex (date, ticker) parquet for the Phase 4 loop.

Columns included:
  OHLCV                — from per-year cache
  ret_1d/5d/21d/63d/252d — computed from Close
  roe, net_margin      — from FMP fundamentals (forward-filled quarterly)
  earnings_surprise, revenue_surprise — from FMP earnings (forward-filled 63d)
  vix                  — from yfinance ^VIX (same for all tickers)
  pe_ratio, pb_ratio, ps_ratio, ev_ebitda,
  analyst_revision_3m, yield_10y, yield_2y,
  credit_spread, sentiment_score  — NaN (not yet in pipeline)

Output: data/loop_data.parquet
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

CACHE_ROOT = Path("data/cache")
OUT_PATH = Path("data/loop_data.parquet")
VAULT_START = pd.Timestamp("2024-05-21")
TICKERS = sorted(d.name for d in CACHE_ROOT.iterdir() if d.is_dir())

LOOKBACKS = {"ret_1d": 1, "ret_5d": 5, "ret_21d": 21, "ret_63d": 63, "ret_252d": 252}
NAN_COLS = [
    "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
    "analyst_revision_3m", "yield_10y", "yield_2y",
    "credit_spread", "sentiment_score",
]


def load_ohlcv(ticker: str) -> pd.DataFrame:
    frames = []
    for year_file in sorted((CACHE_ROOT / ticker).glob("[0-9]*.parquet")):
        try:
            frames.append(pd.read_parquet(year_file))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[df.index < VAULT_START]
    df.index.name = "date"
    return df


def compute_returns(prices: pd.Series) -> pd.DataFrame:
    out = {}
    for col, n in LOOKBACKS.items():
        out[col] = prices.pct_change(n)
    return pd.DataFrame(out, index=prices.index)


def load_fundamentals(ticker: str) -> pd.DataFrame:
    path = CACHE_ROOT / ticker / "fmp_fundamentals.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] < VAULT_START].sort_values("date")

    equity = df["totalStockholdersEquity"].replace(0, np.nan)
    revenue = df["revenue"].replace(0, np.nan)
    df["roe"] = df["netIncome"] / equity
    df["net_margin"] = df["netIncome"] / revenue
    return df[["date", "roe", "net_margin"]].dropna()


def load_earnings(ticker: str) -> pd.DataFrame:
    path = CACHE_ROOT / ticker / "fmp_earnings.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] < VAULT_START].sort_values("date")

    eps_est = df["epsEstimated"].replace(0, np.nan).abs()
    rev_est = df["revenueEstimated"].replace(0, np.nan).abs()
    df["earnings_surprise"] = (df["epsActual"] - df["epsEstimated"]) / eps_est
    df["revenue_surprise"] = (df["revenueActual"] - df["revenueEstimated"]) / rev_est
    return df[["date", "earnings_surprise", "revenue_surprise"]].dropna(subset=["date"])


def fetch_vix(start: str, end: str) -> pd.Series:
    log.info("Fetching VIX...")
    try:
        vix = yf.Ticker("^VIX").history(start=start, end=end, auto_adjust=True)
        s = vix["Close"].rename("vix")
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s
    except Exception as exc:
        log.warning("VIX fetch failed: %s", exc)
        return pd.Series(dtype=float, name="vix")


def point_in_time_join(daily_index: pd.DatetimeIndex, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Forward-fill quarterly/event data onto a daily trading calendar."""
    if df.empty:
        return pd.DataFrame(index=daily_index)
    df = df.set_index(date_col).reindex(daily_index, method="ffill")
    return df


def build_ticker(ticker: str, vix: pd.Series) -> pd.DataFrame:
    ohlcv = load_ohlcv(ticker)
    if ohlcv.empty or len(ohlcv) < 300:
        log.warning("Skipping %s — insufficient OHLCV data (%d rows)", ticker, len(ohlcv))
        return pd.DataFrame()

    idx = ohlcv.index

    ret = compute_returns(ohlcv["Close"])
    fund = point_in_time_join(idx, load_fundamentals(ticker), "date")
    earn = point_in_time_join(idx, load_earnings(ticker), "date")

    vix_aligned = vix.reindex(idx, method="ffill")

    df = pd.concat([ohlcv, ret, fund, earn, vix_aligned], axis=1)
    for col in NAN_COLS:
        df[col] = np.nan

    df.index = pd.MultiIndex.from_product([[ticker], df.index], names=["ticker", "date"])
    return df


def main() -> None:
    log.info("Building loop dataset for tickers: %s", TICKERS)

    start = "2018-01-01"
    end = VAULT_START.strftime("%Y-%m-%d")
    vix = fetch_vix(start, end)
    vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()

    frames = []
    for ticker in TICKERS:
        log.info("Processing %s...", ticker)
        df = build_ticker(ticker, vix)
        if not df.empty:
            frames.append(df)

    if not frames:
        log.error("No data built — aborting")
        return

    combined = pd.concat(frames).sort_index()
    # Reorder index to (date, ticker) as the harness expects
    combined = combined.swaplevel().sort_index()
    combined.index.names = ["date", "ticker"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH)
    log.info("Saved %s  shape=%s  date range=%s to %s  columns=%s",
             OUT_PATH, combined.shape,
             combined.index.get_level_values("date").min().date(),
             combined.index.get_level_values("date").max().date(),
             list(combined.columns))


if __name__ == "__main__":
    main()
