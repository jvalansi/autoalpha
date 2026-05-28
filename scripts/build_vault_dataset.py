"""Build feature-enriched vault dataset for holdout evaluation.

Same schema as loop_data.parquet but covering the vault period
(2024-05-21 → today). Full history is loaded so 252-day lookbacks
near the vault start are accurate.

Output: data/vault_data.parquet
"""
from __future__ import annotations

import io
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

CACHE_ROOT = Path("data/cache")
OUT_PATH = Path("data/vault_data.parquet")
VAULT_START = pd.Timestamp("2024-05-21")
VAULT_END = pd.Timestamp.today().normalize()
TICKERS = sorted(d.name for d in CACHE_ROOT.iterdir() if d.is_dir())

LOOKBACKS = {"ret_1d": 1, "ret_5d": 5, "ret_21d": 21, "ret_63d": 63, "ret_252d": 252}
NAN_COLS = ["sentiment_score"]


def load_ohlcv_full(ticker: str) -> pd.DataFrame:
    frames = []
    for year_file in sorted((CACHE_ROOT / ticker).glob("[0-9]*.parquet")):
        try:
            frames.append(pd.read_parquet(year_file))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[df.index <= VAULT_END]
    df.index.name = "date"
    return df


def compute_returns(prices: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({col: prices.pct_change(n) for col, n in LOOKBACKS.items()},
                        index=prices.index)


def compute_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    close = ohlcv["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_14 = 100 - (100 / (1 + rs))
    high_252 = close.rolling(252, min_periods=63).max()
    pct_from_52w_high = (close - high_252) / high_252
    vol_21d = close.pct_change().rolling(21).std() * np.sqrt(252)
    return pd.DataFrame({"rsi_14": rsi_14, "pct_from_52w_high": pct_from_52w_high, "vol_21d": vol_21d},
                        index=close.index)


def load_fundamentals(ticker: str) -> pd.DataFrame:
    path = CACHE_ROOT / ticker / "fmp_fundamentals.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    equity = df["totalStockholdersEquity"].replace(0, np.nan)
    revenue = df["revenue"].replace(0, np.nan)
    df["roe"] = df["netIncome"] / equity
    df["net_margin"] = df["netIncome"] / revenue
    df["debt_to_equity"] = df["netDebt"] / equity
    return df[["date", "roe", "net_margin", "debt_to_equity"]].dropna(subset=["date"])


def load_earnings(ticker: str) -> pd.DataFrame:
    path = CACHE_ROOT / ticker / "fmp_earnings.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    eps_est = df["epsEstimated"].replace(0, np.nan).abs()
    rev_est = df["revenueEstimated"].replace(0, np.nan).abs()
    df["earnings_surprise"] = (df["epsActual"] - df["epsEstimated"]) / eps_est
    df["revenue_surprise"] = (df["revenueActual"] - df["revenueEstimated"]) / rev_est
    return df[["date", "earnings_surprise", "revenue_surprise"]].dropna(subset=["date"])


def load_valuation(ticker: str) -> pd.DataFrame:
    path = CACHE_ROOT / ticker / "fmp_valuation.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    keep = [c for c in ["date", "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda"] if c in df.columns]
    return df[keep]


def load_estimates(ticker: str) -> pd.DataFrame:
    path = CACHE_ROOT / ticker / "fmp_estimates.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" not in df.columns or "epsAvg" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    eps = df["epsAvg"].replace(0, np.nan).abs()
    df["analyst_revision_3m"] = (df["epsAvg"] - df["epsAvg"].shift(1)) / eps.shift(1)
    return df[["date", "analyst_revision_3m"]].dropna(subset=["date"])


def load_key_metrics(ticker: str) -> pd.DataFrame:
    path = CACHE_ROOT / ticker / "fmp_key_metrics.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    keep = [c for c in ["date", "dividend_yield", "fcf_yield"] if c in df.columns]
    return df[keep]


def load_sector(ticker: str) -> str:
    path = CACHE_ROOT / ticker / "fmp_sector.json"
    if not path.exists():
        return "Unknown"
    try:
        return json.loads(path.read_text())["sector"]
    except Exception:
        return "Unknown"


def fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), parse_dates=["observation_date"])
        df = df.rename(columns={"observation_date": "date", series_id: series_id})
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date")[series_id]
        df = df.replace(".", np.nan).astype(float)
        return df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]
    except Exception as exc:
        log.warning("FRED %s fetch failed: %s", series_id, exc)
        return pd.Series(dtype=float, name=series_id)


def fetch_macro(start: str, end: str) -> pd.DataFrame:
    log.info("Fetching FRED macro series...")
    dgs10 = fetch_fred_series("DGS10", start, end).rename("yield_10y")
    dgs2 = fetch_fred_series("DGS2", start, end).rename("yield_2y")
    spread = fetch_fred_series("BAA10Y", start, end).rename("credit_spread")
    macro = pd.concat([dgs10, dgs2, spread], axis=1).ffill()
    macro["yield_curve"] = macro["yield_10y"] - macro["yield_2y"]
    return macro


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
    if df.empty:
        return pd.DataFrame(index=daily_index)
    df = df.set_index(date_col)
    df = df[~df.index.duplicated(keep="last")]
    return df.reindex(daily_index, method="ffill")


def build_ticker(ticker: str, vix: pd.Series, macro: pd.DataFrame) -> pd.DataFrame:
    full = load_ohlcv_full(ticker)
    if full.empty or len(full) < 300:
        log.warning("Skipping %s — insufficient OHLCV data", ticker)
        return pd.DataFrame()

    full_idx = full.index

    ret = compute_returns(full["Close"])
    tech = compute_technical_features(full)
    fund = point_in_time_join(full_idx, load_fundamentals(ticker), "date")
    earn = point_in_time_join(full_idx, load_earnings(ticker), "date")
    val = point_in_time_join(full_idx, load_valuation(ticker), "date")
    est = point_in_time_join(full_idx, load_estimates(ticker), "date")
    km = point_in_time_join(full_idx, load_key_metrics(ticker), "date")
    vix_aligned = vix.reindex(full_idx, method="ffill")
    macro_aligned = macro.reindex(full_idx, method="ffill")
    sector = load_sector(ticker)

    df = pd.concat([full, ret, tech, fund, earn, val, est, km, vix_aligned, macro_aligned], axis=1)
    df["sector"] = sector
    for col in NAN_COLS:
        df[col] = np.nan

    # Trim to vault period only
    df = df[df.index >= VAULT_START]
    if df.empty:
        return pd.DataFrame()

    df.index = pd.MultiIndex.from_product([[ticker], df.index], names=["ticker", "date"])
    return df


def main() -> None:
    log.info("Building vault dataset: %s → %s", VAULT_START.date(), VAULT_END.date())
    log.info("Tickers: %s", TICKERS)

    start = "2018-01-01"
    end = (VAULT_END + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    vix = fetch_vix(start, end)
    vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()
    macro = fetch_macro(start, end)
    macro.index = pd.to_datetime(macro.index).tz_localize(None).normalize()

    frames = []
    for ticker in TICKERS:
        log.info("Processing %s...", ticker)
        df = build_ticker(ticker, vix, macro)
        if not df.empty:
            frames.append(df)

    if not frames:
        log.error("No vault data built — aborting")
        return

    combined = pd.concat(frames).sort_index()
    combined = combined.swaplevel().sort_index()
    combined.index.names = ["date", "ticker"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH)
    log.info("Saved %s  shape=%s  date range=%s → %s  columns=%s",
             OUT_PATH, combined.shape,
             combined.index.get_level_values("date").min().date(),
             combined.index.get_level_values("date").max().date(),
             list(combined.columns))


if __name__ == "__main__":
    main()
