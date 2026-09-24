"""Earnings-event features mirroring earnings-trader's PEAD entry filters.

    EVENT_COLS                                   new dataset columns, in canonical order
    fetch_earnings_calendar(start, end, key)     -> DataFrame[ticker, date, time, eps, epsEstimated, revenue,
                                                              revenueEstimated]  (FMP v3, has amc/bmo)
    latest_surprise(bars, calendar)              -> DataFrame[earnings_surprise, revenue_surprise] aligned to bars
    fetch_sector_closes(start, end)              -> DataFrame(index=date, columns=ETF symbols)
    event_features(bars, calendar, sector_closes)-> DataFrame[EVENT_COLS] aligned to bars.index

Column definitions (all known at the bar's close):
  gap_1d               Open / previous Close - 1 (earnings-trader's after-hours proxy)
  ret_10d              Close / Close 10 bars ago - 1
  sector_ret_1d        sector SPDR ETF close-to-close return on this date (SPY if sector unknown)
  days_since_earnings  bars since the first bar that could react to the latest report:
                       the report date for 'bmo', the next bar otherwise (amc / unknown time),
                       matching earnings-trader's backtest. 0 = reaction bar, NaN = none seen yet.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger(__name__)

EVENT_COLS = ["gap_1d", "ret_10d", "sector_ret_1d", "days_since_earnings"]

# Same mapping as earnings-trader/src/data/sector.py
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
}
FALLBACK_ETF = "SPY"

_CALENDAR_URL = "https://financialmodelingprep.com/api/v3/earning_calendar"
_WINDOW_DAYS = 90  # the endpoint accepts at most ~3 months per request
_CAL_COLS = ["symbol", "date", "time", "eps", "epsEstimated", "revenue", "revenueEstimated"]


def fetch_earnings_calendar(start: str, end: str, api_key: str) -> pd.DataFrame:
    """Return every FMP earnings-calendar entry between start and end (inclusive)."""
    frames = []
    lo = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    while lo <= stop:
        hi = min(lo + pd.Timedelta(days=_WINDOW_DAYS - 1), stop)
        resp = requests.get(
            _CALENDAR_URL,
            params={"from": lo.strftime("%Y-%m-%d"), "to": hi.strftime("%Y-%m-%d"), "apikey": api_key},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError(f"FMP earnings calendar error: {data}")
        if data:
            frames.append(pd.DataFrame(data).reindex(columns=_CAL_COLS))
        log.info("Earnings calendar %s → %s: %d rows", lo.date(), hi.date(), len(data))
        lo = hi + pd.Timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["ticker", *_CAL_COLS[1:]])
    cal = pd.concat(frames).rename(columns={"symbol": "ticker"})
    cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()
    cal["time"] = cal["time"].fillna("").str.lower()
    for col in _CAL_COLS[3:]:
        cal[col] = pd.to_numeric(cal[col], errors="coerce")
    return cal.drop_duplicates(["ticker", "date"]).reset_index(drop=True)


def latest_surprise(bars: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Surprise of the latest reported quarter on or before each bar's date (NaN if none).

    Same formula as the build scripts: (actual - estimate) / |estimate|, NaN when estimate is 0.
    Keyed on the report date, like the build scripts' forward-fill.
    """
    cal = calendar.dropna(subset=["eps"]).copy()
    cal["earnings_surprise"] = (cal["eps"] - cal["epsEstimated"]) / cal["epsEstimated"].replace(0, np.nan).abs()
    cal["revenue_surprise"] = (cal["revenue"] - cal["revenueEstimated"]) / cal["revenueEstimated"].replace(0, np.nan).abs()
    left = pd.DataFrame({"date": pd.to_datetime(bars["date"]).to_numpy().astype("datetime64[ns]"),
                         "ticker": bars["ticker"].astype(str).to_numpy(),
                         "_row": np.arange(len(bars))}).sort_values("date")
    right = cal[["date", "ticker", "earnings_surprise", "revenue_surprise"]].sort_values("date")
    right["date"] = right["date"].astype("datetime64[ns]")
    merged = pd.merge_asof(left, right, on="date", by="ticker").sort_values("_row")
    return pd.DataFrame(merged[["earnings_surprise", "revenue_surprise"]].to_numpy(),
                        index=bars.index, columns=["earnings_surprise", "revenue_surprise"])


def fetch_sector_closes(start: str, end: str) -> pd.DataFrame:
    """Adjusted daily closes for the sector ETFs plus the SPY fallback."""
    etfs = sorted(set(SECTOR_ETF_MAP.values()) | {FALLBACK_ETF})
    raw = yf.download(" ".join(etfs), start=start, end=end, auto_adjust=True, progress=False)
    closes = raw["Close"]
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    return closes


def event_features(bars: pd.DataFrame, calendar: pd.DataFrame, sector_closes: pd.DataFrame) -> pd.DataFrame:
    """Compute EVENT_COLS for long-format bars.

    bars: columns date, ticker, Open, Close, sector (any order, any index; one row per ticker-date).
    Returns a DataFrame indexed like `bars`.
    """
    df = bars[["date", "ticker", "Open", "Close", "sector"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"], kind="stable")
    by_ticker = df.groupby("ticker", sort=False)["Close"]

    out = pd.DataFrame(index=df.index)
    out["gap_1d"] = df["Open"] / by_ticker.shift(1) - 1
    out["ret_10d"] = df["Close"] / by_ticker.shift(10) - 1

    etf = df["sector"].map(SECTOR_ETF_MAP).fillna(FALLBACK_ETF)
    # fill_method=None: a missing ETF bar must not borrow the previous day's move
    etf_ret = sector_closes.pct_change(fill_method=None).stack().rename("r")
    out["sector_ret_1d"] = etf_ret.reindex(pd.MultiIndex.from_arrays([df["date"], etf])).to_numpy()

    out["days_since_earnings"] = _days_since_earnings(df, calendar)
    return out.reindex(bars.index)


def _days_since_earnings(df: pd.DataFrame, calendar: pd.DataFrame) -> np.ndarray:
    """df sorted by (ticker, date). Returns values aligned to df's row order."""
    bar_no = df.groupby("ticker", sort=False).cumcount().to_numpy()
    reaction = np.zeros(len(df), dtype=bool)

    cal = calendar[calendar["ticker"].isin(df["ticker"].unique())]
    cal_by_ticker = dict(tuple(cal.groupby("ticker")))
    start = 0
    for ticker, n in df.groupby("ticker", sort=False).size().items():
        rows = cal_by_ticker.get(ticker)
        if rows is not None:
            dates = df["date"].to_numpy()[start:start + n]
            rows = rows[rows["date"].to_numpy() >= dates[0]]  # reaction bar of earlier reports is unknowable
            bmo = rows["time"].to_numpy() == "bmo"
            ev = rows["date"].to_numpy()
            # bmo reacts on the report date; amc/unknown on the next bar
            pos = np.where(bmo, np.searchsorted(dates, ev, side="left"),
                           np.searchsorted(dates, ev, side="right"))
            pos = pos[pos < n]
            reaction[start + pos] = True
        start += n

    last = pd.Series(np.where(reaction, bar_no, np.nan), index=df.index)
    last = last.groupby(df["ticker"].to_numpy(), sort=False).ffill().to_numpy()
    return bar_no - last
