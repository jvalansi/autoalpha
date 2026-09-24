"""Point-in-time quarterly fundamentals, keyed on SEC filing date.

The original build joined statements on the fiscal period-end `date`, so a quarter's
numbers appeared ~30-45 days before they were filed (look-ahead). Here every value
becomes visible on the first bar strictly after its filing date.

    FUND_COLS                                  dataset columns this module owns
    fetch_fundamentals(ticker, api_key)        -> DataFrame[period_date, filing_date, *FUND_COLS]
    pit_fundamentals(bars, table)              -> DataFrame[FUND_COLS] aligned to bars.index

Column definitions are unchanged from the build scripts:
  roe = netIncome / totalStockholdersEquity, net_margin = netIncome / revenue,
  debt_to_equity = netDebt / totalStockholdersEquity (quarterly statements);
  pe/pb/ps_ratio from /ratios; ev_ebitda, fcf_yield from /key-metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

FMP_BASE = "https://financialmodelingprep.com/stable"
FUND_COLS = ["roe", "net_margin", "debt_to_equity", "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda", "fcf_yield"]
FALLBACK_FILING_LAG = pd.Timedelta(days=90)  # ratios with no matching statement: 10-K deadline, conservative


def _get(session: requests.Session, endpoint: str, ticker: str, api_key: str) -> pd.DataFrame:
    resp = session.get(f"{FMP_BASE}/{endpoint}",
                       params={"symbol": ticker, "period": "quarter", "limit": 40, "apikey": api_key},
                       timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json() or [])


def fetch_fundamentals(ticker: str, api_key: str, session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests.Session()
    inc = _get(s, "income-statement", ticker, api_key)
    bal = _get(s, "balance-sheet-statement", ticker, api_key)
    rat = _get(s, "ratios", ticker, api_key)
    km = _get(s, "key-metrics", ticker, api_key)
    if inc.empty or "filingDate" not in inc.columns:
        return pd.DataFrame(columns=["period_date", "filing_date", *FUND_COLS])

    out = inc[["date", "filingDate", "netIncome", "revenue"]].rename(
        columns={"date": "period_date", "filingDate": "filing_date"})
    if not bal.empty:
        out = out.merge(bal[["date", "totalStockholdersEquity", "netDebt"]].rename(columns={"date": "period_date"}),
                        on="period_date", how="left")
    else:
        out[["totalStockholdersEquity", "netDebt"]] = np.nan
    if not rat.empty:
        rat = rat.reindex(columns=["date", "priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio"])
        rat.columns = ["period_date", "pe_ratio", "pb_ratio", "ps_ratio"]
        out = out.merge(rat, on="period_date", how="outer")
    if not km.empty:
        km = km.reindex(columns=["date", "evToEBITDA", "freeCashFlowYield"])
        km.columns = ["period_date", "ev_ebitda", "fcf_yield"]
        out = out.merge(km, on="period_date", how="outer")
    out = out.reindex(columns=[*out.columns, *[c for c in FUND_COLS if c not in out.columns]])

    for col in ["netIncome", "revenue", "totalStockholdersEquity", "netDebt", *FUND_COLS[3:]]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    equity = out["totalStockholdersEquity"].replace(0, np.nan)
    out["roe"] = out["netIncome"] / equity
    out["net_margin"] = out["netIncome"] / out["revenue"].replace(0, np.nan)
    out["debt_to_equity"] = out["netDebt"] / equity

    out["period_date"] = pd.to_datetime(out["period_date"])
    out["filing_date"] = pd.to_datetime(out["filing_date"]).fillna(out["period_date"] + FALLBACK_FILING_LAG)
    return out[["period_date", "filing_date", *FUND_COLS]].dropna(subset=["period_date"])


def pit_fundamentals(bars: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """For each (date, ticker) bar, the latest quarter whose filing date is strictly before the bar.

    bars: columns date, ticker. table: columns ticker, period_date, filing_date, *FUND_COLS.
    A late filing of an older quarter (e.g. an amendment) never replaces a newer quarter.
    """
    t = table.sort_values(["ticker", "filing_date", "period_date"])
    newest = t.groupby("ticker")["period_date"].cummax()
    t = t[t["period_date"] == newest].drop_duplicates(["ticker", "period_date"], keep="first")
    right = t[["ticker", "filing_date", *FUND_COLS]].rename(columns={"filing_date": "date"})
    right["date"] = right["date"].astype("datetime64[ns]")
    right = right.sort_values("date")

    left = pd.DataFrame({"date": pd.to_datetime(bars["date"]).to_numpy().astype("datetime64[ns]"),
                         "ticker": bars["ticker"].astype(str).to_numpy(),
                         "_row": np.arange(len(bars))}).sort_values("date")
    merged = pd.merge_asof(left, right, on="date", by="ticker", allow_exact_matches=False).sort_values("_row")
    return pd.DataFrame(merged[FUND_COLS].to_numpy(dtype=float), index=bars.index, columns=FUND_COLS)
