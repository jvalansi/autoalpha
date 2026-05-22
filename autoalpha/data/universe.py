"""Survivorship-bias-free S&P 500 universe via Sharadar (Nasdaq Data Link).

Point-in-time join: for backtest date t, returns tickers where:
    date_added <= t  AND  (date_removed IS NULL OR date_removed > t)

Uses the effective index entry date (not announcement date) to avoid
trading on pre-announcement information.

Requires NASDAQ_DATA_LINK_API_KEY env var.
"""
from __future__ import annotations

import logging
import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import nasdaqdatalink
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_PATH = Path("data/cache/sharadar_sp500.parquet")


@lru_cache(maxsize=1)
def _load_sharadar() -> pd.DataFrame:
    """Load Sharadar S&P 500 constituent history, cached to disk."""
    api_key = os.environ.get("NASDAQ_DATA_LINK_API_KEY", "")
    if not api_key:
        raise EnvironmentError("NASDAQ_DATA_LINK_API_KEY not set")

    if _CACHE_PATH.exists():
        df = pd.read_parquet(_CACHE_PATH)
        logger.info("Loaded Sharadar universe from cache (%d rows)", len(df))
        return df

    nasdaqdatalink.ApiConfig.api_key = api_key
    df = nasdaqdatalink.get_table("SHARADAR/SP500", paginate=True)

    # Normalize columns (Sharadar returns: ticker, date, action)
    df.columns = [c.lower() for c in df.columns]
    # Reconstruct entry/removal from action column
    # actions: 'added', 'removed'
    if "action" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values(["ticker", "date"])
        # Pair each 'added' event with the next 'removed' event for that ticker.
        # A ticker can cycle in/out of the index multiple times; simple merge
        # produces a Cartesian product in that case.
        rows = []
        for ticker, grp in df.groupby("ticker"):
            adds = grp[grp["action"] == "added"]["date"].tolist()
            removals = grp[grp["action"] == "removed"]["date"].tolist()
            rem_iter = iter(removals)
            next_removal = next(rem_iter, None)
            for add_date in adds:
                while next_removal is not None and next_removal <= add_date:
                    next_removal = next(rem_iter, None)
                rows.append({
                    "ticker": ticker,
                    "date_added": add_date,
                    "date_removed": next_removal,
                })
                if next_removal is not None:
                    next_removal = next(rem_iter, None)
        df = pd.DataFrame(rows)
    else:
        df["date_added"] = pd.to_datetime(df.get("date_added", df.get("dateadded"))).dt.normalize()
        df["date_removed"] = pd.to_datetime(df.get("date_removed", df.get("dateremoved"))).dt.normalize()
        df = df[["ticker", "date_added", "date_removed"]]

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CACHE_PATH)
    logger.info("Cached Sharadar universe (%d rows)", len(df))
    return df


def get_sp500_universe(as_of: date) -> list[str]:
    """Return list of S&P 500 tickers that were members as of the given date.

    Applies strict point-in-time filter using effective entry/removal dates.
    """
    df = _load_sharadar()
    ts = pd.Timestamp(as_of)
    mask = df["date_added"] <= ts
    not_removed = df["date_removed"].isna() | (df["date_removed"] > ts)
    members = df[mask & not_removed]["ticker"].unique().tolist()
    logger.debug("Universe as of %s: %d tickers", as_of, len(members))
    return members


def get_universe_history(start: date, end: date, freq: str = "ME") -> dict[date, list[str]]:
    """Return a dict mapping rebalance dates to their point-in-time S&P 500 universe.

    freq: pandas offset string for rebalance frequency (default: 'ME' = month-end).
    """
    rebalance_dates = pd.date_range(start=start, end=end, freq=freq)
    return {
        d.date(): get_sp500_universe(d.date())
        for d in rebalance_dates
    }
