"""DataProvider abstractions: HistoricalProvider and LiveProvider."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Yields bars as (bar_date, bar_data_df) for a given universe."""

    @abstractmethod
    def bars(self, tickers: list[str], start: date, end: date) -> Iterator[tuple[date, pd.DataFrame]]:
        """Yield (bar_date, df) in chronological order.

        df: DataFrame with index=ticker, columns=OHLCV (+ any cached features).
        """

    @abstractmethod
    def history(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Return full OHLCV history as MultiIndex(date, ticker) DataFrame."""


class HistoricalProvider(DataProvider):
    """Replays daily OHLCV bars from yfinance (or Parquet cache)."""

    def __init__(self, cache_dir: str | Path = "data/cache"):
        self._cache_dir = Path(cache_dir)

    def history(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        frames = []
        for ticker in tickers:
            df = self._fetch_ticker(ticker, start, end)
            if df.empty:
                continue
            df["ticker"] = ticker
            df = df.reset_index().rename(columns={"Date": "date", "index": "date"})
            df = df.set_index(["date", "ticker"])
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()

    def bars(self, tickers: list[str], start: date, end: date) -> Iterator[tuple[date, pd.DataFrame]]:
        full = self.history(tickers, start, end)
        if full.empty:
            return
        for bar_date, group in full.groupby(level="date"):
            bar_df = group.droplevel("date")
            yield bar_date, bar_df

    def _fetch_ticker(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        cache_path = self._cache_dir / ticker / f"{start.year}_{end.year}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
            return df[mask]

        tk = yf.Ticker(ticker)
        df = tk.history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                        interval="1d", auto_adjust=True)
        if df.empty:
            return df
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        return df


class LiveProvider(DataProvider):
    """Fetches the latest bar from yfinance for paper/live trading."""

    def history(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        return HistoricalProvider().history(tickers, start, end)

    def bars(self, tickers: list[str], start: date, end: date) -> Iterator[tuple[date, pd.DataFrame]]:
        end_date = date.today()
        frames = []
        for ticker in tickers:
            tk = yf.Ticker(ticker)
            df = tk.history(period="2d", interval="1d", auto_adjust=True)
            if df.empty:
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index.name = "date"
            latest = df.iloc[[-1]]
            latest["ticker"] = ticker
            latest = latest.reset_index().set_index(["date", "ticker"])
            frames.append(latest)

        if not frames:
            return
        full = pd.concat(frames).sort_index()
        for bar_date, group in full.groupby(level="date"):
            yield bar_date, group.droplevel("date")
