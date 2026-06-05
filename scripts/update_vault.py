"""Incrementally update vault_data.parquet with bars since the last date.

Fetches new OHLCV from yfinance, computes technical features using existing
vault history as the lookback window, forward-fills all quarterly fundamentals,
and appends new rows to vault_data.parquet.

Usage:
    python scripts/update_vault.py
"""
from __future__ import annotations

import io
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

VAULT_PATH = Path("data/vault_data.parquet")
LOOKBACK_DAYS = 260  # enough for 252-day pct_from_52w_high
BATCH_SIZE = 200


def fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), parse_dates=["observation_date"])
        df = df.rename(columns={"observation_date": "date", series_id: series_id})
        s = df.set_index("date")[series_id].replace(".", np.nan).astype(float)
        s.index = pd.to_datetime(s.index).normalize()
        return s[(s.index >= pd.Timestamp(start)) & (s.index < pd.Timestamp(end))]
    except Exception as exc:
        log.warning("FRED %s failed: %s", series_id, exc)
        return pd.Series(dtype=float)


def compute_features(close_history: pd.Series, new_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features for new_ohlcv rows, using close_history for lookback."""
    # Combine history close + new close for window calculations
    new_close = new_ohlcv["Close"]
    combined_close = pd.concat([close_history, new_close]).sort_index()
    combined_close = combined_close[~combined_close.index.duplicated(keep="last")]

    rows = []
    for date, row in new_ohlcv.iterrows():
        hist = combined_close[combined_close.index <= date]

        # Returns
        ret = {}
        for col, n in [("ret_1d", 1), ("ret_5d", 5), ("ret_21d", 21), ("ret_63d", 63), ("ret_252d", 252)]:
            if len(hist) > n:
                ret[col] = float(hist.iloc[-1] / hist.iloc[-1 - n] - 1)
            else:
                ret[col] = np.nan

        # RSI (14)
        if len(hist) >= 15:
            delta = hist.diff().iloc[-15:]
            gain = delta.clip(lower=0).mean()
            loss = (-delta.clip(upper=0)).mean()
            rs = gain / loss if loss != 0 else np.nan
            rsi = float(100 - 100 / (1 + rs)) if rs is not np.nan and not np.isnan(rs) else np.nan
        else:
            rsi = np.nan

        # 52-week high distance
        if len(hist) >= 2:
            high_252 = hist.iloc[-min(252, len(hist)):].max()
            pct_high = float((hist.iloc[-1] - high_252) / high_252) if high_252 > 0 else np.nan
        else:
            pct_high = np.nan

        # 21d vol
        if len(hist) >= 22:
            vol = float(hist.pct_change().iloc[-22:].std() * np.sqrt(252))
        else:
            vol = np.nan

        rows.append({
            "Open": row.get("Open", np.nan),
            "High": row.get("High", np.nan),
            "Low": row.get("Low", np.nan),
            "Close": row.get("Close", np.nan),
            "Volume": row.get("Volume", np.nan),
            **ret,
            "rsi_14": rsi,
            "pct_from_52w_high": pct_high,
            "vol_21d": vol,
        })

    return pd.DataFrame(rows, index=new_ohlcv.index)


def main() -> None:
    if not VAULT_PATH.exists():
        log.error("vault_data.parquet not found — run build_vault_dataset.py first")
        return

    today = pd.Timestamp.today().normalize()
    log.info("Loading existing vault...")
    import pyarrow.dataset as _ds
    vault_ds = _ds.dataset(str(VAULT_PATH), format="parquet")

    # Get last date cheaply
    last_date_table = vault_ds.to_table(columns=["date"])
    all_dates = pd.to_datetime(last_date_table.column("date").to_pylist())
    last_date = max(all_dates)
    log.info("Vault last date: %s", last_date.date())

    if last_date >= today:
        log.info("Already up to date")
        return

    fetch_start = last_date + pd.Timedelta(days=1)
    fetch_end = today + pd.Timedelta(days=1)
    log.info("Fetching new bars: %s → %s", fetch_start.date(), today.date())

    # Get tickers
    ticker_table = vault_ds.to_table(columns=["ticker"])
    tickers = sorted(set(ticker_table.column("ticker").to_pylist()))
    log.info("Tickers: %d", len(tickers))

    # Fetch new OHLCV in batches
    new_data: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        try:
            raw = yf.download(
                " ".join(batch),
                start=fetch_start.strftime("%Y-%m-%d"),
                end=fetch_end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            log.warning("Batch %d-%d failed: %s", i, i + len(batch), exc)
            continue

        if raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in batch:
                try:
                    df = raw.xs(ticker, axis=1, level=1).dropna(how="all")
                    if not df.empty:
                        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                        new_data[ticker] = df
                except (KeyError, Exception):
                    pass
        else:
            ticker = batch[0]
            raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
            new_data[ticker] = raw.dropna(how="all")

        log.info("  Fetched batch %d-%d (%d tickers with data)", i, i + len(batch), len(new_data))

    if not new_data:
        log.info("No new data available yet")
        return

    # Fetch macro/VIX for new dates (fall back to NaN if unavailable — forward-filled at row level)
    start_str = fetch_start.strftime("%Y-%m-%d")
    end_str = fetch_end.strftime("%Y-%m-%d")
    log.info("Fetching macro data...")
    dgs10 = fetch_fred_series("DGS10", start_str, end_str).rename("yield_10y")
    dgs2 = fetch_fred_series("DGS2", start_str, end_str).rename("yield_2y")
    spread = fetch_fred_series("BAA10Y", start_str, end_str).rename("credit_spread")
    macro_new = pd.concat([dgs10, dgs2, spread], axis=1) if any(not s.empty for s in [dgs10, dgs2, spread]) else pd.DataFrame()
    if not macro_new.empty:
        macro_new = macro_new.ffill()
        macro_new["yield_curve"] = macro_new.get("yield_10y", pd.Series()) - macro_new.get("yield_2y", pd.Series())
    log.info("Macro: %d rows fetched", len(macro_new))

    try:
        vix_new = yf.Ticker("^VIX").history(start=start_str, end=end_str, auto_adjust=True)["Close"]
        vix_new.index = pd.to_datetime(vix_new.index).tz_localize(None).normalize()
        log.info("VIX: %d rows fetched", len(vix_new))
    except Exception as exc:
        log.warning("VIX fetch failed: %s", exc)
        vix_new = pd.Series(dtype=float)

    # Load historical close prices (last LOOKBACK_DAYS) from vault for feature computation
    log.info("Loading close history window from vault...")
    history_start = (last_date - pd.Timedelta(days=LOOKBACK_DAYS)).isoformat()
    hist_table = vault_ds.to_table(
        filter=_ds.field("date") >= pd.Timestamp(history_start).to_pydatetime(),
    )
    hist_df = hist_table.to_pandas()
    # date/ticker may be in the index (pandas MultiIndex from parquet) or as columns
    if isinstance(hist_df.index, pd.MultiIndex):
        hist_df = hist_df.reset_index()
    elif "date" not in hist_df.columns:
        hist_df = hist_df.reset_index()
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    close_by_ticker = hist_df.pivot_table(index="date", columns="ticker", values="Close", aggfunc="last")

    # Also load last row of fundamentals for forward-fill
    log.info("Loading last fundamentals row from vault...")
    fund_cols = [
        "roe", "net_margin", "debt_to_equity", "earnings_surprise", "revenue_surprise",
        "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda", "analyst_revision_3m",
        "dividend_yield", "fcf_yield", "sector",
    ]
    last_date_table2 = vault_ds.to_table(
        filter=_ds.field("date") == last_date.to_pydatetime(),
    )
    last_fund_df = last_date_table2.to_pandas()
    if isinstance(last_fund_df.index, pd.MultiIndex):
        last_fund_df = last_fund_df.reset_index()
    elif "ticker" not in last_fund_df.columns:
        last_fund_df = last_fund_df.reset_index()
    avail_fund_cols = [c for c in fund_cols if c in last_fund_df.columns]
    last_fund = last_fund_df.set_index("ticker")[avail_fund_cols]

    # Build new rows per ticker
    canonical = [
        "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits",
        "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_252d",
        "rsi_14", "pct_from_52w_high", "vol_21d",
        "roe", "net_margin", "debt_to_equity",
        "earnings_surprise", "revenue_surprise",
        "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
        "analyst_revision_3m", "dividend_yield", "fcf_yield",
        "vix", "yield_10y", "yield_2y", "credit_spread", "yield_curve",
        "sector", "sentiment_score",
    ]

    new_rows = []
    for ticker, ohlcv in new_data.items():
        close_hist = close_by_ticker.get(ticker, pd.Series(dtype=float))
        if isinstance(close_hist, pd.DataFrame):
            close_hist = close_hist.squeeze()

        tech = compute_features(close_hist, ohlcv)

        # Forward-fill fundamentals from last known values
        fund = last_fund.loc[ticker] if ticker in last_fund.index else pd.Series(dtype=object)

        for date, tech_row in tech.iterrows():
            row = dict(tech_row)
            row["Dividends"] = np.nan
            row["Stock Splits"] = np.nan
            # Forward-fill fundamentals
            for col in ["roe", "net_margin", "debt_to_equity", "earnings_surprise", "revenue_surprise",
                        "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda", "analyst_revision_3m",
                        "dividend_yield", "fcf_yield", "sector"]:
                row[col] = fund.get(col, np.nan) if not fund.empty else np.nan
            # Macro
            row["vix"] = float(vix_new.get(date, np.nan)) if not vix_new.empty else np.nan
            for col in ["yield_10y", "yield_2y", "credit_spread", "yield_curve"]:
                row[col] = float(macro_new.at[date, col]) if not macro_new.empty and date in macro_new.index and col in macro_new.columns else np.nan
            row["sentiment_score"] = np.nan

            mi = pd.MultiIndex.from_tuples([(date, ticker)], names=["date", "ticker"])
            new_rows.append(pd.DataFrame([row], index=mi, columns=canonical))

    if not new_rows:
        log.info("No new rows to append")
        return

    new_df = pd.concat(new_rows).sort_index(level="date")
    for col in new_df.select_dtypes(include="number").columns:
        new_df[col] = new_df[col].astype("float64")

    log.info("Appending %d new rows (%d tickers, dates: %s → %s)",
             len(new_df),
             new_df.index.get_level_values("ticker").nunique(),
             new_df.index.get_level_values("date").min().date(),
             new_df.index.get_level_values("date").max().date())

    # Append to parquet
    table = pa.Table.from_pandas(new_df)
    existing_schema = pq.read_schema(VAULT_PATH)
    table = table.cast(existing_schema)
    with pq.ParquetWriter(str(VAULT_PATH) + ".tmp", existing_schema) as writer:
        for batch in pq.ParquetFile(VAULT_PATH).iter_batches():
            writer.write_batch(batch)
        writer.write_table(table)
    Path(str(VAULT_PATH) + ".tmp").rename(VAULT_PATH)
    log.info("Done — vault updated to %s", today.date())


if __name__ == "__main__":
    main()
