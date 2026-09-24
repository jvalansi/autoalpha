"""Backfill the earnings-event columns (gap_1d, ret_10d, sector_ret_1d,
days_since_earnings) into existing loop/vault parquets without a full rebuild.

The per-ticker FMP cache the build scripts read from is no longer on disk, so
this derives the new columns from each file's own OHLCV + sector plus a fresh
FMP earnings calendar. Re-running overwrites the columns (idempotent).

Usage:
    FMP_API_KEY=... python scripts/add_event_features.py [--only vault|loop]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoalpha.data.event_features import (
    EVENT_COLS, event_features, fetch_earnings_calendar, fetch_sector_closes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

LOOP_PATH = Path("data/loop_data.parquet")
VAULT_PATH = Path("data/vault_data.parquet")
BAR_COLS = ["date", "ticker", "Open", "Close", "sector"]
HISTORY_START = "2017-09-01"  # a quarter before loop_data starts, so early reports are placed
WARMUP_BARS = 70              # loop_data bars prepended to the vault: lookbacks + one prior report


def read_bars(path: Path) -> pd.DataFrame:
    return pq.read_table(path, columns=BAR_COLS).to_pandas().reset_index()[BAR_COLS]


def backfill(path: Path, calendar: pd.DataFrame, sector_closes: pd.DataFrame,
             warmup: pd.DataFrame | None = None) -> None:
    bars = read_bars(path)  # file row order
    n = len(bars)
    if warmup is not None:
        bars = pd.concat([bars, warmup], ignore_index=True)
    ev = event_features(bars, calendar, sector_closes).iloc[:n]
    del bars
    rewrite_columns(path, ev)
    log.info("%s: %d rows, non-null share %s", path, n,
             {c: round(float(ev[c].notna().mean()), 3) for c in EVENT_COLS})


def rewrite_columns(path: Path, new: pd.DataFrame) -> None:
    """Replace/append new's columns in path; new is aligned to the file's row order."""
    tmp = Path(str(path) + ".tmp")
    src = pq.ParquetFile(path)
    writer = None
    offset = 0
    for i in range(src.num_row_groups):
        df = src.read_row_group(i).to_pandas()
        k = len(df)
        for col in new.columns:
            df[col] = new[col].to_numpy()[offset:offset + k]
        offset += k
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema)
        writer.write_table(table.cast(writer.schema))
    writer.close()
    assert offset == len(new), (offset, len(new))
    tmp.rename(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["vault", "loop"])
    args = parser.parse_args()

    today = pd.Timestamp.today().normalize()
    calendar = fetch_earnings_calendar(HISTORY_START, today.strftime("%Y-%m-%d"), os.environ["FMP_API_KEY"])
    sector_closes = fetch_sector_closes(HISTORY_START, (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))

    loop_bars = read_bars(LOOP_PATH)
    warmup = loop_bars.sort_values("date").groupby("ticker").tail(WARMUP_BARS)
    del loop_bars

    if args.only != "loop":
        backfill(VAULT_PATH, calendar, sector_closes, warmup)
    if args.only != "vault":
        backfill(LOOP_PATH, calendar, sector_closes)


if __name__ == "__main__":
    main()
