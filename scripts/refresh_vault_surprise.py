"""One-off repair: recompute earnings_surprise / revenue_surprise for vault rows
from FREEZE_START on, from the FMP earnings calendar.

update_vault.py used to forward-fill the last known surprise into every appended
row, so no report after the incremental updater took over (2026-06) reached the
vault. update_vault.py now refreshes surprise itself; this fixes the rows it
already wrote. Rows with no calendar report keep their existing value.

Usage:
    FMP_API_KEY=... python scripts/refresh_vault_surprise.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_event_features import VAULT_PATH, rewrite_columns
from autoalpha.data.event_features import fetch_earnings_calendar, latest_surprise

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FREEZE_START = pd.Timestamp("2026-06-01")  # last surprise change in the vault is the week of 2026-05-25
COLS = ["earnings_surprise", "revenue_surprise"]


def main() -> None:
    cur = pq.read_table(VAULT_PATH, columns=["date", "ticker", *COLS]).to_pandas().reset_index()
    cur["date"] = pd.to_datetime(cur["date"])
    # Start a quarter before the freeze so every ticker's pre-freeze report is found
    calendar = fetch_earnings_calendar((FREEZE_START - pd.Timedelta(days=120)).strftime("%Y-%m-%d"),
                                       pd.Timestamp.today().strftime("%Y-%m-%d"), os.environ["FMP_API_KEY"])
    fresh = latest_surprise(cur, calendar)
    frozen = (cur["date"] >= FREEZE_START).to_numpy()
    out = cur[COLS].copy()
    for col in COLS:
        vals = fresh[col].to_numpy()
        use = frozen & ~np.isnan(vals)
        out.loc[use, col] = vals[use]
        changed = use & ~np.isclose(vals, cur[col].to_numpy(), equal_nan=True)
        log.info("%s: %d of %d post-freeze rows changed", col, changed.sum(), frozen.sum())
    rewrite_columns(VAULT_PATH, out)


if __name__ == "__main__":
    main()
