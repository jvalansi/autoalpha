"""Fetch and cache OHLCV + FMP fundamentals for an expanded ~50-stock universe.

Idempotent: skips tickers that already have a complete cache.
Run this before rebuild_loop_dataset.py / build_vault_dataset.py.

Usage:
    FMP_API_KEY=<key> python scripts/fetch_universe.py [--tickers A B C ...] [--start 2018-01-01]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoalpha.data.fetcher import get_earnings, get_fundamentals, get_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default ~50-stock universe: diversified S&P 500 large-caps since 2018
# ---------------------------------------------------------------------------

UNIVERSE: list[str] = [
    # Technology (existing + additions)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AVGO", "ORCL", "ADBE", "CRM", "INTC", "AMD", "QCOM",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "V", "MA", "AXP",
    # Healthcare
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE",
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST",
    # Energy
    "XOM", "CVX", "COP",
    # Industrials
    "BA", "CAT", "HON", "UPS",
    # Communication Services
    "VZ", "DIS", "NFLX",
    # Materials
    "LIN",
    # Utilities
    "NEE",
]

CACHE_ROOT = Path("data/cache")
FMP_SLEEP = 0.25  # seconds between FMP calls to avoid rate-limiting


def _is_cached(ticker: str, start: date, end: date) -> bool:
    """Return True if OHLCV + fundamentals + earnings are already cached."""
    d = CACHE_ROOT / ticker
    if not d.exists():
        return False
    # Check at least one year parquet exists
    if not list(d.glob("[0-9]*.parquet")):
        return False
    if not (d / "fmp_fundamentals.parquet").exists():
        return False
    if not (d / "fmp_earnings.parquet").exists():
        return False
    return True


def fetch_ticker(ticker: str, start: date, end: date, api_key: str, force: bool = False) -> bool:
    if not force and _is_cached(ticker, start, end):
        log.info("%-6s  already cached — skipping", ticker)
        return True

    log.info("%-6s  fetching OHLCV...", ticker)
    try:
        get_ohlcv(ticker, start, end)
    except Exception as exc:
        log.warning("%-6s  OHLCV failed: %s", ticker, exc)
        return False

    log.info("%-6s  fetching fundamentals...", ticker)
    try:
        get_fundamentals(ticker, start, end, api_key=api_key)
        time.sleep(FMP_SLEEP)
    except Exception as exc:
        log.warning("%-6s  fundamentals failed: %s", ticker, exc)

    log.info("%-6s  fetching earnings...", ticker)
    try:
        get_earnings(ticker, start, end, api_key=api_key)
        time.sleep(FMP_SLEEP)
    except Exception as exc:
        log.warning("%-6s  earnings failed: %s", ticker, exc)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch universe data into cache")
    parser.add_argument("--tickers", nargs="+", default=None, help="Override ticker list")
    parser.add_argument("--start", default="2018-01-01", help="History start date")
    parser.add_argument("--end", default="2024-05-20", help="History end date (training cutoff)")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if already cached")
    args = parser.parse_args()

    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        log.error("FMP_API_KEY not set — fundamentals and earnings will fail")
        sys.exit(1)

    tickers = args.tickers or UNIVERSE
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    log.info("Fetching %d tickers from %s to %s", len(tickers), start, end)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, []
    for ticker in tickers:
        success = fetch_ticker(ticker, start, end, api_key, force=args.force)
        if success:
            ok += 1
        else:
            failed.append(ticker)

    log.info("Done: %d OK, %d failed", ok, len(failed))
    if failed:
        log.warning("Failed tickers: %s", failed)


if __name__ == "__main__":
    main()
