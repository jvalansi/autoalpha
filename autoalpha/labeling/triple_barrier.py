"""Triple-barrier labeling.

Labels each event with +1 (profit-take hit), -1 (stop-loss hit), or 0 (time expiry).

Default parameters:
  profit_take_mult = 2.0 × ATR(21)
  stop_loss_mult   = 1.0 × ATR(21)
  time_expiry      = 20 trading days

ATR computed as rolling mean of |close - prev_close| (daily close-to-close).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _atr(close: pd.Series, window: int = 21) -> pd.Series:
    """Average True Range from daily close-to-close absolute changes."""
    return close.diff().abs().rolling(window).mean()


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DatetimeIndex,
    profit_take_mult: float = 2.0,
    stop_loss_mult: float = 1.0,
    time_expiry: int = 20,
    atr_window: int = 21,
    entry_prices: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute triple-barrier labels for a set of entry events.

    Parameters
    ----------
    close : pd.Series
        Daily close prices indexed by date (pd.DatetimeIndex).
        Used to check barrier hits on forward bars. ATR is computed from this series.
    events : pd.DatetimeIndex
        Entry execution dates — the bar at which the fill occurs.
        For strategies that enter at the next bar's open, pass the execution date
        (signal_date + 1 trading day), not the signal date.
    profit_take_mult : float
        ATR multiplier for the upper (profit-take) barrier.
    stop_loss_mult : float
        ATR multiplier for the lower (stop-loss) barrier.
    time_expiry : int
        Maximum holding period in trading days (vertical barrier).
    atr_window : int
        ATR lookback window.
    entry_prices : pd.Series, optional
        Actual fill prices indexed by event date. When provided, barriers are
        anchored to these prices instead of close[event_date]. Use this to
        pass next-bar open prices for strategies that enter at the open.

    Returns
    -------
    pd.DataFrame indexed by entry date with columns:
        t1     – exit date
        label  – +1 (profit-take), -1 (stop-loss), 0 (time expiry)
        ret    – realized return from entry to exit
    """
    atr = _atr(close, atr_window)
    results = []

    for event_date in events:
        if event_date not in close.index:
            continue

        if entry_prices is not None and event_date in entry_prices.index:
            entry_price = float(entry_prices[event_date])
        else:
            entry_price = close[event_date]
        atr_val = atr.get(event_date, np.nan)  # type: ignore[call-overload]

        if pd.isna(atr_val) or atr_val <= 0:
            continue

        upper = entry_price + profit_take_mult * atr_val
        lower = entry_price - stop_loss_mult * atr_val

        # Slice forward prices up to time_expiry bars
        future_idx = close.index[close.index > event_date]
        if len(future_idx) == 0:
            continue

        n = min(time_expiry, len(future_idx))
        window = close.iloc[close.index.get_loc(event_date) + 1 : close.index.get_loc(event_date) + 1 + n]

        label = 0
        exit_date = window.index[-1]
        exit_price = window.iloc[-1]

        for bar_date, price in window.items():
            if price >= upper:
                label = 1
                exit_date = bar_date
                exit_price = price
                break
            elif price <= lower:
                label = -1
                exit_date = bar_date
                exit_price = price
                break

        realized_ret = (exit_price - entry_price) / entry_price
        results.append(
            {"t0": event_date, "t1": exit_date, "label": label, "ret": realized_ret}
        )

    if not results:
        return pd.DataFrame(columns=["t1", "label", "ret"]).rename_axis("t0")

    df = pd.DataFrame(results).set_index("t0")
    return df
