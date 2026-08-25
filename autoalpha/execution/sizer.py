"""Fractional-Kelly position sizing.

Strategies emit raw portfolio fractions ("2% of the book in AAPL"). The sizer
turns those into the fractions actually sent to the executor:

    final = raw × KELLY_FRACTION × kelly_leverage × darwinian_weight × meta_confidence

then caps each name at MAX_POSITION and scales the whole book down if gross
exposure exceeds MAX_GROSS.

* KELLY_FRACTION = 0.25 (quarter-Kelly). Full Kelly is growth-optimal but its
  drawdown profile is undeployable; quarter-Kelly cuts median drawdown ~75%
  for ~25% of the growth give-up.
* kelly_leverage = mu / sigma^2 from the signal's own realized alpha returns —
  the continuous-outcome Kelly optimum. Estimated on a rolling window, clipped
  to [0, MAX_KELLY_LEVERAGE] because mu/sigma^2 explodes on short samples.
* darwinian_weight comes from SignalLibrary (floor 0.3, ceiling 2.5).
* meta_confidence is the Phase 6 meta-model output; defaults to 1.0 until that
  model exists.
"""
from __future__ import annotations

import logging
from typing import Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KELLY_FRACTION = 0.25          # quarter-Kelly
MAX_POSITION = 0.05            # hard cap: no single name > 5% of the book
MAX_GROSS = 1.0                # total gross exposure cap (1.0 = no leverage)
MAX_KELLY_LEVERAGE = 2.0       # clip on the mu/sigma^2 estimate
MIN_KELLY_OBS = 20             # below this, don't trust the estimate
DEFAULT_KELLY_WINDOW = 63      # trading days (≈ 1 quarter), matches Darwinian window
META_CONFIDENCE_FLOOR = 0.5    # below this, skip the trade entirely (Phase 6)


def kelly_leverage(
    returns: pd.Series,
    window: int = DEFAULT_KELLY_WINDOW,
    max_leverage: float = MAX_KELLY_LEVERAGE,
    min_obs: int = MIN_KELLY_OBS,
) -> float:
    """Continuous Kelly optimum (mu / sigma^2) from realized returns.

    Returns 1.0 (neutral, i.e. "no Kelly opinion") when the sample is too
    short to estimate, and clips to [0, max_leverage] otherwise. A negative
    edge clips to 0 — the sizer will not flip a long signal short.
    """
    if returns is None or len(returns) == 0:
        return 1.0
    recent = returns.tail(window)
    if len(recent) < min_obs:
        return 1.0
    var = float(recent.var(ddof=1))
    if var <= 0:
        return 1.0
    lev = float(recent.mean()) / var
    if not np.isfinite(lev):
        return 1.0
    return float(np.clip(lev, 0.0, max_leverage))


def size_position(
    raw_fraction: float,
    kelly_lev: float = 1.0,
    darwinian_weight: float = 1.0,
    meta_confidence: float = 1.0,
    kelly_fraction: float = KELLY_FRACTION,
    max_position: float = MAX_POSITION,
) -> float:
    """Size a single position. Returns 0.0 when the trade should be skipped."""
    if raw_fraction <= 0:
        return 0.0
    if meta_confidence < META_CONFIDENCE_FLOOR:
        return 0.0
    sized = raw_fraction * kelly_fraction * kelly_lev * darwinian_weight * meta_confidence
    return float(min(sized, max_position))


class PositionSizer:
    """Applies fractional-Kelly sizing to a signal's raw target fractions.

    Stateless w.r.t. the book: `size()` takes one signal's targets at a time.
    Cross-signal aggregation (summing overlapping names, applying the gross
    cap across the whole book) is `combine()`.
    """

    def __init__(
        self,
        kelly_fraction: float = KELLY_FRACTION,
        max_position: float = MAX_POSITION,
        max_gross: float = MAX_GROSS,
        kelly_window: int = DEFAULT_KELLY_WINDOW,
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.max_gross = max_gross
        self.kelly_window = kelly_window

    def size(
        self,
        targets: Mapping[str, float],
        alpha_returns: Optional[pd.Series] = None,
        darwinian_weight: float = 1.0,
        meta_confidence: float = 1.0,
    ) -> dict[str, float]:
        """Size one signal's targets. `alpha_returns` drives the Kelly estimate."""
        if not targets:
            return {}
        lev = kelly_leverage(alpha_returns, window=self.kelly_window) \
            if alpha_returns is not None else 1.0
        sized = {
            ticker: size_position(
                frac,
                kelly_lev=lev,
                darwinian_weight=darwinian_weight,
                meta_confidence=meta_confidence,
                kelly_fraction=self.kelly_fraction,
                max_position=self.max_position,
            )
            for ticker, frac in targets.items()
        }
        return {t: v for t, v in sized.items() if v > 0}

    def combine(self, per_signal: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
        """Sum sized targets across signals, then apply per-name and gross caps.

        Overlapping names add up — two signals each wanting 3% of AAPL means
        6%, which the per-name cap then trims to 5%.
        """
        book: dict[str, float] = {}
        for targets in per_signal.values():
            for ticker, frac in targets.items():
                book[ticker] = book.get(ticker, 0.0) + frac

        book = {t: min(v, self.max_position) for t, v in book.items() if v > 0}

        gross = sum(book.values())
        if gross > self.max_gross and gross > 0:
            scale = self.max_gross / gross
            logger.info("Gross exposure %.1f%% exceeds cap %.1f%% — scaling by %.3f",
                        gross * 100, self.max_gross * 100, scale)
            book = {t: v * scale for t, v in book.items()}
        return book
