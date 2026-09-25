"""Cross-sectional ML ranker in the style of Gu, Kelly & Xiu (2020, RFS).

One gradient-boosted tree model over all per-stock characteristics predicts each
stock's next-month return; each month the strategy holds the top decile, equal
weight. Long-only: SimExecutor and the paper account drop negative weights.

  Features: every numeric per-stock column in the dataset, ranked cross-sectionally
            to [0, 1] per date (missing -> 0.5, the median), as in GKX. Market-wide
            columns (VIX, yields) are constant across stocks, so they are dropped.
  Target:   return from the next bar's open to the open HORIZON bars later (the
            executor fills at the next open), ranked cross-sectionally per date.
  Training: every TRAIN_STRIDE-th date of the fit window; fixed hyperparameters.
  Trading:  rebalances on the first bar of each month and repeats the same targets
            in between — run with SimExecutor(hold_unchanged=True) so the book
            drifts instead of being traded back to equal weight every day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from autoalpha.core.strategy import Strategy

FEATURES = [
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_252d", "rsi_14", "pct_from_52w_high",
    "vol_21d", "gap_1d", "sector_ret_1d", "roe", "net_margin", "debt_to_equity", "earnings_surprise",
    "revenue_surprise", "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda", "analyst_revision_3m",
    "dividend_yield", "fcf_yield", "sentiment_score", "days_since_earnings",
]
HORIZON = 21          # trading days in the forward-return label
TRAIN_STRIDE = 5      # sample every 5th date for training (labels overlap anyway)
MIN_PRICE = 5.0       # skip penny stocks
TOP_QUANTILE = 0.10


def rank_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile ranks of FEATURES for one date; missing -> 0.5."""
    cols = [c for c in FEATURES if c in bars.columns]
    return bars[cols].rank(pct=True).reindex(columns=FEATURES).fillna(0.5)


def forward_returns(data: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """Open[t+1+horizon] / Open[t+1] - 1 per (date, ticker): the return of a position
    decided on bar t and filled at the next open."""
    opens = data["Open"].where(data["Open"] > 0).unstack("ticker").sort_index()  # data has zero prices
    fwd = opens.shift(-(horizon + 1)) / opens.shift(-1) - 1
    return fwd.stack().rename("fwd").reindex(data.index)


class MLCrossSection(Strategy):
    def __init__(self, random_state: int = 0) -> None:
        self._random_state = random_state
        self._model: HistGradientBoostingRegressor | None = None
        self._reset_book()

    def _reset_book(self) -> None:
        self._targets: dict[str, float] = {}
        self._month: tuple[int, int] | None = None
        self._last_date: pd.Timestamp | None = None

    def fit(self, data: pd.DataFrame) -> None:
        self._reset_book()
        fwd = forward_returns(data)
        all_dates = data.index.get_level_values("date")
        mask = all_dates.isin(all_dates.unique().sort_values()[::TRAIN_STRIDE]) & (data["Close"] >= MIN_PRICE)
        cols = [c for c in FEATURES if c in data.columns]
        sample = data.loc[mask, cols].assign(fwd=fwd[mask]).dropna(subset=["fwd"])
        X, y = [], []
        for _, day in sample.groupby(level="date"):
            if len(day) < 50:
                continue
            X.append(rank_features(day))
            y.append(day["fwd"].rank(pct=True))
        self._model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
            l2_regularization=1.0, early_stopping=False, random_state=self._random_state,
        ).fit(pd.concat(X).to_numpy(), np.concatenate([s.to_numpy() for s in y]))

    def score(self, bar_data: pd.DataFrame) -> pd.Series:
        bars = bar_data[(bar_data["Close"] >= MIN_PRICE) & bar_data["Open"].notna()]
        return pd.Series(self._model.predict(rank_features(bars).to_numpy()), index=bars.index)

    def predict(self, bar_data: pd.DataFrame, bar_date: pd.Timestamp | None = None) -> dict[str, float]:
        if self._model is None:
            return {}
        if bar_date is not None and self._last_date is not None and bar_date <= self._last_date:
            self._reset_book()  # a new backtest fold started earlier in time
        self._last_date = bar_date
        month = (bar_date.year, bar_date.month) if bar_date is not None else None
        if month is None or month != self._month:
            scores = self.score(bar_data)
            top = scores.nlargest(max(1, int(len(scores) * TOP_QUANTILE))).index
            self._targets = {t: 1.0 / len(top) for t in top}
            self._month = month
        return dict(self._targets)
