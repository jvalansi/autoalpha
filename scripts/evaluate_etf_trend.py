"""Backtest long-only ETF trend following (autoalpha/strategies/etf_trend.py).

Pre-registered before the first run (2026-09-25):
  Universe:   SPY IWM EFA EEM TLT IEF LQD GLD DBC VNQ — each joins once it has
              252 bars of history (yfinance adjusted OHLC, 2002 →).
  Execution:  SimExecutor at next open, 11 bps, hold_unchanged, no position cap.
  Benchmarks: 60/40 SPY/IEF and equal-weight of the same ETFs, both open-to-open,
              daily rebalanced.
  Windows:    pre-publication (→ 2012-12) and post-publication (2013-01 →) of
              Moskowitz-Ooi-Pedersen 2012. Parameters come from the literature,
              so 2013 → is out of sample for them.
  Success:    post-publication alpha vs 60/40 with t ≥ 2.0 and positive alpha
              pre-publication. Sharpe and max drawdown are reported alongside.

Usage:
    python scripts/evaluate_etf_trend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoalpha.core.executors import SimExecutor
from autoalpha.evaluation.alpha import compute_benchmark_alpha, equal_weight_benchmark
from autoalpha.strategies.etf_trend import ETFTrend, add_trend_features

ETFS = ["SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "DBC", "VNQ"]
START = "2002-01-01"
PUBLICATION = pd.Timestamp("2013-01-01")
MIN_ALPHA_T = 2.0


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    px = yf.download(ETFS, start=START, auto_adjust=True, progress=False)
    return px["Open"][ETFS], px["Close"][ETFS]


def backtest(opens: pd.DataFrame, closes: pd.DataFrame) -> pd.Series:
    feats = add_trend_features(closes)
    strategy = ETFTrend()
    executor = SimExecutor(initial_capital=100_000, cost_bps=11, max_weight=None, hold_unchanged=True)
    prev: dict[str, float] = {}
    for d in closes.index:
        executor.execute(prev, d.date(), opens.loc[d].dropna().to_dict())
        bar = pd.DataFrame({"Open": opens.loc[d], "Close": closes.loc[d],
                            "trend": feats["trend"].loc[d], "vol": feats["vol"].loc[d]})
        prev = strategy.predict(bar, bar_date=d)
    rets = executor.returns()
    first_trade = rets[rets != 0].index.min()
    return rets[rets.index >= first_trade]


def stats(r: pd.Series) -> dict:
    nav = (1 + r).cumprod()
    return {"CAGR%": round((nav.iloc[-1] ** (252 / len(r)) - 1) * 100, 1),
            "vol%": round(r.std() * np.sqrt(252) * 100, 1),
            "Sharpe": round(r.mean() / r.std() * np.sqrt(252), 2),
            "maxDD%": round(((nav / nav.cummax()) - 1).min() * 100, 1)}


def main() -> None:
    opens, closes = load()
    strat = backtest(opens, closes)
    o2o = opens.pct_change(fill_method=None)
    bms = {"60/40": (0.6 * o2o["SPY"] + 0.4 * o2o["IEF"]).dropna(),
           "EW ETFs": equal_weight_benchmark(opens)}

    windows = {"full": (strat.index.min(), strat.index.max()),
               "pre-pub": (strat.index.min(), PUBLICATION - pd.Timedelta(days=1)),
               "post-pub": (PUBLICATION, strat.index.max())}
    results = {}
    for w, (s, e) in windows.items():
        r = strat[(strat.index >= s) & (strat.index <= e)]
        print(f"\n{w}: {r.index.min().date()} → {r.index.max().date()} ({len(r)} bars)")
        print(f"  {'trend':8} {stats(r)}")
        for name, bm in bms.items():
            b = bm[(bm.index >= s) & (bm.index <= e)]
            a = compute_benchmark_alpha(r, b)
            results[(w, name)] = a
            print(f"  {name:8} {stats(b)}  alpha={a['alpha_annualized']*100:+.1f}%/yr "
                  f"t={a['alpha_t']:.2f} beta={a['beta']:.2f}")

    pre, post = results[("pre-pub", "60/40")], results[("post-pub", "60/40")]
    passed = post["alpha_t"] >= MIN_ALPHA_T and pre["alpha_annualized"] > 0
    print("\nPASS" if passed else
          f"\nFAIL — post-pub alpha vs 60/40 t={post['alpha_t']:.2f} (need {MIN_ALPHA_T}), "
          f"pre-pub alpha {pre['alpha_annualized']*100:+.1f}%/yr")


if __name__ == "__main__":
    main()
