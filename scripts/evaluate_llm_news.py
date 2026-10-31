"""Does an LLM read of earnings press releases predict returns beyond the numbers?

Pre-registered in PLAN.md (Phase 5, "LLM press-release test", 2026-10-31) before any
score was looked at. Inputs: data/llm_news/scores.parquet from
scripts/score_press_releases.py, bars from data/vault_data.parquet.

Outcomes per event, reaction bar t0, abnormal vs the equal-weight universe
(stocks that opened/closed >= $5 the bar before), winsorized 1/99%:
    drift0    open(t0)  -> open(t0+10)   primary: PEAD-style entry at the reaction open
    intraday  open(t0)  -> close(t0)     reacting at the open, after the gap is priced
    drift1    open(t1)  -> open(t1+10)   entering a day late, after the reaction day

Regression: outcome ~ score + ranked eps surprise + ranked revenue surprise + ranked gap,
standard errors clustered by reaction date.

Pass (all three, on drift0):
    1. score coefficient t >= 2.0
    2. score coefficient > 0 in both halves (split 2025-10-01)
    3. events with score >= 3: mean abnormal return > 0.22% (2 x 11 bps) with t >= 2.0

Second run (--source sec, pre-registered 2026-10-31 after the first run failed):
full 8-K releases, gpt-4o-mini, reaction bars 2023-11-01 -> 2024-06-28, bars from
the loop + vault datasets. Its gate is the veto hypothesis the first run suggested,
on the PEAD-like subset (EPS beat >= 5%, revenue beat, gap >= 3%):
    1. events with score < 3 have mean abnormal drift0 < 0
    2. mean drift0 of score >= 3 minus score < 3 has Welch t >= 2.0
The first run's gate is printed as a secondary result.

Usage:
    python scripts/evaluate_llm_news.py [--source sec]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoalpha.evaluation.alpha import BENCHMARK_MIN_PRICE, equal_weight_benchmark

RUNS = {
    "fmp": {"scores": Path("data/llm_news/scores.parquet"), "split": pd.Timestamp("2025-10-01"),
            "bars": ["data/vault_data.parquet"]},
    "sec": {"scores": Path("data/llm_news_sec/scores.parquet"), "split": pd.Timestamp("2024-03-01"),
            "bars": ["data/loop_data.parquet", "data/vault_data.parquet"]},
}
HORIZON = 10
COST = 0.0022
MIN_T = 2.0
OUTCOMES = ["drift0", "intraday", "drift1"]


def load_bars(start: pd.Timestamp, end: pd.Timestamp, paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        t = ds.dataset(path).to_table(
            columns=["date", "ticker", "Open", "Close", "gap_1d", "earnings_surprise", "revenue_surprise"],
            filter=(ds.field("date") >= (start - pd.Timedelta(days=10)).to_pydatetime())
            & (ds.field("date") <= (end + pd.Timedelta(days=40)).to_pydatetime())).to_pandas()
        frames.append(t.reset_index() if "ticker" not in t.columns else t)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates(["date", "ticker"], keep="last").sort_values(["ticker", "date"])


def event_outcomes(bars: pd.DataFrame) -> pd.DataFrame:
    """Raw and abnormal outcome returns for every (ticker, date) bar."""
    opens = bars.pivot(index="date", columns="ticker", values="Open")
    closes = bars.pivot(index="date", columns="ticker", values="Close")
    bench_oo = equal_weight_benchmark(opens)  # dated d: open(d-1) -> open(d)
    # same universe as equal_weight_benchmark: opened >= $5 the bar before
    oc = (closes / opens.where(opens > 0) - 1).where(opens.shift(1) >= BENCHMARK_MIN_PRICE)
    bench_oc = oc.mean(axis=1)

    growth = (1 + bench_oo.reindex(opens.index).fillna(0.0)).cumprod()
    # benchmark compounded from open(d) to open(d + HORIZON) and from open(d+1) to open(d+1+HORIZON)
    b0 = growth.shift(-HORIZON) / growth - 1
    b1 = growth.shift(-HORIZON - 1) / growth.shift(-1) - 1

    # stock legs shifted on the same market calendar as the benchmark, so a ticker's
    # missing bar gives NaN instead of a longer window than the benchmark's
    o = opens.where(opens > 0)
    legs = {
        "drift0": o.shift(-HORIZON) / o - 1,
        "drift1": o.shift(-HORIZON - 1) / o.shift(-1) - 1,
        "intraday": closes / o - 1,
    }
    out = bars[["ticker", "date", "gap_1d", "earnings_surprise", "revenue_surprise"]].copy()
    for name, leg in legs.items():
        out = out.merge(leg.stack().rename(name).reset_index(), on=["date", "ticker"], how="left")
    out["drift0"] -= out["date"].map(b0)
    out["drift1"] -= out["date"].map(b1)
    out["intraday"] -= out["date"].map(bench_oc)
    return out


def regress(df: pd.DataFrame, y: str) -> tuple[float, float, int]:
    d = df.dropna(subset=[y, "score", "eps_r", "rev_r", "gap_r"])
    if len(d) < 50:
        return np.nan, np.nan, len(d)
    m = smf.ols(f"{y} ~ score + eps_r + rev_r + gap_r", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["date"].astype("int64")})
    return float(m.params["score"]), float(m.tvalues["score"]), len(d)


def mean_t(x: pd.Series) -> tuple[float, float, int]:
    x = x.dropna()
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 else np.nan, len(x)


def welch_t(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=list(RUNS), default="fmp")
    args = parser.parse_args()
    run = RUNS[args.source]
    SPLIT = run["split"]
    scores = pd.read_parquet(run["scores"]).dropna(subset=["score"])
    scores["date"] = pd.to_datetime(scores["date"])
    bars = load_bars(scores["date"].min(), scores["date"].max(), run["bars"])
    df = scores.merge(event_outcomes(bars), on=["ticker", "date"], how="left")

    for y in OUTCOMES:
        lo, hi = df[y].quantile([0.01, 0.99])
        df[y] = df[y].clip(lo, hi)
    for src, dst in [("earnings_surprise", "eps_r"), ("revenue_surprise", "rev_r"), ("gap_1d", "gap_r")]:
        df[dst] = df.groupby(df["date"].dt.to_period("Q"))[src].rank(pct=True) - 0.5

    print(f"{len(df)} scored events, {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
    print("\nScore distribution and mean abnormal return (%):")
    tab = df.groupby("score").agg(n=("score", "size"), **{y: (y, "mean") for y in OUTCOMES})
    tab[OUTCOMES] *= 100
    print(tab.round(2).to_string())
    print("\nCorrelation of score with controls:",
          df[["score", "eps_r", "rev_r", "gap_r"]].corr()["score"].drop("score").round(2).to_dict())

    print("\nScore coefficient (abnormal return per score point, %), controls: eps/rev surprise, gap")
    for y in OUTCOMES:
        b, t, n = regress(df, y)
        b_a, t_a, _ = regress(df[df["date"] < SPLIT], y)
        b_b, t_b, _ = regress(df[df["date"] >= SPLIT], y)
        print(f"  {y:9s} all {b * 100:+.3f} (t {t:+.2f}, n {n})  "
              f"first half {b_a * 100:+.3f} (t {t_a:+.2f})  second half {b_b * 100:+.3f} (t {t_b:+.2f})")

    print("\nGuidance label, mean abnormal drift0 (%):")
    print((df.groupby("guidance")["drift0"].agg(["size", "mean"]).assign(mean=lambda x: x["mean"] * 100)).round(2).to_string())

    pead = df[(df["earnings_surprise"] >= 0.05) & (df["revenue_surprise"] > 0) & (df["gap_1d"] >= 0.03)]
    print(f"\nPEAD-like subset (eps beat >= 5%, revenue beat, gap >= 3%): n {len(pead)}")
    for lab, sub in [("score >= 3", pead[pead["score"] >= 3]), ("score < 3", pead[pead["score"] < 3])]:
        m, t, n = mean_t(sub["drift0"])
        print(f"  {lab:10s} n {n:4d}  drift0 {m * 100:+.2f}% (t {t:+.2f})")

    b, t, _ = regress(df, "drift0")
    b_a, _, _ = regress(df[df["date"] < SPLIT], "drift0")
    b_b, _, _ = regress(df[df["date"] >= SPLIT], "drift0")
    m, mt, n = mean_t(df.loc[df["score"] >= 3, "drift0"])
    checks = {
        f"score t >= {MIN_T} (t {t:+.2f})": t >= MIN_T,
        f"positive in both halves ({b_a * 100:+.3f}, {b_b * 100:+.3f})": b_a > 0 and b_b > 0,
        f"score >= 3 mean > {COST:.2%} with t >= {MIN_T} ({m:.2%}, t {mt:+.2f}, n {n})": m > COST and mt >= MIN_T,
    }
    hi, lo = pead.loc[pead["score"] >= 3, "drift0"], pead.loc[pead["score"] < 3, "drift0"]
    veto = {
        f"PEAD-like score < 3 mean < 0 ({lo.mean():+.2%}, n {lo.notna().sum()})": lo.mean() < 0,
        f"PEAD-like score >= 3 minus < 3, Welch t >= {MIN_T} "
        f"({hi.mean() - lo.mean():+.2%}, t {welch_t(hi, lo):+.2f})": welch_t(hi, lo) >= MIN_T,
    }
    gates = [("first-run gate (drift0)", checks), ("veto gate (drift0)", veto)]
    if args.source == "sec":
        gates.reverse()  # the veto gate is this run's pre-registered test
    for i, (name, g) in enumerate(gates):
        print(f"\n{'Pre-registered' if i == 0 else 'Secondary'} {name}:")
        for k, v in g.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("VERDICT:", "PASS" if all(gates[0][1].values()) else "FAIL")


if __name__ == "__main__":
    main()
