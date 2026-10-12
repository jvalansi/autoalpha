import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_live_pead import hold_weight


def test_held_shares_survive_equity_drift():
    # 2026-10-12: LW 211 @ 47.88, NEOG 798 @ 12.02, equity ~$99.7k re-read by the executor
    for shares, price in [(211, 47.88), (798, 12.02)]:
        w = hold_weight(shares, price, 99_726.0)
        for drift in (0.9995, 1.0, 1.0005):  # tolerance is ±0.5 share: ±0.06% for 798 NEOG
            assert int(w * 99_726.0 * drift / price) == shares
