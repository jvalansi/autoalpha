from autoalpha.execution.risk import RiskGuard


def test_risk_guard_latches_and_resumes(tmp_path):
    g = RiskGuard(tmp_path / "risk.json", tmp_path / "HALT")
    assert g.evaluate(100_000, "2026-01-02").entries_allowed
    assert g.evaluate(99_000, "2026-01-05").entries_allowed          # -1% day
    s = g.evaluate(94_000, "2026-01-05")                             # -5% vs day open
    assert not s.entries_allowed and "daily loss" in s.reasons[0]
    assert not g.evaluate(95_000, "2026-01-06").entries_allowed      # latched overnight
    g.resume()
    assert g.evaluate(95_000, "2026-01-07").entries_allowed
    assert not g.evaluate(84_000, "2026-01-08").entries_allowed      # 16% below the 100k peak
    g.resume()
    g.halt("testing")
    assert "manual halt (testing)" in g.evaluate(84_000, "2026-01-09").reasons
