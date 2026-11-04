import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_live_pead as rlp
from autoalpha.execution.risk import RiskStatus


def test_report_shows_last_run_summary(tmp_path, monkeypatch):
    state = tmp_path / "pead_state.json"
    state.write_text(json.dumps({"positions": {}, "last_run": "2026-11-04",
                                 "last_report": ["🟢 Risk OK", "entries: ADV", "Holding (1/10): ADV"]}))
    monkeypatch.setattr(rlp, "STATE_PATH", state)
    assert rlp.report_line("2026-11-04") == "*Live PEAD (last run 2026-11-04, at the open):* 🟢 Risk OK\nentries: ADV\nHolding (1/10): ADV"
    assert "DID NOT RUN 2026-11-05" in rlp.report_line("2026-11-05")
    assert "DID NOT RUN" not in rlp.report_line("2026-11-03")  # vault lagging is not a missed run


def test_no_report_before_first_run(tmp_path, monkeypatch):
    monkeypatch.setattr(rlp, "STATE_PATH", tmp_path / "missing.json")
    assert rlp.report_line("2026-11-04") is None


def test_alert_only_on_halt_or_notes():
    ok, halted = RiskStatus(entries_allowed=True), RiskStatus(entries_allowed=False)
    assert not rlp.alert_needed(ok, [])
    assert rlp.alert_needed(ok, ["FMP quotes failed"])
    assert rlp.alert_needed(halted, [])
