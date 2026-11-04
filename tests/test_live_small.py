import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_live_small as rls


def test_report_compares_account_with_sim_after_start(tmp_path, monkeypatch):
    state = tmp_path / "small_state.json"
    state.write_text(json.dumps({"start_date": "2026-11-02", "start_equity": 10_000.0, "signal": "sig",
                                 "last_run": "2026-11-04", "last_targets": 130, "last_orders": 134, "last_rejected": 1}))
    monkeypatch.setattr(rls, "STATE_PATH", state)

    class _Ex:
        def account_equity(self):
            return 10_150.0

    monkeypatch.setattr(rls, "executor_from_env", lambda dry_run=False: _Ex())
    # dated 11-02 is open(10-30) -> open(11-02): before the account traded, so excluded
    sim = [{"name": "sig", "daily_returns": {"2026-11-02": 0.50, "2026-11-03": 0.01, "2026-11-04": 0.01}}]
    line = rls.report_line(sim)
    assert "$10,150 (+1.50%)" in line
    assert "simulated +2.01%" in line
    assert "130 names, 134 orders, 1 rejected legs" in line


def test_no_report_before_first_run(tmp_path, monkeypatch):
    monkeypatch.setattr(rls, "STATE_PATH", tmp_path / "missing.json")
    assert rls.report_line([]) is None
