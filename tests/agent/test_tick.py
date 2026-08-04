from orgpulse.agent.baseline_memory import BaselineMemory
from orgpulse.agent.tick import run_tick
from orgpulse.detect.baseline import Baseline
from orgpulse.governance.audit import HashChainedAuditLog

_KPIS = {"daily_enrollment", "pass_rate", "calibration_offset", "lead_time_days"}


def test_baseline_memory_round_trip(tmp_path):
    mem = BaselineMemory(tmp_path / "m.json")
    assert mem.load() == {}
    mem.save({"enroll": Baseline(mean=10.0, std=2.0, n=30)})
    got = mem.load()
    assert got["enroll"].mean == 10.0 and got["enroll"].n == 30


def test_tick0_bootstraps_and_populates_memory(tmp_path):
    mem = BaselineMemory(tmp_path / "m.json")
    f = run_tick(mem, 0, out_root=tmp_path / "runs", now="2026-08-04T07:00:00", offline=True)
    assert f.kpi_findings and (tmp_path / "runs" / "latest" / "findings.json").exists()
    assert _KPIS <= set(mem.load())                       # bootstrap populated memory
    assert HashChainedAuditLog(tmp_path / "runs" / "latest" / "audit.jsonl").verify() is True
    # bootstrap tick does not alarm on a KPI it has never seen before
    assert all("bootstrap" in kf.detail for kf in f.kpi_findings)


def test_tick1_uses_remembered_baseline(tmp_path):
    mem = BaselineMemory(tmp_path / "m.json")
    run_tick(mem, 0, out_root=tmp_path / "runs", now="2026-08-04T07:00:00", offline=True)  # bootstrap
    saved0 = mem.load()
    f1 = run_tick(mem, 1, out_root=tmp_path / "runs", now="2026-08-05T07:00:00", offline=True)
    # tick 1 detected against the REMEMBERED baseline — no KPI bootstrapped
    assert all("bootstrap" not in kf.detail for kf in f1.kpi_findings)
    assert (tmp_path / "m.json").exists() and set(mem.load()) == set(saved0)  # memory persisted+updated
    assert f1.run_id.startswith("tick-1-")
