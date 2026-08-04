import pytest

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
    # ...and the remembered baseline is USABLE: a real detection, never DATA_QUALITY
    # from an undersized baseline (baseline window must stay >= CUSUM's 60-sample floor).
    assert all(kf.severity != "DATA_QUALITY" for kf in f1.kpi_findings), [
        (kf.kpi, kf.severity) for kf in f1.kpi_findings
    ]
    assert (tmp_path / "m.json").exists() and set(mem.load()) == set(saved0)  # memory persisted+updated
    assert f1.run_id.startswith("tick-1-")
    # runs/latest mirrors exactly the LATEST run — audit log does not accumulate across ticks
    latest_audit = HashChainedAuditLog(tmp_path / "runs" / "latest" / "audit.jsonl")
    assert latest_audit.entries()  # non-empty
    assert all(e.trace_id.startswith(f1.run_id) for e in latest_audit.entries())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline_days": 30},                       # below CUSUM floor
        {"base_days": 50},                           # tick 0 baseline < 60 samples
        {"base_days": 30, "days_per_period": 30},    # base_days below floor
        {"base_days": 60, "days_per_period": 90},    # detection window longer than history
        {"days_per_period": 0},                      # non-positive period
    ],
)
def test_run_tick_rejects_unsafe_params_instead_of_degrading(tmp_path, kwargs):
    """Codex R3: outside default params the old fix silently produced all-DATA_QUALITY or
    empty findings/memory. Preconditions are now enforced at entry."""
    mem = BaselineMemory(tmp_path / "m.json")
    with pytest.raises(ValueError):
        run_tick(mem, 1, out_root=tmp_path / "runs", now="2026-08-04T07:00:00", offline=True, **kwargs)


def test_run_tick_refuses_to_overwrite_a_durable_run(tmp_path):
    """Two ticks pinned to the same wall-clock second must not collide onto one dir and
    destroy the first run's durable record."""
    mem = BaselineMemory(tmp_path / "m.json")
    same = "2026-01-01T07:00:00"
    run_tick(mem, 0, out_root=tmp_path / "runs", now=same, offline=True)
    # tick 1 at the same instant lands in a DISTINCT dir (tick number in the name) — no collision
    run_tick(mem, 1, out_root=tmp_path / "runs", now=same, offline=True)
    run_dirs = {p.name for p in (tmp_path / "runs").iterdir() if p.is_dir() and p.name != "latest"}
    assert len(run_dirs) == 2
    # re-running the exact same tick+instant refuses to clobber the existing durable run
    with pytest.raises(FileExistsError):
        run_tick(mem, 0, out_root=tmp_path / "runs", now=same, offline=True)
