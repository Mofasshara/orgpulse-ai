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


def test_run_tick_rejects_non_integer_params(tmp_path):
    """The public Python API must honor its 'params validated at entry' claim (Typer guards
    the CLI, but direct callers can pass a float)."""
    mem = BaselineMemory(tmp_path / "m.json")
    with pytest.raises(TypeError):
        run_tick(mem, 0, baseline_days=60.5, out_root=tmp_path / "runs", now="2026-08-04T07:00:00", offline=True)


def test_bootstrap_tick_with_corrupt_current_is_data_quality_not_safe(tmp_path, monkeypatch):
    """Fail-open regression (#3): on the bootstrap tick (no remembered baseline) run_tick set
    SAFE unconditionally, without validating the current data. Corrupt current data must be
    DATA_QUALITY + alert, even with no prior baseline."""
    import numpy as np
    from orgpulse.agent import tick as tickmod

    orig = tickmod.generate_qc
    monkeypatch.setattr(tickmod, "generate_qc",
                        lambda *a, **k: orig(*a, **k).assign(pass_rate=np.nan))
    f = run_tick(BaselineMemory(tmp_path / "m.json"), 0, out_root=tmp_path / "runs",
                 now="2026-08-04T07:00:00", offline=True)
    sev = {kf.kpi: kf.severity for kf in f.kpi_findings}
    assert sev["pass_rate"] == "DATA_QUALITY"          # corrupt current, no baseline -> DQ (not SAFE)
    assert f.overall_severity == "DATA_QUALITY" and f.alert is True


def test_tick_survives_wrong_dtype_generator_output(tmp_path, monkeypatch):
    """#2: a stringified/corrupt generator column must become DATA_QUALITY, not crash the tick."""
    from orgpulse.agent import tick as tickmod
    orig = tickmod.generate_qc
    monkeypatch.setattr(tickmod, "generate_qc",
                        lambda *a, **k: orig(*a, **k).assign(pass_rate="not-a-number"))
    f = run_tick(BaselineMemory(tmp_path / "m.json"), 0, out_root=tmp_path / "runs",
                 now="2026-08-04T07:00:00", offline=True)          # must not raise
    sev = {kf.kpi: kf.severity for kf in f.kpi_findings}
    assert sev["pass_rate"] == "DATA_QUALITY" and f.overall_severity == "DATA_QUALITY"


def test_tick_audit_records_remembered_baseline_lineage(tmp_path):
    """#5: the remembered baseline drives detection and is overwritten each tick, so it must be
    in the recorded lineage. Two ticks identical except for remembered baseline must record
    different baseline_input_hash (and produce different findings)."""
    from orgpulse.detect.baseline import Baseline
    from orgpulse.governance.audit import HashChainedAuditLog

    def run_with_baseline(mean, out):
        mem = BaselineMemory(tmp_path / f"m{mean}.json")
        mem.save({k: Baseline(mean=mean, std=1.0, n=120)
                  for k in ("daily_enrollment", "pass_rate", "calibration_offset", "lead_time_days")})
        run_tick(mem, 1, out_root=out, now="2026-01-01T00:00:00", offline=True, seed=77)
        entry = HashChainedAuditLog(out / "latest" / "audit.jsonl").entries()[0].payload
        return entry["baseline_input_hash"], entry["input_data_hash"]

    b_lo, in_lo = run_with_baseline(0.0, tmp_path / "A")
    b_hi, in_hi = run_with_baseline(1000.0, tmp_path / "B")
    assert in_lo == in_hi          # same seed/params -> same generated inputs
    assert b_lo != b_hi            # ...but different remembered baseline IS in the lineage


def test_tick_rejects_domain_violation_in_baseline_history_and_saves_no_baseline(tmp_path, monkeypatch):
    """#3 (r7): a domain violation anywhere in the ingested stream — including old baseline
    history, not just the current window — must be DATA_QUALITY, and no baseline may be
    persisted from corrupt data (a poisoned baseline would corrupt the next tick)."""
    import numpy as np
    from orgpulse.agent import tick as tickmod

    orig = tickmod.generate_qc

    def corrupt_history(*a, **k):
        df = orig(*a, **k).copy()
        df.iloc[0, df.columns.get_loc("pass_rate")] = 1.5   # impossible value in ancient history
        return df

    monkeypatch.setattr(tickmod, "generate_qc", corrupt_history)
    mem = BaselineMemory(tmp_path / "m.json")
    f = run_tick(mem, 0, out_root=tmp_path / "runs", now="2026-08-04T07:00:00", offline=True)
    sev = {kf.kpi: kf.severity for kf in f.kpi_findings}
    assert sev["pass_rate"] == "DATA_QUALITY"
    assert "pass_rate" not in mem.load()                    # corrupt baseline NOT persisted


def test_corrupt_baseline_memory_fails_clearly(tmp_path):
    """A corrupt memory file must raise a clear diagnosis, not a raw JSONDecodeError mid-tick."""
    p = tmp_path / "m.json"
    p.write_text("{bad")
    mem = BaselineMemory(p)
    with pytest.raises(ValueError, match="corrupt or unreadable"):
        mem.load()
    with pytest.raises(ValueError):
        run_tick(mem, 0, out_root=tmp_path / "runs", now="2026-08-04T07:00:00", offline=True)


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
