import json
from pathlib import Path

from typer.testing import CliRunner

from orgpulse.cli import app, write_run


def test_write_run_offline_writes_artifacts(tmp_path: Path):
    f = write_run(out=tmp_path, days=200, seed=42, offline=True)
    findings = json.loads((tmp_path / "findings.json").read_text())
    assert findings["run_id"] and "recommendations" in findings
    assert (tmp_path / "report.md").exists()
    assert f.overall_severity in {"SAFE", "WARNING", "CRITICAL", "DATA_QUALITY"}


def test_cli_run_offline(tmp_path: Path):
    result = CliRunner().invoke(app, ["run", "--days", "200", "--out", str(tmp_path), "--offline"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "findings.json").exists() and (tmp_path / "report.md").exists()


def test_write_run_twice_into_same_dir_stays_bound(tmp_path: Path):
    """Re-running into the same output dir must leave findings.json bound to its audit trail
    (audit was previously appended while findings was overwritten, breaking verify_run)."""
    from streamlit_app import data

    write_run(out=tmp_path, days=200, seed=1, offline=True)
    write_run(out=tmp_path, days=200, seed=2, offline=True)   # same dir, different data
    ok, reason = data.verify_run(tmp_path)
    assert ok is True, reason


def test_write_run_produces_verifying_audit_log(tmp_path: Path):
    from orgpulse.governance.audit import HashChainedAuditLog

    f = write_run(out=tmp_path, days=200, seed=42, offline=True)
    log = HashChainedAuditLog(tmp_path / "audit.jsonl")
    assert log.verify() is True                                  # tamper-evident chain intact
    kinds = [e.kind for e in log.entries()]
    assert kinds[0] == "run" and kinds.count("recommendation") == len(f.recommendations)
    # Every recommendation is reconstructable with trace_id + timestamp + model + input hash.
    for i, rec in enumerate(f.recommendations):
        got = log.reconstruct(f"{f.run_id}:{rec.kpi}:{i}")
        assert len(got) == 1
        p = got[0].payload
        assert p["review_status"] == "PENDING_HUMAN_REVIEW"
        assert "findings_hash" in p and p["model"] in ("offline-stub", "claude-sonnet-5")
        assert got[0].timestamp == f.generated_at


def test_run_audit_records_input_lineage_not_just_output_hash(tmp_path: Path):
    """Compliance honesty (#7): the run entry must record real INPUT lineage (seed, params,
    input-data hash) and OUTPUT integrity hashes — not an output hash mislabeled as input."""
    from orgpulse.governance.audit import HashChainedAuditLog

    write_run(out=tmp_path, days=200, seed=42, offline=True)
    run_entry = HashChainedAuditLog(tmp_path / "audit.jsonl").entries()[0].payload
    for key in ("findings_hash", "report_hash", "input_data_hash", "seed", "days"):
        assert key in run_entry, key
    assert run_entry["seed"] == 42 and run_entry["days"] == 200
    assert "input_hash" not in run_entry  # the misleading name is gone
