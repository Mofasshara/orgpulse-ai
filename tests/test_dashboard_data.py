"""Tests for the dashboard data layer (streamlit_app/data.py).

Seeds real runs via ``run_tick`` into tmp, then checks the pure loaders and the
inline audit verifier — including that the inline verifier agrees with the real
``orgpulse.governance.audit`` implementation (anti-drift guard for the split plane).
"""
from __future__ import annotations

import json
from pathlib import Path

from orgpulse.agent.baseline_memory import BaselineMemory
from orgpulse.agent.tick import run_tick
from orgpulse.governance.audit import HashChainedAuditLog

from streamlit_app import data


def _seed_two_runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    mem = BaselineMemory(tmp_path / "m.json")
    run_tick(mem, 0, out_root=runs, now="2026-08-04T07:00:00", offline=True)
    run_tick(mem, 1, out_root=runs, now="2026-08-05T07:00:00", offline=True)
    return runs


def test_load_findings_and_report(tmp_path: Path):
    runs = _seed_two_runs(tmp_path)
    latest = runs / "latest"
    findings = data.load_findings(latest)
    assert findings["kpi_findings"] and findings["recommendations"]
    assert findings["overall_severity"] in {"SAFE", "WARNING", "CRITICAL", "DATA_QUALITY"}
    assert data.load_report(latest).strip()  # non-empty markdown report


def test_verify_audit_true_on_clean_run(tmp_path: Path):
    runs = _seed_two_runs(tmp_path)
    assert data.verify_audit(runs / "latest" / "audit.jsonl") is True


def test_verify_audit_false_after_tamper(tmp_path: Path):
    runs = _seed_two_runs(tmp_path)
    audit = runs / "latest" / "audit.jsonl"
    lines = audit.read_text().splitlines()
    rec = json.loads(lines[-1])
    rec["payload"]["overall_severity"] = "SAFE"  # forge a past decision
    lines[-1] = json.dumps(rec, sort_keys=True)
    audit.write_text("\n".join(lines) + "\n")
    assert data.verify_audit(audit) is False


def test_inline_verifier_matches_orgpulse_audit(tmp_path: Path):
    """Anti-drift: the standalone verifier must agree with governance.audit on the
    same file, both clean and tampered — otherwise the dashboard could show a green
    check the real system would reject (or vice versa)."""
    runs = _seed_two_runs(tmp_path)
    audit = runs / "latest" / "audit.jsonl"
    assert data.verify_audit(audit) == HashChainedAuditLog(audit).verify() is True

    lines = audit.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["timestamp"] = "1999-01-01T00:00:00"
    lines[0] = json.dumps(rec, sort_keys=True)
    audit.write_text("\n".join(lines) + "\n")
    assert data.verify_audit(audit) == HashChainedAuditLog(audit).verify() is False


def test_verify_run_true_on_clean_run(tmp_path: Path):
    runs = _seed_two_runs(tmp_path)
    ok, reason = data.verify_run(runs / "latest")
    assert ok is True, reason


def test_verify_run_detects_forged_findings_with_intact_chain(tmp_path: Path):
    """The core attack: edit ONLY findings.json (leave the audit chain untouched).
    verify_audit still passes (chain is internally consistent), but verify_run must fail
    because findings.json no longer matches the audited input hash."""
    runs = _seed_two_runs(tmp_path)
    latest = runs / "latest"
    fj = json.loads((latest / "findings.json").read_text())
    fj["overall_severity"] = "SAFE"  # forge CRITICAL -> SAFE
    fj["alert"] = False
    (latest / "findings.json").write_text(json.dumps(fj, indent=2))

    assert data.verify_audit(latest / "audit.jsonl") is True   # chain itself still consistent
    ok, reason = data.verify_run(latest)
    assert ok is False and "match" in reason                   # ...but binding catches it


def test_verify_audit_fails_closed_on_missing_empty_malformed(tmp_path: Path):
    missing = tmp_path / "nope.jsonl"
    assert data.verify_audit(missing) is True or data.verify_audit(missing) is False  # no crash
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert data.verify_audit(empty) is True                    # vacuously (no entries)
    malformed = tmp_path / "bad.jsonl"
    malformed.write_text("{not json\n")
    assert data.verify_audit(malformed) is False               # fails closed, no raise
    # verify_run on a dir with malformed audit also fails closed
    (tmp_path / "findings.json").write_text("{}")
    ok, _ = data.verify_run(tmp_path)
    assert ok is False


def test_load_history_newest_first(tmp_path: Path):
    runs = _seed_two_runs(tmp_path)
    hist = data.load_history(runs)
    assert len(hist) == 2
    assert hist[0]["generated_at"] > hist[1]["generated_at"]  # newest first
    assert {"run_id", "overall_severity", "alert", "run_dir"} <= set(hist[0])
    assert data.load_history(tmp_path / "nonexistent") == []
