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


def test_verify_run_detects_recommendation_tail_truncation(tmp_path: Path):
    """A forward hash chain can't detect deleted trailing entries on its own. verify_run binds
    the recommendation count from findings.json, so truncating rec records must fail."""
    runs = _seed_two_runs(tmp_path)
    latest = runs / "latest"
    lines = (latest / "audit.jsonl").read_text().splitlines()
    (latest / "audit.jsonl").write_text(lines[0] + "\n")       # keep only the genesis run entry
    assert data.verify_audit(latest / "audit.jsonl") is True   # chain still internally consistent
    ok, reason = data.verify_run(latest)
    assert ok is False and "recommendation" in reason


def test_verify_audit_fails_closed_on_missing_empty_malformed(tmp_path: Path):
    from orgpulse.governance.audit import HashChainedAuditLog

    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert data.verify_audit(empty) is True                    # vacuously (no entries)
    for bad in ("{not json\n", "[]\n", "123\n", '{"seq": 0}\n'):
        p = tmp_path / "bad.jsonl"
        p.write_text(bad)
        assert data.verify_audit(p) is False, bad              # dashboard verifier: no raise
        assert HashChainedAuditLog(p).verify() is False, bad   # core verifier: no raise either
    # verify_run on a dir with malformed audit also fails closed
    (tmp_path / "findings.json").write_text("{}")
    (tmp_path / "audit.jsonl").write_text("[]\n")
    ok, _ = data.verify_run(tmp_path)
    assert ok is False


def test_verify_run_does_not_crash_on_wrong_shaped_but_hash_consistent_logs(tmp_path: Path):
    """#6: semantically malformed but internally hash-consistent inputs previously crashed
    verify_run (list payload -> TypeError; findings=[] -> AttributeError). Must return False."""
    runs = _seed_two_runs(tmp_path)
    latest = runs / "latest"
    # findings.json replaced with a JSON list (valid JSON, wrong shape)
    (latest / "findings.json").write_text("[]")
    ok, _ = data.verify_run(latest)                    # must not raise
    assert ok is False
    # a brand-new dir whose genesis payload is a list, not a dict
    d = tmp_path / "weird"; d.mkdir()
    (d / "findings.json").write_text("{}")
    from orgpulse.governance.audit import HashChainedAuditLog
    HashChainedAuditLog(d / "audit.jsonl").append("run", {"not": "expected"}, trace_id="x")
    # hand-craft a payload that is a list to hit the shape guard
    import json as _json
    line = _json.loads((d / "audit.jsonl").read_text())
    line["payload"] = ["input_hash"]
    (d / "audit.jsonl").write_text(_json.dumps(line) + "\n")
    ok2, _ = data.verify_run(d)                        # must not raise
    assert ok2 is False


def test_verify_run_catches_forged_report(tmp_path: Path):
    """#8: report.md is a human-consumed artifact and is now audit-bound — forging it fails."""
    runs = _seed_two_runs(tmp_path)
    latest = runs / "latest"
    (latest / "report.md").write_text("FORGED SAFE REPORT")
    ok, reason = data.verify_run(latest)
    assert ok is False and "report.md" in reason


def test_load_history_marks_forged_and_malformed_runs_unverified(tmp_path: Path):
    """#5: historical runs are verified; a forged one shows UNVERIFIED (not its forged severity),
    and a malformed one never crashes the strip."""
    runs = _seed_two_runs(tmp_path)
    run_dirs = sorted(p for p in runs.iterdir() if p.is_dir() and p.name != "latest")
    # forge one historical findings.json (leave its audit intact)
    fj = json.loads((run_dirs[0] / "findings.json").read_text()); fj["overall_severity"] = "SAFE"
    (run_dirs[0] / "findings.json").write_text(json.dumps(fj))
    # corrupt another's findings.json entirely
    (run_dirs[1] / "findings.json").write_text("{bad json")

    hist = data.load_history(runs)                     # must not raise
    by_dir = {r["run_dir"]: r for r in hist}
    assert by_dir[str(run_dirs[0])]["verified"] is False
    assert by_dir[str(run_dirs[0])]["overall_severity"] == "UNVERIFIED"  # forged status NOT shown
    assert by_dir[str(run_dirs[1])]["verified"] is False


def test_load_history_newest_first(tmp_path: Path):
    runs = _seed_two_runs(tmp_path)
    hist = data.load_history(runs)
    assert len(hist) == 2
    assert hist[0]["generated_at"] > hist[1]["generated_at"]  # newest first
    assert {"run_id", "overall_severity", "alert", "run_dir"} <= set(hist[0])
    assert data.load_history(tmp_path / "nonexistent") == []
