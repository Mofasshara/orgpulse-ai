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
    assert data.verify_audit(empty) is False                   # an empty log is NOT verified
    missing = tmp_path / "nope.jsonl"
    assert data.verify_audit(missing) is False                 # nor a missing one
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


def _seal_run(run_dir: Path, findings: dict, *, report: str = "report", payload_extra: dict | None = None,
              drop_fields: tuple[str, ...] = ()) -> Path:
    """Build a genuinely hash-consistent run (findings + report + a valid chained audit with
    full lineage), so schema/lineage gates are exercised AFTER the hash binding passes."""
    import hashlib
    from orgpulse.governance.audit import HashChainedAuditLog

    run_dir.mkdir(parents=True, exist_ok=True)
    # Fill in schema-required top-level fields unless the test overrode them. Every valid Domain
    # has >=1 KPI and verify_run now rejects an empty kpi_findings (a false all-clear), so the
    # default includes one SAFE finding consistent with the default SAFE/alert=False severity.
    findings = {"run_id": "tick-0-t", "generated_at": "2026-01-01T00:00:00",
                "overall_severity": "SAFE",
                "kpi_findings": [{"kpi": "daily_enrollment", "severity": "SAFE", "detail": "nominal",
                                  "latest_value": 1.0}],
                "forecast": {}, "recommendations": [], "alert": False, **findings}
    fj = json.dumps(findings)
    (run_dir / "findings.json").write_text(fj)
    (run_dir / "report.md").write_text(report)
    payload = {
        "findings_hash": hashlib.sha256(fj.encode()).hexdigest(),
        "report_hash": hashlib.sha256(report.encode()).hexdigest(),
        "input_data_hash": "0" * 64, "seed": 1, "model": "offline-stub", "offline": True,
    }
    payload.update(payload_extra or {})
    for f in drop_fields:
        payload.pop(f, None)
    log = HashChainedAuditLog(run_dir / "audit.jsonl")
    log.append("run", payload, trace_id="tick-0-t", timestamp="t")
    recs = findings.get("recommendations")
    for i, rec in enumerate(r for r in (recs if isinstance(recs, list) else []) if isinstance(r, dict)):
        log.append("recommendation", dict(rec), trace_id=f"tick-0-t:{i}", timestamp="t")
    return run_dir


def test_verify_run_rejects_structurally_invalid_findings_even_when_hash_bound(tmp_path: Path):
    """#4: a hash-consistent run whose findings has recommendations as a non-list (or bad
    kpi_findings/forecast shape) must fail closed — the dashboard must never render it."""
    d = _seal_run(tmp_path / "bad_recs",
                  {"overall_severity": "SAFE", "forecast": {}, "recommendations": "not-a-list"})
    ok, reason = data.verify_run(d)
    assert ok is False and "recommendations" in reason

    d2 = _seal_run(tmp_path / "bad_kf",
                   {"overall_severity": "SAFE", "kpi_findings": "nope", "forecast": {}, "recommendations": []})
    assert data.verify_run(d2)[0] is False


def test_verify_run_rejects_nested_schema_malformations(tmp_path: Path):
    """#4 (r8): verify_run checked only top-level container TYPES; nested malformations passed as
    verified and crashed the app. Each hash-consistent-but-invalid shape must fail closed."""
    valid = {"forecast": {}, "recommendations": []}
    cases = {
        "kf_not_object": {**valid, "kpi_findings": ["not-an-object"]},
        "kf_missing_severity": {**valid, "kpi_findings": [{"kpi": "x"}]},
        "rec_not_object": {**valid, "recommendations": ["not-an-object"]},
        "rec_missing_text": {**valid, "recommendations": [{"kpi": "x", "capa_frame": "C", "review_status": "P", "disclaimer": "d"}]},
        "forecast_only_h30": {**valid, "forecast": {"daily_enrollment": {"h30": {"yhat": 1, "lower": 0, "upper": 2}}}},
        # r10 #4: a metadata-only forecast (the dashboard would pick this first key as the hero and
        # crash on h30) must fail closed — verify the hero the renderer actually uses.
        "forecast_metadata_only": {**valid, "forecast": {"calibration": {"applied": True, "scale_factor": 2.0}}},
        "generated_at_numeric": {**valid, "generated_at": 123},
        # r9 #3: non-finite numbers, malformed calibration, bad enum / boolean
        "nan_latest_value": {**valid, "kpi_findings": [{"kpi": "x", "severity": "SAFE", "detail": "d", "latest_value": float("nan")}]},
        "unknown_severity": {**valid, "kpi_findings": [{"kpi": "x", "severity": "BANANA", "detail": "d", "latest_value": None}]},
        "alert_not_bool": {**valid, "alert": "yes"},
        "calibration_string": {**valid, "forecast": {"daily_enrollment": {
            "calibration": "bad",
            "h30": {"yhat": 1, "lower": 0, "upper": 2}, "h60": {"yhat": 1, "lower": 0, "upper": 2}, "h90": {"yhat": 1, "lower": 0, "upper": 2}}}},
        "calibration_bad_scale": {**valid, "forecast": {"daily_enrollment": {
            "calibration": {"applied": True, "scale_factor": "bad"},
            "h30": {"yhat": 1, "lower": 0, "upper": 2}, "h60": {"yhat": 1, "lower": 0, "upper": 2}, "h90": {"yhat": 1, "lower": 0, "upper": 2}}}},
        "inf_yhat": {**valid, "forecast": {"daily_enrollment": {
            "h30": {"yhat": float("inf"), "lower": 0, "upper": 2}, "h60": {"yhat": 1, "lower": 0, "upper": 2}, "h90": {"yhat": 1, "lower": 0, "upper": 2}}}},
    }
    for name, findings in cases.items():
        ok, reason = data.verify_run(_seal_run(tmp_path / name, findings))
        assert ok is False, name


def test_load_history_survives_non_string_generated_at(tmp_path: Path):
    """#4 (r8): a historical run with a numeric generated_at crashed the sort (int vs str)."""
    runs = _seed_two_runs(tmp_path)
    run_dirs = sorted(p for p in runs.iterdir() if p.is_dir() and p.name != "latest")
    fj = json.loads((run_dirs[0] / "findings.json").read_text()); fj["generated_at"] = 123
    (run_dirs[0] / "findings.json").write_text(json.dumps(fj))
    hist = data.load_history(runs)                          # must not raise on mixed types
    assert len(hist) == 2


def _hero_fc(hero):
    horizons = {h: {"yhat": 1.0, "lower": 0.0, "upper": 2.0} for h in ("h30", "h60", "h90")}
    # nominal is always present in real cal_meta (both the applied and fallback branches of
    # forecast_node) — a fixture omitting it doesn't represent a real artifact.
    return {hero: {"calibration": {"applied": False, "raw_coverage": None, "nominal": 0.9}, **horizons}}


def _kf(kpi, sev="SAFE"):
    return [{"kpi": kpi, "severity": sev, "detail": "d", "latest_value": 1.0}]


def test_verify_run_validates_any_hero_and_rejects_multi_or_malformed_forecast(tmp_path: Path):
    """r10/r11 #4: the verifier must validate WHICHEVER hero the dashboard renders (not only
    pharma's), accept a valid non-pharma hero, and reject multi-key or malformed forecasts."""
    base = {"recommendations": []}
    # a valid NON-pharma hero forecast, for an ASSESSED KPI, must verify (mutation: restore the
    # daily_enrollment-only check)
    ok, _ = data.verify_run(_seal_run(
        tmp_path / "npl_ok", {**base, "kpi_findings": _kf("npl_ratio"), "forecast": _hero_fc("npl_ratio")}))
    assert ok is True
    # multiple top-level forecast keys are structurally ambiguous -> reject
    two = {**_hero_fc("risk"), **_hero_fc("other")}
    assert data.verify_run(_seal_run(
        tmp_path / "twokeys", {**base, "kpi_findings": _kf("risk"), "forecast": two}))[0] is False
    # a valid hero PLUS a hidden malformed key -> reject (the hidden one would escape a first-key-only check)
    hidden = {**_hero_fc("risk"), "hidden_bad": {"h30": {}}}
    assert data.verify_run(_seal_run(
        tmp_path / "hidden", {**base, "kpi_findings": _kf("risk"), "forecast": hidden}))[0] is False
    # a malformed non-pharma hero (missing h60/h90) -> reject
    bad = {"npl_ratio": {"h30": {"yhat": 1, "lower": 0, "upper": 2}}}
    assert data.verify_run(_seal_run(
        tmp_path / "npl_bad", {**base, "kpi_findings": _kf("npl_ratio"), "forecast": bad}))[0] is False
    # a forecast for a KPI that was never assessed is an invented entity -> reject
    assert data.verify_run(_seal_run(
        tmp_path / "unassessed", {**base, "kpi_findings": _kf("npl_ratio"), "forecast": _hero_fc("invented")}))[0] is False


def test_verify_run_requires_lineage_fields(tmp_path: Path):
    """#6: a run advertised as Article-12 lineage-complete must not verify without its lineage."""
    good = {"overall_severity": "SAFE", "forecast": {}, "recommendations": []}
    assert data.verify_run(_seal_run(tmp_path / "ok", good))[0] is True
    for missing in ("input_data_hash", "seed", "model"):
        d = _seal_run(tmp_path / f"no_{missing}", good, drop_fields=(missing,))
        ok, reason = data.verify_run(d)
        assert ok is False and missing in reason


def test_verify_run_and_history_never_crash_on_unreadable_artifacts(tmp_path: Path):
    """#5: directory-as-file, invalid UTF-8, and malformed history must fail closed, never raise."""
    d = tmp_path / "dir_findings"; d.mkdir(); (d / "findings.json").mkdir()
    assert data.verify_run(d)[0] is False
    u = tmp_path / "bad_utf8"; u.mkdir()
    (u / "findings.json").write_text("{}"); (u / "audit.jsonl").write_bytes(b"\xff\xfe")
    assert data.verify_audit(u / "audit.jsonl") is False
    # a runs root with a malformed run must not crash load_history
    root = tmp_path / "runs"; bad = root / "2026-01-01"; bad.mkdir(parents=True)
    (bad / "findings.json").write_bytes(b"\xff\xfe invalid")
    hist = data.load_history(root)
    assert hist and hist[0]["verified"] is False


def test_load_history_newest_first(tmp_path: Path):
    runs = _seed_two_runs(tmp_path)
    hist = data.load_history(runs)
    assert len(hist) == 2
    assert hist[0]["generated_at"] > hist[1]["generated_at"]  # newest first
    assert {"run_id", "overall_severity", "alert", "run_dir"} <= set(hist[0])
    assert data.load_history(tmp_path / "nonexistent") == []
