"""Pure data-access layer for the OrgPulse dashboard.

Deliberately self-contained: this module reads the *committed run artifacts*
(``findings.json``, ``report.md``, ``audit.jsonl``) and imports **no** ``orgpulse``
package code. The dashboard is the presentation plane of a split-plane system —
keeping it free of the heavy compute stack (Prophet, LangGraph, scikit-learn) means
Streamlit Cloud installs only ``streamlit``/``pandas``/``plotly``.

The inline :func:`verify_audit` re-implements the hash-chain check from
``orgpulse.governance.audit`` so the dashboard can prove tamper-evidence without
importing it. ``tests/test_dashboard_data.py`` asserts the two agree (anti-drift).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

_GENESIS = "0" * 64
_SEVERITIES = {"SAFE", "WARNING", "CRITICAL", "DATA_QUALITY"}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload_hash(payload: dict) -> str:
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _entry_hash(prev_hash: str, seq: int, timestamp: str, trace_id: str, kind: str, payload_hash: str) -> str:
    return _sha256("|".join([prev_hash, str(seq), timestamp, trace_id, kind, payload_hash]))


def _safe_read_text(path: Path) -> str | None:
    """Read a file's text, returning None on ANY filesystem/encoding problem (missing, a
    directory, invalid UTF-8, permissions). Every artifact read on the presentation plane goes
    through this so a malformed file fails closed instead of crashing the dashboard/history."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def load_findings(run_dir: str | Path) -> dict:
    """Load ``findings.json`` for one run directory."""
    return json.loads((Path(run_dir) / "findings.json").read_text())


def load_report(run_dir: str | Path) -> str:
    """Load the human-readable ``report.md`` for one run (empty string if absent/unreadable)."""
    return _safe_read_text(Path(run_dir) / "report.md") or ""


def load_audit(audit_path: str | Path) -> list[dict]:
    """Return the audit entries (list of dicts) for one run, in sequence order.

    Malformed lines raise ``json.JSONDecodeError`` to the caller; ``verify_*`` treat that
    (and any read/encoding error, surfaced as JSONDecodeError) as a verification failure.
    """
    path = Path(audit_path)
    if not path.exists():
        return []
    text = _safe_read_text(path)
    if text is None:  # unreadable / invalid UTF-8 / a directory → treat as malformed
        raise json.JSONDecodeError("unreadable audit log", "", 0)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def verify_audit(audit_path: str | Path) -> bool:
    """Re-verify the hash-chained audit log. Any tampering breaks the chain → False.

    Standalone re-implementation of ``orgpulse.governance.audit.HashChainedAuditLog.verify``
    so the dashboard stays free of the compute stack (proven equivalent in tests).
    Malformed/partial logs fail closed (return False) instead of raising.

    NOTE: this proves only the chain's *internal* consistency. It does NOT prove the log
    is bound to the findings on screen — use :func:`verify_run` for that.
    """
    try:
        entries = load_audit(audit_path)
    except (ValueError, OSError):
        # ValueError (not just json.JSONDecodeError, its subclass): json.loads raises a plain
        # ValueError — not JSONDecodeError — when a line contains an integer literal beyond
        # Python's int-string conversion digit limit (e.g. a 5001-digit number).
        return False
    if not entries:                       # an empty/missing log is NOT a verified log
        return False
    prev_hash = _GENESIS
    for i, e in enumerate(entries):
        if not isinstance(e, dict):        # e.g. a bare JSON list/scalar line
            return False
        try:
            if e.get("seq") != i or e.get("prev_hash") != prev_hash:
                return False
            if _payload_hash(e["payload"]) != e.get("payload_hash"):
                return False
            expected = _entry_hash(prev_hash, i, e["timestamp"], e["trace_id"], e["kind"], e["payload_hash"])
            if expected != e.get("entry_hash"):
                return False
            prev_hash = e["entry_hash"]
        except (KeyError, TypeError, AttributeError):
            return False
    return True


def _is_num(x) -> bool:
    # A FINITE real number — NaN/Inf are rejected (the dashboard formats these values and a
    # NaN/Inf would render as garbage or, for a bad forecast bound, plot nonsensically).
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    try:
        # math.isfinite raises OverflowError on an int too large to convert to float
        # (e.g. 10**1000) — that is not a finite number either; fail closed, don't crash.
        return math.isfinite(x)
    except OverflowError:
        return False


def _findings_schema_reason(fj: dict) -> str | None:
    """Validate the FULL findings schema the dashboard renders — not just the top-level
    container types. A hash-consistent but structurally invalid findings (a kpi_finding
    missing `severity`, a forecast with only `h30`, a numeric `generated_at`, a recommendation
    missing `text`) must fail closed here, before the app indexes into it and crashes."""
    for f in ("run_id", "generated_at"):
        if not isinstance(fj.get(f), str):
            return f"{f} is not a string"
    if fj.get("overall_severity") not in _SEVERITIES:
        return "overall_severity is not a known severity"
    # alert is a required, always-checked field — not merely validated when present. A missing
    # `alert` previously skipped both the type and the CRITICAL/DATA_QUALITY-implies-alert checks
    # below, letting a hash-consistent CRITICAL artifact with no alert flag verify as OK.
    if not isinstance(fj.get("alert"), bool):
        return "alert is not present or not a boolean"
    kfs = fj.get("kpi_findings", [])
    if not isinstance(kfs, list):
        return "kpi_findings is not a list"
    # Every valid Domain declares at least one KPI (enforced at Domain construction), and
    # detect_node treats "no findings" as DATA_QUALITY, never SAFE. An artifact with zero KPI
    # findings is therefore impossible to produce honestly — reject it rather than let it verify
    # as a false all-clear.
    if not kfs:
        return "kpi_findings is empty; every assessed run must have at least one KPI finding"
    for k in kfs:
        if not isinstance(k, dict) or not all(
            isinstance(k.get(x), str) and k.get(x).strip() for x in ("kpi", "detail")
        ):
            return "a kpi_finding is not an object with non-empty kpi/detail strings"
        if k.get("severity") not in _SEVERITIES:
            return "a kpi_finding severity is not a known severity"
        if not (k.get("latest_value") is None or _is_num(k.get("latest_value"))):
            return "a kpi_finding latest_value is not a finite number or null"
    # Computed here (not just near the recommendations check below) so the forecast validation
    # below can also reject a forecast for a KPI that was never assessed.
    assessed = {k.get("kpi") for k in kfs}
    fc = fj.get("forecast", {})
    if not isinstance(fc, dict):
        return "forecast is not an object"
    if len(fc) > 1:
        # The compute engine publishes EXACTLY ONE hero. Multiple top-level forecast keys are
        # structurally ambiguous (the dashboard would render only the first, hiding the rest) —
        # reject rather than partly validate.
        return f"forecast has {len(fc)} top-level keys; exactly one hero is expected"
    if fc:
        # Validate the single hero the dashboard renders (domain-agnostic: pharma
        # "daily_enrollment", banking "npl_ratio", …). A metadata-only or malformed forecast
        # (e.g. a stray top-level "calibration" key) must fail closed.
        hero_key = next(iter(fc))
        hero = fc[hero_key]
        # The compute engine only ever forecasts an ASSESSED KPI (forecast_node enforces
        # detect/forecast parity). A forecast key for a KPI that was never assessed is an
        # invented entity in governed output — reject rather than render it.
        if hero_key not in assessed:
            return f"forecast key {hero_key!r} is not among the assessed KPIs"
        if not isinstance(hero, dict):
            return f"forecast.{hero_key} is not an object"
        for h in ("h30", "h60", "h90"):  # the dashboard requires all three when a forecast is present
            hh = hero.get(h)
            if not isinstance(hh, dict) or not all(_is_num(hh.get(b)) for b in ("yhat", "lower", "upper")):
                return f"forecast.{hero_key}.{h} missing finite yhat/lower/upper"
            if not (hh["lower"] <= hh["yhat"] <= hh["upper"]):
                # The dashboard plots this as an interval band around the point — a producer bug
                # that emits lower > yhat or upper < yhat is a contradictory artifact; fail closed.
                return f"forecast.{hero_key}.{h} bounds violate lower <= yhat <= upper"
        # The dashboard dereferences and FORMATS the calibration block — validate it too.
        cal = hero.get("calibration")
        if cal is not None:
            if not isinstance(cal, dict) or not isinstance(cal.get("applied"), bool):
                return "forecast calibration is malformed (applied not boolean)"
            if cal.get("applied"):
                sf = cal.get("scale_factor")
                # A calibration scale is a positive widening factor (normally >= 1); zero,
                # negative, or non-finite is not a real calibration output.
                if not (_is_num(sf) and sf > 0.0):
                    return "forecast calibration scale_factor is not a positive finite number"
            rc = cal.get("raw_coverage")
            if not (rc is None or (_is_num(rc) and 0.0 <= rc <= 1.0)):
                return "forecast calibration raw_coverage is not a finite number in [0,1] or null"
            nom = cal.get("nominal")
            if not (_is_num(nom) and 0.0 <= nom <= 1.0):
                return "forecast calibration nominal is not a finite number in [0,1]"
    recs = fj.get("recommendations", [])
    if not isinstance(recs, list):
        return "recommendations is not a list"
    for r in recs:
        # Every governed-recommendation field must be a NON-EMPTY string — a blank text/frame is a
        # semantically hollow rec (a malformed domain callable could produce one), not a valid one.
        if not isinstance(r, dict) or not all(
            isinstance(r.get(x), str) and r.get(x).strip()
            for x in ("kpi", "text", "capa_frame", "review_status", "disclaimer")
        ):
            return "a recommendation is missing a required non-empty string field"
    # A flagged run MUST carry at least one governed recommendation. An alerting artifact
    # (WARNING/CRITICAL/DATA_QUALITY) with an empty recommendation stage is semantically
    # incomplete — the governance step silently vanished — so fail closed.
    sev = fj.get("overall_severity")
    if sev in ("WARNING", "CRITICAL", "DATA_QUALITY") and not recs:
        return f"overall_severity is {sev} but there are no governed recommendations"

    # Cross-field (relational) consistency. A hash-consistent artifact can still be internally
    # CONTRADICTORY from a producer bug — the dashboard renders these fields as one coherent whole,
    # so reject combinations the engine could never emit (not covered by the unkeyed-chain limit).
    kfs_sevs = [k.get("severity") for k in kfs]
    if kfs_sevs:
        # overall_severity is derived: DATA_QUALITY dominates, else the max of SAFE<WARNING<CRITICAL.
        _rank = {"SAFE": 0, "WARNING": 1, "CRITICAL": 2}
        expected_overall = "DATA_QUALITY" if "DATA_QUALITY" in kfs_sevs \
            else max(kfs_sevs, key=lambda s: _rank.get(s, -1))
        if sev != expected_overall:
            return f"overall_severity {sev!r} disagrees with the KPI findings (expected {expected_overall!r})"
    # The alert fires exactly when the run is CRITICAL or DATA_QUALITY. `alert` was already
    # required to be present and boolean above, so this check is unconditional here.
    if fj["alert"] != (sev in ("CRITICAL", "DATA_QUALITY")):
        return f"alert {fj['alert']!r} disagrees with overall_severity {sev!r}"
    # Every recommendation must reference an ASSESSED KPI — no invented entities in governed output.
    for r in recs:
        if r.get("kpi") not in assessed:
            return f"recommendation references KPI {r.get('kpi')!r} not among the assessed KPIs"
    return None


def verify_run(run_dir: str | Path) -> tuple[bool, str]:
    """Verify the audit chain AND bind it to the findings on screen.

    Returns ``(ok, reason)``. Fails closed if: the chain doesn't verify; the log is
    empty or its first entry isn't a genesis ``run`` entry; or — critically — the current
    ``findings.json`` bytes don't hash to the ``findings_hash`` the run entry attests to;
    ``report.md`` doesn't match its ``report_hash``; the required lineage fields are missing;
    or the findings schema is invalid.
    This closes the forged-``findings.json`` hole where editing only the findings (leaving
    the intact chain untouched) would otherwise still show "verified".
    """
    run_dir = Path(run_dir)
    findings_path = run_dir / "findings.json"
    report_path = run_dir / "report.md"
    audit_path = run_dir / "audit.jsonl"

    if not findings_path.exists():
        return False, "findings.json missing"
    if not verify_audit(audit_path):
        return False, "audit chain does not verify"
    try:
        entries = load_audit(audit_path)
    except (ValueError, OSError):
        return False, "audit log unreadable"
    if not entries:
        return False, "audit log is empty"

    # Validate the genesis run entry's SHAPE before indexing into it — a semantically
    # malformed but hash-consistent log (payload a list, etc.) must fail closed, not crash.
    genesis = entries[0]
    if not isinstance(genesis, dict) or genesis.get("kind") != "run":
        return False, "no genesis run entry"
    payload = genesis.get("payload")
    if not isinstance(payload, dict):
        return False, "malformed genesis payload"
    findings_hash = payload.get("findings_hash")
    if not isinstance(findings_hash, str):
        return False, "genesis run entry has no findings hash"

    # Require the Article-12 LINEAGE fields the docs advertise — a run missing them is not
    # "lineage-complete" and must not verify (the schema always writes these).
    for field in ("report_hash", "input_data_hash", "model"):
        if not isinstance(payload.get(field), str):
            return False, f"genesis run entry missing lineage field: {field}"
    if "seed" not in payload:
        return False, "genesis run entry missing lineage field: seed"
    report_hash = payload["report_hash"]

    findings_text = _safe_read_text(findings_path)
    if findings_text is None:
        return False, "findings.json is unreadable (bad encoding / not a file)"
    if _sha256(findings_text) != findings_hash:
        return False, "findings.json does not match the audited findings hash (tampered)"

    # report.md is a human-consumed artifact, so it is REQUIRED to be bound (forging it fails).
    report_text = _safe_read_text(report_path)
    if report_text is None or _sha256(report_text) != report_hash:
        return False, "report.md missing/unreadable or does not match the audited report hash"

    # Validate the findings SCHEMA before trusting it — a hash-consistent but structurally
    # invalid findings (e.g. recommendations not a list) must fail closed, not be rendered.
    try:
        fj = json.loads(findings_text)
    except ValueError:
        # ValueError, not just json.JSONDecodeError: an integer literal beyond Python's
        # int-string conversion digit limit (e.g. a 5001-digit number) raises a plain ValueError.
        return False, "findings.json is not valid JSON"
    if not isinstance(fj, dict):
        return False, "findings.json has an unexpected shape"
    schema_bad = _findings_schema_reason(fj)
    if schema_bad is not None:
        return False, f"findings.json schema invalid ({schema_bad})"

    # Bind the recommendation COUNT so tail-truncation is caught (a forward hash chain can't
    # detect deleted trailing entries on its own). Schema validation already confirmed it's a list.
    recs = fj.get("recommendations", [])
    n_logged = sum(1 for e in entries if isinstance(e, dict) and e.get("kind") == "recommendation")
    if n_logged != len(recs):
        return False, (
            f"audit records {n_logged} recommendation(s) but findings declares {len(recs)} "
            "(entries truncated or added)"
        )
    return True, "chain verified and bound to findings.json + report.md"


def load_history(runs_root: str | Path) -> list[dict]:
    """Summarise every timestamped run under ``runs_root`` (excluding ``latest``).

    Returns newest-first dicts: ``run_id``, ``generated_at``, ``overall_severity``,
    ``alert``, ``run_dir`` — powering the dashboard's run-history trend strip.
    """
    root = Path(runs_root)
    if not root.exists():
        return []
    rows: list[dict] = []
    for child in sorted(root.iterdir()):
        try:
            if not child.is_dir() or child.name == "latest":
                continue
            if not (child / "findings.json").exists():
                continue
            # Every historical run is verified and bound to its audit trail before it is trusted.
            # An unverified/forged/malformed run is surfaced as UNVERIFIED — its (untrusted)
            # severity is NOT shown as ordinary history, and one bad run never crashes the strip.
            ok, _reason = verify_run(child)
            text = _safe_read_text(child / "findings.json")
            d = json.loads(text) if text is not None else None
            if not isinstance(d, dict):
                ok, d = False, {}
        except (json.JSONDecodeError, OSError, ValueError):
            ok, d = False, {}
        rows.append(
            {
                "run_id": str(d.get("run_id") or child.name),
                # Always a string: an unverified run's findings may carry a non-string
                # generated_at, which would crash the sort below (int vs str comparison).
                "generated_at": str(d.get("generated_at") or child.name),
                "verified": bool(ok),
                # Only a verified run may show its severity; otherwise it reads UNVERIFIED
                # (never a forged "Safe"). Missing/blank severity is DATA_QUALITY, not Safe.
                "overall_severity": (d.get("overall_severity") or "DATA_QUALITY") if ok else "UNVERIFIED",
                "alert": bool(d.get("alert", False)) if ok else True,
                "run_dir": str(child),
            }
        )
    rows.sort(key=lambda r: r["generated_at"], reverse=True)
    return rows
