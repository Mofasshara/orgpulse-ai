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
from pathlib import Path

_GENESIS = "0" * 64


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload_hash(payload: dict) -> str:
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _entry_hash(prev_hash: str, seq: int, timestamp: str, trace_id: str, kind: str, payload_hash: str) -> str:
    return _sha256("|".join([prev_hash, str(seq), timestamp, trace_id, kind, payload_hash]))


def load_findings(run_dir: str | Path) -> dict:
    """Load ``findings.json`` for one run directory."""
    return json.loads((Path(run_dir) / "findings.json").read_text())


def load_report(run_dir: str | Path) -> str:
    """Load the human-readable ``report.md`` for one run (empty string if absent)."""
    path = Path(run_dir) / "report.md"
    return path.read_text() if path.exists() else ""


def load_audit(audit_path: str | Path) -> list[dict]:
    """Return the audit entries (list of dicts) for one run, in sequence order.

    Malformed lines raise ``json.JSONDecodeError`` to the caller; ``verify_*`` treat that
    as a verification failure rather than crashing the dashboard.
    """
    path = Path(audit_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
    except (json.JSONDecodeError, OSError):
        return False
    prev_hash = _GENESIS
    for i, e in enumerate(entries):
        try:
            if e.get("seq") != i or e.get("prev_hash") != prev_hash:
                return False
            if _payload_hash(e["payload"]) != e.get("payload_hash"):
                return False
            expected = _entry_hash(prev_hash, i, e["timestamp"], e["trace_id"], e["kind"], e["payload_hash"])
            if expected != e.get("entry_hash"):
                return False
            prev_hash = e["entry_hash"]
        except (KeyError, TypeError):
            return False
    return True


def verify_run(run_dir: str | Path) -> tuple[bool, str]:
    """Verify the audit chain AND bind it to the findings on screen.

    Returns ``(ok, reason)``. Fails closed if: the chain doesn't verify; the log is
    empty or its first entry isn't a genesis ``run`` entry; or — critically — the current
    ``findings.json`` bytes don't hash to the ``input_hash`` the run entry attests to.
    This closes the forged-``findings.json`` hole where editing only the findings (leaving
    the intact chain untouched) would otherwise still show "verified".
    """
    run_dir = Path(run_dir)
    findings_path = run_dir / "findings.json"
    audit_path = run_dir / "audit.jsonl"

    if not findings_path.exists():
        return False, "findings.json missing"
    if not verify_audit(audit_path):
        return False, "audit chain does not verify"
    try:
        entries = load_audit(audit_path)
    except (json.JSONDecodeError, OSError):
        return False, "audit log unreadable"
    if not entries:
        return False, "audit log is empty"
    genesis = entries[0]
    if genesis.get("kind") != "run" or "input_hash" not in genesis.get("payload", {}):
        return False, "no genesis run entry with an input hash"

    actual = _sha256(findings_path.read_text())
    if actual != genesis["payload"]["input_hash"]:
        return False, "findings.json does not match the audited input hash (tampered)"
    return True, "chain verified and bound to findings.json"


def load_history(runs_root: str | Path) -> list[dict]:
    """Summarise every timestamped run under ``runs_root`` (excluding ``latest``).

    Returns newest-first dicts: ``run_id``, ``generated_at``, ``overall_severity``,
    ``alert``, ``run_dir`` — powering the dashboard's run-history trend strip.
    """
    root = Path(runs_root)
    if not root.exists():
        return []
    rows: list[dict] = []
    for child in root.iterdir():
        if not child.is_dir() or child.name == "latest":
            continue
        findings = child / "findings.json"
        if not findings.exists():
            continue
        d = json.loads(findings.read_text())
        rows.append(
            {
                "run_id": d.get("run_id", child.name),
                "generated_at": d.get("generated_at", child.name),
                # Fail-safe: a missing/blank severity is a data-quality problem, not "Safe".
                "overall_severity": d.get("overall_severity") or "DATA_QUALITY",
                "alert": bool(d.get("alert", False)),
                "run_dir": str(child),
            }
        )
    rows.sort(key=lambda r: r["generated_at"], reverse=True)
    return rows
