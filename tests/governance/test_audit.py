import json
from pathlib import Path

from orgpulse.governance.audit import AuditEntry, HashChainedAuditLog


def _log(tmp_path: Path) -> HashChainedAuditLog:
    log = HashChainedAuditLog(tmp_path / "audit.jsonl")
    log.append("run", {"severity": "CRITICAL"}, trace_id="run-1", timestamp="2026-08-04T07:00:00")
    log.append("recommendation", {"kpi": "enroll", "text": "t"}, trace_id="run-1:enroll:0", timestamp="2026-08-04T07:00:01")
    log.append("recommendation", {"kpi": "pass_rate", "text": "u"}, trace_id="run-1:pass_rate:1", timestamp="2026-08-04T07:00:02")
    return log


def test_append_and_entries_chain(tmp_path):
    log = _log(tmp_path)
    entries = log.entries()
    assert len(entries) == 3 and all(isinstance(e, AuditEntry) for e in entries)
    assert entries[0].prev_hash == "0" * 64                 # genesis
    assert entries[1].prev_hash == entries[0].entry_hash    # chained
    assert entries[2].prev_hash == entries[1].entry_hash
    assert [e.seq for e in entries] == [0, 1, 2]


def test_verify_true_on_clean_log(tmp_path):
    assert _log(tmp_path).verify() is True


def test_verify_false_after_tampering(tmp_path):
    log = _log(tmp_path)
    p = tmp_path / "audit.jsonl"
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["payload"]["text"] = "TAMPERED"                     # alter a past entry's payload
    lines[1] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n")
    assert HashChainedAuditLog(p).verify() is False         # chain detects it


def test_reconstruct_returns_only_trace(tmp_path):
    got = _log(tmp_path).reconstruct("run-1:enroll:0")
    assert len(got) == 1 and got[0].payload["kpi"] == "enroll"


def test_payload_hash_deterministic_and_order_independent(tmp_path):
    log = HashChainedAuditLog(tmp_path / "a.jsonl")
    e1 = log.append("k", {"a": 1, "b": 2}, trace_id="t", timestamp="2026-08-04T00:00:00")
    log2 = HashChainedAuditLog(tmp_path / "b.jsonl")
    e2 = log2.append("k", {"b": 2, "a": 1}, trace_id="t", timestamp="2026-08-04T00:00:00")
    assert e1.payload_hash == e2.payload_hash                # canonical (sort_keys)
