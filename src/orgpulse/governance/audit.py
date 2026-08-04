from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

_GENESIS = "0" * 64


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload_hash(payload: dict) -> str:
    # Canonical serialization so hashes are order-independent and reproducible.
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _entry_hash(prev_hash: str, seq: int, timestamp: str, trace_id: str, kind: str, payload_hash: str) -> str:
    return _sha256("|".join([prev_hash, str(seq), timestamp, trace_id, kind, payload_hash]))


@dataclass(frozen=True)
class AuditEntry:
    seq: int
    timestamp: str
    trace_id: str
    kind: str
    payload: dict
    payload_hash: str
    prev_hash: str
    entry_hash: str


class HashChainedAuditLog:
    """Append-only, tamper-evident audit log backed by a JSONL file.

    Each entry's ``entry_hash`` chains the previous entry's hash, so altering any
    past entry breaks the chain and ``verify()`` returns False. Supports full
    reconstruction of any decision by ``trace_id`` (EU AI Act Article 12).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_raw(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, kind: str, payload: dict, trace_id: str, timestamp: str | None = None) -> AuditEntry:
        raw = self._read_raw()
        seq = len(raw)
        prev_hash = raw[-1]["entry_hash"] if raw else _GENESIS
        ts = timestamp or datetime.now().isoformat(timespec="seconds")
        ph = _payload_hash(payload)
        eh = _entry_hash(prev_hash, seq, ts, trace_id, kind, ph)
        entry = AuditEntry(
            seq=seq, timestamp=ts, trace_id=trace_id, kind=kind, payload=payload,
            payload_hash=ph, prev_hash=prev_hash, entry_hash=eh,
        )
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
        return entry

    def entries(self) -> list[AuditEntry]:
        return [AuditEntry(**e) for e in self._read_raw()]

    def verify(self) -> bool:
        # Fail closed on any malformed log (bad JSON, wrong shape) rather than raising.
        try:
            raw = self._read_raw()
        except (json.JSONDecodeError, OSError):
            return False
        if not raw:                       # an empty/missing log is NOT a verified record
            return False
        prev_hash = _GENESIS
        for i, e in enumerate(raw):
            if not isinstance(e, dict):
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

    def reconstruct(self, trace_id: str) -> list[AuditEntry]:
        return [e for e in self.entries() if e.trace_id == trace_id]
