# OrgPulse P5 — Governance & Compliance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Make the recommendations auditable to the EU AI Act high-risk bar: a **hash-chained, append-only, tamper-evident audit log** where every AI-generated recommendation is recorded with a `trace_id`, timestamp, input-data hash, and model+version so any past decision is fully **reconstructable** (Article 12), wired into `orgpulse run`; plus an **EU AI Act self-classification** (Annex III high-risk) as both a machine-readable classification and a Markdown technical-documentation doc (Article 11), with human-oversight already gated in the recommendations (Article 14, Plan 4).

**Architecture:** A new `orgpulse.governance` subpackage. `audit.py` — a `HashChainedAuditLog` backed by a JSONL file where each entry's hash chains the previous (SHA-256 over prev_hash + seq + timestamp + trace_id + kind + payload_hash); `verify()` detects any tampering; `reconstruct(trace_id)` returns the full decision context. `classification.py` — the Annex-III self-classification object + obligations map. `orgpulse run` appends a run entry + one entry per recommendation and (optionally) verifies the chain. Deterministic + keyless (no new deps — stdlib `hashlib`/`json`).

**Tech Stack:** Python 3.11, uv, stdlib hashlib/json, pytest.

## Global Constraints
- Python 3.11; deps via uv; **no new deps** (stdlib only). Use `uv run`.
- Audit log is **append-only** and **tamper-evident**: modifying any past entry must make `verify()` return False.
- Every recommendation entry carries `trace_id`, timestamp, input hash, model+version. JSON-safe (allow_nan=False).
- Deterministic given inputs; keyless. Commit style: no Co-Authored-By.
- TDD; never weaken a test.

---

### Task 1: Hash-chained tamper-evident audit log

**Files:** Create `src/orgpulse/governance/__init__.py`, `src/orgpulse/governance/audit.py`; Test `tests/governance/__init__.py` (empty), `tests/governance/test_audit.py`.

**Interfaces:**
- `AuditEntry` frozen dataclass: `seq:int, timestamp:str, trace_id:str, kind:str, payload:dict, payload_hash:str, prev_hash:str, entry_hash:str`.
- `HashChainedAuditLog(path: Path)` — `append(kind:str, payload:dict, trace_id:str, timestamp:str|None=None) -> AuditEntry` (appends one JSONL line; entry_hash = sha256 of `prev_hash|seq|timestamp|trace_id|kind|payload_hash`, payload_hash = sha256 of canonical `json.dumps(payload, sort_keys=True)`); `entries() -> list[AuditEntry]`; `verify() -> bool` (recompute each hash + check prev linkage, seq monotonic; genesis prev_hash = 64 zeros); `reconstruct(trace_id) -> list[AuditEntry]`.

- [ ] Step 1–5: TDD tests (append→entries; verify True on a clean log; verify False after tampering a middle line's payload; reconstruct returns only the trace_id's entries; deterministic payload_hash) → implement → pass → commit `feat: hash-chained tamper-evident audit log (Article 12)`.

---

### Task 2: Wire the audit log into `orgpulse run`

**Files:** Modify `src/orgpulse/cli.py` (`write_run`); Test `tests/test_cli_run.py` (extend).

**Interfaces:** `write_run` also writes `out/audit.jsonl` via `HashChainedAuditLog`: one `kind="run"` entry (overall_severity, alert, generated_at, model, offline) and one `kind="recommendation"` entry per recommendation (payload = the rec dict + kpi + review_status; `trace_id` = `f"{run_id}:{kpi}:{i}"`; model = "offline-stub" or "claude-sonnet-5"; input hash = sha256 of the findings JSON). The log must `verify()` True after a run, and each recommendation is `reconstruct`-able by its trace_id.

- [ ] Step 1–5: extend tests (a run produces a verifying audit.jsonl with a run entry + one entry per recommendation, each carrying trace_id+timestamp+model+input hash) → implement → pass → commit.

---

### Task 3: EU AI Act self-classification (machine-readable + doc)

**Files:** Create `src/orgpulse/governance/classification.py`, `docs/EU_AI_ACT_SELF_CLASSIFICATION.md`; Test `tests/governance/test_classification.py`.

**Interfaces:** `classification.SELF_CLASSIFICATION` — a dict/dataclass: `risk_tier="high-risk"`, `annex="Annex III"`, `rationale`, `obligations` (map Article 11 technical docs / 12 logging / 14 human oversight → how OrgPulse meets each), `is_medical_device=False` (decision-support, not a device). `classification.as_markdown() -> str`. The Markdown doc is generated/kept in sync.

- [ ] Step 1–5: test (SELF_CLASSIFICATION.risk_tier == "high-risk"; obligations include Articles 11/12/14 mapped to the audit log + human gate; is_medical_device False; as_markdown contains "Annex III" + "human oversight") → implement + write the doc → pass → commit.

---

## Self-Review
- Article 12 (tamper-evident logging, reconstructable decisions) → Task 1 + 2. ✓
- Article 14 (human oversight) → already gated in Plan 4 recs (PENDING_HUMAN_REVIEW + disclaimer); referenced in the classification. ✓
- Article 11 (technical documentation) + Annex III self-classification → Task 3. ✓
- No new deps (stdlib hashlib/json). Deterministic, keyless, JSON-safe. Tamper-evidence is the core test.
