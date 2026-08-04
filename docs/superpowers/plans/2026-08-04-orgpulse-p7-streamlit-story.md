# OrgPulse P7 — Streamlit Dashboard & Story Implementation Plan

> superpowers:subagent-driven-development. Checkbox steps.

**Goal:** The public-facing layer: a **self-contained, lightweight** Streamlit dashboard that reads the committed `runs/latest` artifacts (never importing the heavy pipeline), a **pre-seeded run history** so the live demo looks alive, and the narrative deliverables (README, architecture diagram, case-study writeup).

**Architecture:** `streamlit_app/` is standalone — `data.py` (pure loaders + an inline audit-chain verifier, no streamlit/orgpulse import) and `app.py` (Streamlit UI + Plotly forecast band). Its `requirements.txt` is only `streamlit, pandas, plotly` so Streamlit Cloud never installs Prophet/LangGraph (split-plane). A drift test asserts the inline verifier matches `orgpulse.governance.audit`. A seed script writes a handful of deterministic committed runs. README + `docs/ARCHITECTURE.md` (Mermaid) + `docs/CASE_STUDY.md` complete the story.

**Global Constraints:** Python 3.11; streamlit/plotly are an OPTIONAL `dashboard` dep group — the app itself imports NO `orgpulse` module (self-contained). Deterministic seed. No NaN in artifacts. No Co-Authored-By. TDD for `data.py`.

---

### Task 1: Dashboard data layer (`streamlit_app/data.py`)
**Files:** Create `streamlit_app/__init__.py`, `streamlit_app/data.py`; Test `tests/test_dashboard_data.py`.
**Interfaces:** `load_findings(run_dir)->dict`, `load_report(run_dir)->str`, `verify_audit(audit_path)->bool` (inline SHA-256 chain, no orgpulse import), `load_history(runs_root)->list[dict]` (per timestamped run: run_id, generated_at, overall_severity, alert). Pure Python (stdlib + json). 
- [ ] TDD: seed a run via `orgpulse tick` into tmp; assert load_findings has recommendations, verify_audit True on clean + False after tampering, and **verify_audit matches `orgpulse.governance.audit.HashChainedAuditLog.verify`** (anti-drift); load_history lists runs newest-first. Implement → pass → commit.

---

### Task 2: Streamlit app (`streamlit_app/app.py`) + requirements
**Files:** Create `streamlit_app/app.py`, `streamlit_app/requirements.txt`, `.streamlit/config.toml`.
**Interfaces:** `app.py` renders from `data.py`: header + overall severity/alert banner; KPI cards (severity-coloured Safe/Warning/Critical/Data-quality zoning); a Plotly forecast band (30/60/90 yhat + interval) for the hero; the recommendations panel (each with its CAPA frame + `PENDING_HUMAN_REVIEW` badge + disclaimer); an **audit-trail viewer** with a live "chain verified ✅/tamper detected ❌" check; and an **EU AI Act panel** (Annex III high-risk, Art. 11/12/14). Reads `runs/latest` by default (env/arg override). `requirements.txt` = streamlit, pandas, plotly only.
- [ ] Smoke: `python -c "import ast; ast.parse(open('streamlit_app/app.py').read())"` (syntax) and confirm `app.py` imports no `orgpulse`. Commit.

---

### Task 3: Seed committed run history
**Files:** Create `scripts/seed_runs.py`; commit `runs/` with ~6 deterministic runs + latest + baseline_memory.json.
- [ ] Run `scripts/seed_runs.py` (ticks 0..5, fixed dates, offline) → commit the `runs/` tree so the deployed dashboard is populated on first load. Verify `runs/latest` present + audit verifies.

---

### Task 4: Story — README, architecture diagram, case-study
**Files:** `README.md`, `docs/ARCHITECTURE.md` (Mermaid), `docs/CASE_STUDY.md`.
- [ ] Write the narrative README (the one-liner, quickstart `uv run orgpulse tick`, what it does, the EU AI Act + human-gating angle, the Streamlit link placeholder, test/architecture pointers); a Mermaid architecture diagram (Hermes → DAG → artifacts → Streamlit, governance cross-cut); and a postable case-study (problem → architecture → EU AI Act angle → honest results incl. MASE/coverage). Commit.

## Self-Review
- Live Streamlit demo reading runs/latest, lightweight deps → Tasks 1–2. ✓
- Pre-seeded committed run history → Task 3. ✓
- README + architecture diagram + case-study → Task 4. ✓
- Split-plane preserved (app imports no heavy stack; drift-tested). Deterministic seed. ✓
