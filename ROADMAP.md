# OrgPulse AI — PharmaOps — Project Roadmap

> **Goal:** Autonomous pharma-ops intelligence platform — Hermes-scheduled multi-stream ingestion → anomaly detection → 30/60/90-day forecasting → GxP-aware, human-gated recommendations, built to the EU AI Act high-risk bar.
> **Started:** 2026-08-03
> **Constraints:** Solo build (Claude-assisted); public portfolio target (live Streamlit Cloud demo + README + architecture diagram + case-study writeup); no hard deadline.

---

## Status Overview

| Status       | Count |
|--------------|-------|
| ✅ Done       | 2     |
| 🔄 In Progress | 0   |
| ⏳ Pending    | 32    |
| 🚫 Blocked    | 0     |

**Total estimated time:** 75h (~9.4d)
**Elapsed time:** 2h
**Remaining estimate:** 73h (~9.1d)

---

## Stack & Tools

| Tool / Technology | Purpose | Introduced In |
|-------------------|---------|---------------|
| Python 3.11 | Runtime (Hermes venv target) | Phase 1 |
| uv | Dependency management | Phase 1 |
| numpy / pandas | Synthetic data + series math | Phase 1 |
| Typer | CLI (`orgpulse generate` / `run`) | Phase 1 |
| pytest | TDD test runner | Phase 1 |
| scikit-learn | Isolation Forest anomaly detection | Phase 2 |
| Prophet | Time-series forecasting + intervals | Phase 3 |
| LangGraph | 4-node analysis DAG orchestration | Phase 4 |
| Claude (Sonnet, configurable) | Domain-analyst + recommendation LLM nodes | Phase 4 |
| Hermes 0.19.1 | Autonomic loop: cron, baseline memory, tool orchestration | Phase 6 |
| Streamlit | Dashboard (split-plane presentation) | Phase 7 |
| Docker | Containerisation | Phase 7 |

---

## Phases

### Phase 1 — Scaffolding & Synthetic Generators (P0+P1)  ⏳

**Goal:** `orgpulse generate` produces realistic 3-stream data; tests prove injected structural breaks land where intended.
**Phase estimate:** 10h
**Plan:** `docs/superpowers/plans/2026-08-03-orgpulse-p1-generators.md`

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Project scaffolding & OrgPulseConfig | ✅ Done | 1h | 2026-08-03 00:00 | 2026-08-03 01:30 | |
| 2 | Shared generator primitives (clock, logistic, holiday dip) | ✅ Done | 1h | 2026-08-03 01:30 | 2026-08-03 02:30 | |
| 3 | Clinical-trial enrollment generator (hero) | ⏳ Pending | 3h | — | — | Injected: laggard site, amendment step, holiday dip |
| 4 | QC/lab generator (calibration drift, OOS clusters) | ⏳ Pending | 2h | — | — | |
| 5 | Supply-chain generator (sawtooth, spikes, seasonality) | ⏳ Pending | 2h | — | — | |
| 6 | `orgpulse generate` CLI | ⏳ Pending | 1h | — | — | |

### Phase 2 — Anomaly Detection (P2)  ⏳

**Goal:** Statistical + ML anomaly detection against remembered baselines, with Safe/Warning/Critical zoning.
**Phase estimate:** 8h

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | CUSUM detector (drift/shift) | ⏳ Pending | 3h | — | — | |
| 2 | Z-score detector (point outliers) | ⏳ Pending | 1h | — | — | |
| 3 | Isolation Forest detector (multivariate) | ⏳ Pending | 2h | — | — | |
| 4 | Safe/Warning/Critical zoning + baseline compare | ⏳ Pending | 2h | — | — | |

### Phase 3 — Forecasting (P3)  ⏳

**Goal:** Prophet forecasts (30/60/90d) with honest held-out backtest metrics + interval calibration on the hero stream.
**Phase estimate:** 8h

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Prophet forecaster (30/60/90d + intervals) | ⏳ Pending | 3h | — | — | |
| 2 | Rolling-origin backtest harness (MAPE/MASE) | ⏳ Pending | 3h | — | — | |
| 3 | Interval calibration check (90% coverage) | ⏳ Pending | 2h | — | — | |

### Phase 4 — LangGraph Analysis DAG (P4)  ⏳

**Goal:** 4-node DAG (detect → forecast → analyze → recommend) producing findings JSON + report + alert; offline mode works keyless.
**Phase estimate:** 13h

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | DAG scaffolding + shared state schema | ⏳ Pending | 2h | — | — | |
| 2 | Anomaly + forecast nodes (wire P2/P3) | ⏳ Pending | 2h | — | — | |
| 3 | Domain Analyst LLM node | ⏳ Pending | 2h | — | — | |
| 4 | Recommendation Writer node (CAPA/ICH E6(R3)) | ⏳ Pending | 3h | — | — | |
| 5 | Offline deterministic stub mode | ⏳ Pending | 2h | — | — | Keyless demo path |
| 6 | Findings JSON schema + report writer | ⏳ Pending | 2h | — | — | |

### Phase 5 — Governance & Compliance (P5)  ⏳

**Goal:** Real EU AI Act high-risk controls: hash-chained audit log, reconstructable decisions, human-oversight gating, self-classification doc.
**Phase estimate:** 7h

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Hash-chained audit log + trace reconstruction | ⏳ Pending | 3h | — | — | Article 12 |
| 2 | `review_status` + disclaimer schema enforcement | ⏳ Pending | 2h | — | — | Article 14 |
| 3 | EU AI Act self-classification doc (Annex III, Art 11) | ⏳ Pending | 2h | — | — | |

### Phase 6 — Hermes Integration (P6)  ⏳

**Goal:** Hermes runs the autonomous loop on cron, remembers baselines across runs, writes artifacts.
**Phase estimate:** 10h

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Author 3× domain `SKILL.md` (clinical/QC/supply) | ⏳ Pending | 3h | — | — | |
| 2 | Cron wiring (default daily 07:00) | ⏳ Pending | 2h | — | — | |
| 3 | Baseline memory across runs | ⏳ Pending | 3h | — | — | |
| 4 | Artifact writing (`/runs/<ts>`, `/runs/latest`) + alert | ⏳ Pending | 2h | — | — | |

### Phase 7 — Streamlit Dashboard & Seeding (P7)  ⏳

**Goal:** Always-populated live dashboard on Streamlit Cloud reading the latest published run.
**Phase estimate:** 10h

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Streamlit app: KPI cards, forecast bands, anomaly highlights | ⏳ Pending | 4h | — | — | |
| 2 | Audit-log / trace viewer + EU AI Act panel | ⏳ Pending | 2h | — | — | |
| 3 | Pre-seed 90+ day committed run history | ⏳ Pending | 2h | — | — | Keeps live demo alive |
| 4 | Streamlit Cloud deploy (lightweight deps) | ⏳ Pending | 2h | — | — | |

### Phase 8 — Story Layer (P8)  ⏳

**Goal:** The public-facing narrative: README, architecture diagram, case-study writeup, final verification.
**Phase estimate:** 9h

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Narrative README | ⏳ Pending | 2h | — | — | |
| 2 | Architecture diagram | ⏳ Pending | 2h | — | — | |
| 3 | Case-study writeup (problem→arch→EU AI Act→results) | ⏳ Pending | 3h | — | — | Postable |
| 4 | Final verification + code review pass | ⏳ Pending | 2h | — | — | |

---

## Change Log

> The roadmap above reflects the current plan. This log records every deviation from the original design — it is permanent and append-only.

| Date | Change | Reason | Impact | Original Plan |
|------|--------|--------|--------|---------------|
| 2026-08-03 02:10 | Clinical generator's protocol-amendment mechanism changed from "multiply daily rate ×1.6" to "raise enrollment ceiling ×1.6" | Plan defect found during Task 3: by the amendment day the logistic curve has plateaued, so daily increments are ~0 and multiplying stays ~0 — the intended post-amendment surge would not appear. Raising the ceiling re-opens a real surge and is more domain-accurate (amendment loosens inclusion criteria → larger addressable population). | Phase 1 Task 3 only; no interface/schema change, determinism preserved; plan doc + brief corrected to match. | Plan §Task 3 step-3 used `daily *= 1.6` after `PROTOCOL_AMENDMENT_DAY`. |
| 2026-08-03 03:05 | Amendment mechanism redesigned again: from "switch to higher-ceiling curve" to "new capacity ramps in over weeks via its own logistic"; amendment test hardened with a `max < 4×mean` cliff-guard | Final whole-branch review found the ceiling-switch dumped the entire capacity gap into a single 1,937-patient cliff day, and the 30-day-mean test passed only because that one outlier dominated the window — a luck-passing test on the hero feature. Redesign (not another patch) per root-cause policy; validated across seeds 42/7/123/2024 (sustained ~2× surge, busiest day ~1.6× window mean). | Phase 1 Task 3 only; no interface/schema change, determinism preserved; plan doc + brief + test corrected. | Plan §Task 3 used two logistic curves switched at day 120; test compared 30-day before/after means only. |
