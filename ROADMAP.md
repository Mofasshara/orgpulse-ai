# OrgPulse AI — PharmaOps — Project Roadmap

> **Goal:** Autonomous pharma-ops intelligence platform — Hermes-scheduled multi-stream ingestion → anomaly detection → 30/60/90-day forecasting → GxP-aware, human-gated recommendations, built to the EU AI Act high-risk bar.
> **Started:** 2026-08-03
> **Constraints:** Solo build (Claude-assisted); public portfolio target (live Streamlit Cloud demo + README + architecture diagram + case-study writeup); no hard deadline.

---

## Status Overview

| Status       | Count |
|--------------|-------|
| ✅ Done       | 36    |
| 🔄 In Progress | 0   |
| ⏳ Pending    | 8     |
| 🚫 Blocked    | 0     |

**Total estimated time:** 86h (~10.8d)
**Elapsed time:** 67h (Phases 1–6 complete)
**Remaining estimate:** 19h (~2.4d)

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

### Phase 1 — Scaffolding & Synthetic Generators (P0+P1)  ✅

**Goal:** `orgpulse generate` produces realistic 3-stream data; tests prove injected structural breaks land where intended.
**Phase estimate:** 10h
**Plan:** `docs/superpowers/plans/2026-08-03-orgpulse-p1-generators.md`

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Project scaffolding & OrgPulseConfig | ✅ Done | 1h | 2026-08-03 00:00 | 2026-08-03 01:30 | |
| 2 | Shared generator primitives (clock, logistic, holiday dip) | ✅ Done | 1h | 2026-08-03 01:30 | 2026-08-03 02:30 | |
| 3 | Clinical-trial enrollment generator (hero) | ✅ Done | 3h | 2026-08-03 02:30 | 2026-08-03 03:10 | Amendment redesigned twice (multiplier→ceiling→ramp-in surge); sustained surge, cliff-guarded test |
| 4 | QC/lab generator (calibration drift, drift-dependent OOS incidence) | ✅ Done | 2h | 2026-08-03 02:40 | 2026-08-03 02:55 | Signed-drift test strengthened |
| 5 | Supply-chain generator (sawtooth, spikes, seasonality) | ✅ Done | 2h | 2026-08-03 02:55 | 2026-08-03 03:05 | Seasonality test hardened (correlation) |
| 6 | `orgpulse generate` CLI | ✅ Done | 1h | 2026-08-03 03:05 | 2026-08-03 03:15 | 24/24 suite green; smoke run OK |

### Phase 2 — Anomaly Detection (P2)  ✅

**Goal:** Statistical + ML anomaly detection against remembered baselines, with Safe/Warning/Critical zoning.
**Phase estimate:** 8h
**Plan:** `docs/superpowers/plans/2026-08-03-orgpulse-p2-detection.md`

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Detection types + Baseline; Z-score detector | ✅ Done | 2h | 2026-08-03 03:20 | 2026-08-03 03:35 | Severity IntEnum, AnomalyResult |
| 2 | CUSUM detector (drift/shift) | ✅ Done | 3h | 2026-08-03 03:35 | 2026-08-03 03:45 | Thresholds corrected to 2h/h (20-seed validated) |
| 3 | Isolation Forest detector (multivariate) | ✅ Done | 2h | 2026-08-03 03:45 | 2026-08-03 03:55 | Built; wiring into orchestrator deferred to Plan 4 |
| 4 | Orchestrator + Safe/Warning/Critical zoning + generator integration tests | ✅ Done | 2h | 2026-08-03 03:55 | 2026-08-03 04:05 | Integration proves detectors catch injected breaks (margins 536/10, 57.8/10) |

### Phase 2.5 — Hardening (Codex findings)  ✅

**Goal:** Fix must-fix defects from the Codex adversarial review so the foundation is trustworthy before forecasting. Mutation-resistant + multi-seed acceptance tests.
**Phase estimate:** 6h
**Plan:** `docs/superpowers/plans/2026-08-03-orgpulse-p2.5-hardening.md`

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Data-quality guarding (finite/1-D validation; corrupt data → DATA_QUALITY, never SAFE) | ✅ Done | 1h | 2026-08-03 05:10 | 2026-08-03 05:25 | Fix-loop: corrupt-baseline SAFE-leak also closed |
| 2 | Clinical intensity/rate model (fixes inert holiday; keeps amendment surge) | ✅ Done | 2h | 2026-08-03 05:25 | 2026-08-03 05:40 | Holiday now real (ratio 0.39); mutation-resistant |
| 3 | CUSUM severity calibrated to window-length false-alarm target | ✅ Done | 2h | 2026-08-03 05:40 | 2026-08-03 05:55 | In-control CRIT 0.5–1.8%; power 99% |
| 4 | Harden detection integration tests (assert direction; bound false-alarm rate) | ✅ Done | 0.5h | 2026-08-03 05:55 | 2026-08-03 06:05 | Fix-loop: pass_rate not calm → screen_fail_rate |
| 5 | Hygiene: honest OOS label, CLI day bounds, pytest dev-group | ✅ Done | 0.5h | 2026-08-03 06:05 | 2026-08-03 06:15 | |

### Phase 2.6 — Robustness Completion (Codex round 2)  ✅

**Goal:** Close the "bad data never SAFE / never crashes" contract completely + make CUSUM false-alarm calibration valid for any usable baseline. All gaps Codex-found + controller-verified.
**Phase estimate:** 5h
**Plan:** `docs/superpowers/plans/2026-08-03-orgpulse-p2.6-robustness.md`

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Scalar-detector DQ contract (n≥1, param+overflow guards, score None) | ✅ Done | 1h | 2026-08-03 06:40 | 2026-08-03 06:50 | JSON-serializable DATA_QUALITY |
| 2 | detect_kpi validates history/recent; empty results → DATA_QUALITY | ✅ Done | 0.5h | 2026-08-03 06:50 | 2026-08-03 06:58 | Codex BLOCK closed; never raises |
| 3 | CUSUM min baseline + recalibrate at worst-case + window-dependency spy test | ✅ Done | 2h | 2026-08-03 06:58 | 2026-08-03 07:15 | n=5→DATA_QUALITY; false-alarm 0.5–1.2% across windows |
| 4 | Isolation Forest finite/shape validation → DATA_QUALITY | ✅ Done | 0.5h | 2026-08-03 07:15 | 2026-08-03 07:22 | jagged/param edge → Plan 4 |
| 5 | Generator arg validation + honest OOS relabel + dedupe dev deps | ✅ Done | 1h | 2026-08-03 07:22 | 2026-08-03 07:35 | |

### Phase 3 — Forecasting (P3)  ✅

**Goal:** Prophet forecasts (30/60/90d) with honest held-out backtest metrics + interval calibration on the hero stream.
**Phase estimate:** 8h
**Plan:** `docs/superpowers/plans/2026-08-03-orgpulse-p3-forecasting.md`

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Prophet forecaster (30/60/90d + intervals) | ✅ Done | 3h | 2026-08-03 07:50 | 2026-08-03 08:00 | Deterministic; lazy import; compute-plane only |
| 2 | Rolling-origin backtest harness (MAPE/MASE) | ✅ Done | 3h | 2026-08-03 08:00 | 2026-08-03 08:12 | Honest: MAPE ~17%, MASE ~1.4 (no leakage) |
| 3 | Interval calibration check (90% coverage) | ✅ Done | 2h | 2026-08-03 08:12 | 2026-08-04 08:25 | Detects overconfidence (0.72→0.88); in-sample disclosed |

### Phase 4 — LangGraph Analysis DAG (P4)  ✅

**Goal:** 4-node DAG (detect → forecast → analyze → recommend) producing findings JSON + report + alert; offline mode works keyless.
**Phase estimate:** 13h
**Plan:** `docs/superpowers/plans/2026-08-04-orgpulse-p4-langgraph-dag.md`

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Pipeline state + findings schema (human-gated, JSON-safe) | ✅ Done | 2h | 2026-08-04 07:40 | 2026-08-04 07:50 | |
| 2 | LLM client interface + OfflineLLM stub + Claude(sonnet-5) adapter | ✅ Done | 2h | 2026-08-04 07:50 | 2026-08-04 08:00 | Keyless default |
| 3 | Detect + forecast nodes (wire P2/P3) | ✅ Done | 2h | 2026-08-04 08:00 | 2026-08-04 08:15 | Driving-detector detail; RNG localized |
| 4 | Analyst + Recommendation LLM nodes (CAPA/ICH E6(R3)) | ✅ Done | 3h | 2026-08-04 08:15 | 2026-08-04 08:30 | DATA_QUALITY handled distinctly |
| 5 | Assemble LangGraph DAG + findings/report/alert | ✅ Done | 2h | 2026-08-04 08:30 | 2026-08-04 08:45 | Offline-deterministic end-to-end |
| 6 | `orgpulse run` CLI (findings.json + report.md) | ✅ Done | 2h | 2026-08-04 08:45 | 2026-08-04 13:05 | Final review fixed DATA_QUALITY dominance/alert |

### Phase 5 — Governance & Compliance (P5)  ✅

**Goal:** Real EU AI Act high-risk controls: hash-chained audit log, reconstructable decisions, human-oversight gating, self-classification doc.
**Phase estimate:** 7h
**Plan:** `docs/superpowers/plans/2026-08-04-orgpulse-p5-governance.md`

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Hash-chained audit log + trace reconstruction | ✅ Done | 3h | 2026-08-04 13:10 | 2026-08-04 13:15 | Article 12; tamper-evidence verified end-to-end |
| 2 | Wire audit log into orgpulse run (each rec reconstructable) | ✅ Done | 2h | 2026-08-04 13:15 | 2026-08-04 13:18 | trace_id + input hash + model |
| 3 | EU AI Act self-classification (machine-readable + doc) | ✅ Done | 2h | 2026-08-04 13:18 | 2026-08-04 13:22 | Annex III; Arts 9/11/12/13/14. review_status/disclaimer gating from Plan 4 |

### Phase 6 — Hermes Integration (P6)  ✅

**Goal:** Hermes runs the autonomous loop on cron, remembers baselines across runs, writes artifacts.
**Phase estimate:** 10h
**Plan:** `docs/superpowers/plans/2026-08-04-orgpulse-p6-hermes.md`

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Baseline memory across runs (BaselineMemory) | ✅ Done | 3h | 2026-08-04 13:25 | 2026-08-04 13:28 | JSON per-KPI baseline, cross-run state |
| 2 | Memory-backed tick + artifact writing (`runs/<ts>`, `runs/latest`) + alert | ✅ Done | 3h | 2026-08-04 13:28 | 2026-08-04 13:32 | Detects current vs remembered baseline; `orgpulse tick` CLI |
| 3 | 3× domain `SKILL.md` (clinical/QC/supply) + cron runbook (daily 07:00) | ✅ Done | 3h | 2026-08-04 13:32 | 2026-08-04 13:36 | Authentic Hermes frontmatter; split-plane |
| 4 | (folded into 2) end-to-end verified: bootstrap→remembered, audit verifies | ✅ Done | 1h | 2026-08-04 13:36 | 2026-08-04 13:38 | |

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
| 2026-08-03 03:40 | CUSUM severity thresholds changed from (CRITICAL≥h, WARNING≥0.6h) to (CRITICAL≥2h, WARNING≥h) | Implementer hit a genuine BLOCK: a stable in-control series peaks up to ~9 over a 60-sample window, so 0.6h=3.0 false-warned. Empirically validated (20 seeds): stable always <2h; any ≥1σ shift always >15. Threshold correction, not a code/interface change. | Phase 2 Task 3 only; tests unchanged (they pass under corrected mapping). | Plan §P2 Task 3 mapped CRITICAL at h and WARNING at 0.6h. |
| 2026-08-03 06:30 | Added **Phase 2.6 — Robustness Completion** after a 2nd Codex review | Codex re-review: 3 findings fully resolved, but the "bad data never SAFE/crash" contract had holes (n=0/empty-list→SAFE, empty/2-D history→crash, nan params→SAFE, IsolationForest NaN→SAFE) and CUSUM calibration collapsed on short baselines (n=5→46% false-alarm; verified). Recalibrated CUSUM at the worst-case minimum baseline (n=60) + enforce minimum; complete DQ contract. | 5 tasks before Phase 3. | Plan 2.5 fixed specific cases, not the general contract; CUSUM table assumed ~90-pt baseline undocumented. |
| 2026-08-03 05:00 | Added **Phase 2.5 — Hardening** after a Codex adversarial review found must-fix defects in Plans 1–2 | Codex mutation-tested and multi-seed-validated; found (a) clinical holiday-dip inert (0/2920 rows change), (b) amendment integration test fires on the OPPOSITE signal, (c) detectors return SAFE on NaN/empty data, (d) CUSUM 2h/h uncalibrated & window-length-dependent (in-control CRITICAL up to 3.6% at n=365 — my "20-seed validated" claim was too thin), (e) calm-QC test seed-selected, (f) OOS mislabeled as clustered. My internal reviews measured margins, not mutations. | Inserts 5 hardening tasks before Phase 3; Isolation Forest wiring + 0.10 cutoff deferred to Phase 4 (claim corrected: P2 is NOT a complete 3-detector stack). | Plans 1–2 were treated as a complete, trustworthy foundation. |
