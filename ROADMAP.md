# OrgPulse AI — PharmaOps — Project Roadmap

> **Goal:** Autonomous pharma-ops intelligence platform — Hermes-scheduled multi-stream ingestion → anomaly detection → 30/60/90-day forecasting → GxP-aware, human-gated recommendations, built to the EU AI Act high-risk bar.
> **Started:** 2026-08-03
> **Constraints:** Solo build (Claude-assisted); public portfolio target (live Streamlit Cloud demo + README + architecture diagram + case-study writeup); no hard deadline.

---

## Status Overview

| Status       | Count |
|--------------|-------|
| ✅ Done       | 44    |
| 🔄 In Progress | 0   |
| ⏳ Pending    | 0     |
| 🚫 Blocked    | 0     |

**Total estimated time:** 86h (~10.8d)
**Elapsed time:** 86h (all 8 phases complete)
**Remaining estimate:** 0h — build complete; remaining action is user-side Streamlit Cloud deploy

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

### Phase 7 — Streamlit Dashboard & Seeding (P7)  ✅

**Goal:** Always-populated live dashboard on Streamlit Cloud reading the latest published run.
**Phase estimate:** 10h
**Plan:** `docs/superpowers/plans/2026-08-04-orgpulse-p7-streamlit-story.md`

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Streamlit app: KPI zoning, forecast fan, anomaly highlights | ✅ Done | 4h | 2026-08-04 13:40 | 2026-08-04 13:44 | Self-contained; AppTest-verified |
| 2 | Audit-log / trace viewer + EU AI Act panel + live chain verify | ✅ Done | 2h | 2026-08-04 13:44 | 2026-08-04 13:45 | Drift-tested inline verifier |
| 3 | Pre-seed committed run history (6 monthly ticks, deterministic) | ✅ Done | 2h | 2026-08-04 13:46 | 2026-08-04 13:52 | Byte-identical reproduction |
| 4 | Deploy-ready (lightweight `requirements.txt`, config, committed `runs/`) | ✅ Done | 2h | 2026-08-04 13:52 | 2026-08-04 13:55 | User connects repo to Streamlit Cloud |

### Phase 8 — Story Layer (P8)  ✅

**Goal:** The public-facing narrative: README, architecture diagram, case-study writeup, final verification.
**Phase estimate:** 9h

| # | Task | Status | Estimate | Started | Completed | Notes |
|---|------|--------|----------|---------|-----------|-------|
| 1 | Narrative README | ✅ Done | 2h | 2026-08-04 13:56 | 2026-08-04 13:58 | |
| 2 | Architecture diagram (Mermaid, split-plane) | ✅ Done | 2h | 2026-08-04 13:58 | 2026-08-04 14:00 | `docs/ARCHITECTURE.md` |
| 3 | Case-study writeup (problem→arch→EU AI Act→results) | ✅ Done | 3h | 2026-08-04 14:00 | 2026-08-04 14:02 | `docs/CASE_STUDY.md` |
| 4 | Final verification + code review pass | ✅ Done | 2h | 2026-08-04 14:02 | 2026-08-04 14:10 | Found+fixed 2 root-cause defects |

---

## Change Log

> The roadmap above reflects the current plan. This log records every deviation from the original design — it is permanent and append-only.

| Date | Change | Reason | Impact | Original Plan |
|------|--------|--------|--------|---------------|
| 2026-08-08 | **Core/Domain refactor — 2nd Codex review completion (NOT-SAFE-TO-MERGE → resolved)** | Codex's re-review confirmed pharma byte-identical but found 4 more couplings + a WEAK proof test I'd mislabelled "proven". Fixed: (1) `run_pipeline` now takes a generic streams dict (was pharma-only → run-vs-tick divergence); (2) `forecast_node` validates the hero-frame DATES (NaT/missing/duplicate ds → no forecast, no Prophet crash); (3) the DQ recommendation text is domain-provided + the anomaly LLM prompt is domain-neutral (was hardcoded "CAPA"); (4) `verify_run` rejects multi-key/ambiguous forecasts; + `Domain` validates its callables. **Crucially, rebuilt the second-domain suite to MUTATION-CONSTRAIN every repaired invariant — reverting any fix now fails a test (verified by 6 mutation runs); the earlier "proof" didn't and the docs overclaimed it.** Docs corrected + 2 residuals documented (mutable/unhashable Domain dicts; the generate() stable-order contract). | tests 169→180; pharma byte-identical throughout; the engine is now genuinely domain-agnostic (mutation-proven). | The first completion (r10) fixed the obvious couplings but left run_pipeline pharma-specific, hero-date validation missing, recommendation text pharma, and a non-constraining proof test. |
| 2026-08-07 | **Core/Domain refactor — completion after Codex review (NOT-SAFE-TO-MERGE)** | Codex confirmed pharma byte-identical + tests strong, but found the "domain-agnostic" claim was an OVERCLAIM: a second domain still hit 5 pharma-specific spots. Fixed all: (1) graph state carries a generic `streams` key (was fixed to clinical/qc/supply, so LangGraph dropped a banking stream); (2) analyst/recommend use `domain.hero()` + offline framing is now domain-provided (`offline_framing`/`offline_recommend`); (3) detect/forecast PARITY enforced (forecast defers to the hero's detection finding); (4) `verify_run` validates whichever hero the dashboard renders (rejecting metadata-only forecasts); (5) `Domain.__post_init__` rejects unsafe configs (NaN/inverted bounds, non-enum mode). Added `tests/core/test_second_domain.py` — a non-pharma domain runs the full stack (tick/forecast/verify/offline/graph) proving domain-agnosticism. Docs corrected: engine is domain-agnostic (proven), with honest residuals (`capa_frame` label + `classification.py` compliance content are domain framing a new domain refines). | tests 167→169; pharma byte-identical throughout (golden-master gated every step); the overclaim is fixed. | The initial refactor (Tasks 1–7) preserved pharma but left the engine partly pharma-coupled; this completes it so banking = a new Domain, not engine edits. |
| 2026-08-07 | **Core/Domain refactor** — extracted a domain-agnostic engine + `Domain` interface so a second regulated domain (banking) can be added without touching the engine (branch `refactor/core-domain`) | The "configurable platform" framing was aspirational — the domain was baked into ~6 modules. Refactor makes it real & honest. Behavior-preserving: golden-master gate (162 tests + byte-identical `runs/`) held after every task. Added `core/domain.py` (Domain/KpiSpec/DetectionMode), `domains/pharma/`; parameterized detect/forecast/pipeline/agent/dashboard by `domain=PHARMA`. Detection-mode dispatch present (TIMESERIES wired; THRESHOLD/DATEMATH data-model only). **Deferred to the banking build (deliberately, to stay behavior-preserving): the physical `core/` folder move (cosmetic), and parameterizing `render_report`/`classification` human-label text (banking supplies its own framing).** | tests 162→167 (+ Domain/pharma tests); pharma output byte-identical; live demo unaffected. | Plan: 7 tasks, golden-master gated. Tasks 1–6 done; Task 7 physical move deferred (docs done). |
| 2026-08-06 11:00 | **Review round 9 (Codex, NO-GO — 0 Critical + 3 Important, narrow tail)** — controller-fixed (not handed to Codex) | Convergence: 2nd round with no Criticals; findings now "hand-edited memory file / exotic malformed artifact" territory. (1) run/tick min-history PARITY still failed for a supported non-default config (base_days=60/baseline_days=60 → tick SAFE, run DATA_QUALITY) → introduced `MIN_ASSESSMENT_DAYS=120` as ONE shared contract; tick baseline_days floor raised 60→120 so the divergent config is rejected. (2) corrupt remembered baseline std/n still SAFE (std=100 for pass_rate flattens z-scores; n=121/1e9) → `baseline_domain_feasible` checks mean+std feasibility, tick also requires n==baseline_days. (3) verifier schema ignored non-finite numbers + calibration metadata → `_is_num` now requires finite; schema validates calibration object, severity enum, alert bool. Codex CONFIRMED HELD (huge list): numeric-string agg, both-path parity on all prior corruptions, nested schema cases, recommendation strictness, stats, determinism, split-plane. | 3 findings + doc counts across pipeline/agent/dashboard/docs; tests 157→162; runs re-seeded; determinism holds. | Round 8 fixed the literal reproductions but left run/tick parity for non-default params, baseline std/n unvalidated, and finite/calibration/enum schema gaps. |
| 2026-08-06 09:30 | **Review round 8 (Codex, NO-GO — 1 Critical + 4 Important, all nested edge cases)** (branch `fix/domain-and-lineage` reused workflow) | Convergence continues (huge "held" list; findings now nested, not fail-open-by-default). Critical: `validated_clinical_daily` coerced values for the domain check but aggregated the RAW string column → pandas concatenated `"1"×8 → "11111111"` (wrong-dtype fail-open) → now aggregates coerced numerics (→8.0), non-numeric → DATA_QUALITY. Importants: (a) run/tick divergence — tick required only `days_per_period` where run requires 120 → tick now requires `baseline_days` + param validation `base_days >= max(baseline_days, days_per_period)`; (b) corrupt remembered baseline (pass_rate mean 1.1) drove SAFE → baseline mean domain-validated; (c) verify_run validated only top-level types → now validates full nested findings schema (kpi_findings/forecast horizons/recommendations members) + load_history coerces generated_at to str (sort crash); (d) recommendation_node coerced malformed LLM values / accepted invented KPIs → now requires requested-KPI + non-empty string fields, else skips. Codex CONFIRMED HELD: centralized clinical guard, both-path parity on prior corruptions, stats, CUSUM ~1%, split-plane, determinism, all lineage/hash bindings. | 6 findings across pipeline/agent/dashboard/docs; tests 152→157; runs re-seeded; determinism holds. | Round 7 centralization + self-review still left string-dtype aggregation, run/tick min-length divergence, unvalidated remembered baselines, and shallow verify_run schema. |
| 2026-08-05 15:20 | **Controller self-review (no external reviewer)** — found a fail-open SAFE in the autonomous tick + 2 minors | Reviewed the whole project directly. NEW defect the 7 external rounds missed because they tested the `run`/detect path, not the tick: `run_tick._kpi_series` built the clinical series with the raw `series_from_generator`, bypassing `validated_clinical_daily` — so per-site clinical corruption masked by the daily sum (negative site total in a positive aggregate, duplicate rows, NaT dates, non-integer counts) read SAFE in the SCHEDULED agent (the primary path). Fixed: the tick now uses the same validated builder; structural failure → DATA_QUALITY. Every raw KPI-series consumer now shares one gate. Also: analyst_node leaked the `calibration` meta key as a fake forecast horizon in the LLM prompt; recommendation_node crashed on malformed online-LLM output (now skips). Verified NOT regressions: generators in-domain across 150 seed×horizon combos, committed runs byte-identical, determinism holds. | 3 fixes across agent/pipeline; tests 150→152; runs unchanged. | Rounds 1-7 hardened the `run`/detect path and the gate; the tick's clinical extraction still bypassed the validated builder. |
| 2026-08-05 14:00 | **Review round 7 (Codex, NO-GO — no Criticals) → CENTRALIZE validation** (branch `fix/centralize-validation`) | R7 confirmed the detection boundary held completely but found 6 Importants that were the SAME class in new consumers: the forecast path, the baseline-save path, and the verifier each re-validated independently and missed cases. Root fix = centralization: `kpi_series_reason()` is now the single usable-series check used by detection + forecasting + tick; `validated_clinical_daily()` validates RAW rows before aggregation (per-site negatives no longer masked by the sum; NaT/duplicate/non-integer counts rejected) and is shared by detect + forecast so an out-of-domain hero never forecasts (#1,#2); run_tick validates the whole series and never saves a corrupt baseline (#3); verify_run validates the findings schema (#4), REQUIRES lineage fields (#6), and routes all reads through a safe reader (directory/invalid-UTF-8 fail closed, incl. history; both verifiers catch UnicodeError) (#5); the "regenerable" claim narrowed (offline deterministic vs online not bit-reproducible). Codex CONFIRMED HELD: detection gate, stats, CUSUM ~1%, split-plane, determinism. | 8 findings across pipeline/agent/dashboard/governance/docs; tests 145→150; runs re-seeded; determinism holds. | Round 6 completed detection's domain gate but the OTHER series-consumers (forecast, baseline-save, verify) still re-validated incompletely. |
| 2026-08-05 12:15 | **Review round 6 (Codex, NO-GO) — completed the redesign at its boundaries** (branch `fix/domain-and-lineage`) | R6 confirmed the round-5 redesign HELD (prior fail-open cases fixed, large "held" list) but found 1 Critical + 5 Importants that were completeness gaps in the same boundaries: (1) the fail-safe gate validated shape but not DOMAIN — impossible values (pass_rate=1.5, enrollment=-1, lead_time=-5) and duplicate (date,site_id) clinical rows read SAFE → added per-KPI domain bounds + uniqueness at the gate AND in run_tick; (2) run_tick crashed on wrong-dtype columns → guarded; (3) dashboard parsed findings.json before verifying → verify-before-parse; (4) empty/missing audit logs "verified" and report_hash was optional → both closed; (5) tick lineage omitted the remembered baseline that drives detection → added baseline_input_hash so remembered-baseline decisions are regenerable; (6-7) docs aligned to the new audit schema + stale counts refreshed. Codex CONFIRMED HELD: all round-5 fixes, stats, CUSUM ~1%, split-plane, determinism. | 7 findings across pipeline/agent/dashboard/governance/docs; tests 141→145; runs re-seeded (new baseline_input_hash); determinism holds. | Round 5 fixed the fail-open STRUCTURE but the gate lacked domain constraints, two crash-guards, and the audit omitted baseline lineage. |
| 2026-08-05 10:30 | **Review round 5 (Codex, NO-GO) → root-cause REDESIGN of the fail-safe + trust boundaries** (branch `fix/failsafe-redesign`) | R5 found 3 Criticals + 4 Importants — all on the SAME two guarantees that rounds 1-4 had "fixed" piecemeal: "bad data never reads SAFE" and "unverified data is withheld". Diagnosis: those invariants were enforced ad hoc (by omission and by SAFE-defaults) at many scattered points, so each patch plugged one hole and left others. Redesigned to enforce each at a SINGLE boundary: (A) a fail-safe gate — every KPI ALWAYS yields a finding, default DATA_QUALITY, SAFE only when earned by detection (fixes partial-NaN masking via any-NaN→NaN aggregation; silently-omitted insufficient KPI; and run_tick's bootstrap SAFE-without-validation); (B) dashboard verifies FIRST before any status renders, verifies EVERY historical run (forged→UNVERIFIED, malformed→no crash), and verify_run validates shapes before indexing; (C) audit records real input LINEAGE (seed+params+input_data_hash) + output hashes (findings+report), renames the misleading input_hash→findings_hash, binds report.md, and the Art.12 compliance text is rewritten honestly. **This was a redesign, not a 5th patch — per the standing "2nd+ failure on the same guarantee = redesign" rule I had violated 3×.** | 9 findings across forecast/pipeline/agent/dashboard/governance/docs; tests 127→141; committed runs re-seeded to new audit schema; determinism holds. Codex CONFIRMED HELD: stats (MASE≈1.4/MAPE≈17%/cov≈70%→90%), CUSUM ~1%, split-plane, determinism, 134 baseline tests. | Rounds 1-4 treated each fail-open/integrity gap as an isolated bug and patched it; the invariant was never enforced at one place. |
| 2026-08-04 16:45 | **Adversarial review round 4 (Claude)** — fixed 2 Criticals + 6 Importants + 3 Minors (branch `fix/review-r4`) | R4 found two fail-open SAFE paths that falsified the central trust claim: (1) an all-NaN enrollment stream aggregated to 0.0 (groupby sum min_count=0) and passed SAFE → fixed to min_count=1 so it surfaces DATA_QUALITY + forecast guards non-finite series; (2) the dashboard rendered forged findings as a green all-clear even when verify_run failed → now fails closed (error + status withheld + st.stop). Plus audit-integrity gaps: tail-truncation of recommendation records still verified → verify_run binds the logged rec count to findings; verifiers didn't fail closed on all malformed logs (list/scalar/bad-JSON) → both hardened; `orgpulse run` twice into one dir broke binding → clears audit per write; CUSUM accepted any k but used the k=0.5 table → rejects non-0.5 k; corrupt baseline memory crashed the tick → clear fail; 2-D/schema-corrupt pipeline inputs crashed → guarded to DATA_QUALITY. Minors: 95%→90% CI label, backtest param validation, run_tick int-type checks. Docs de-overclaimed (synthetic data, hero-only forecast, tamper-EVIDENT not -proof, deploy-ready not deployed). Codex/Claude CONFIRMED HELD: headline stats (MASE≈1.4/MAPE≈17%/raw≈70%/adj≈90%), split-plane, determinism, 127 baseline tests. | 11 findings across forecast/detect/pipeline/governance/agent/cli/dashboard/docs; tests 127→134; seed reproduces byte-identically. | Plan 7 + R3 shipped with 2 fail-open SAFE paths, 4 audit-integrity gaps, and several unguarded crash/param edge cases. |
| 2026-08-04 15:30 | **Codex adversarial review round 3** — fixed 3 Criticals + 5 Importants (branch `fix/codex-r3`) | R3 found 3 false portfolio claims + real bugs: (1) "calibrated 90% interval" was never applied — forecast_node copied raw Prophet bounds → wired the in-sample calibration factor onto published bounds + test; (2) dashboard showed forged `findings.json` as SAFE under "chain verified" → `verify_run` now binds findings bytes to the audited `input_hash` + fails closed on malformed logs; (3) Isolation Forest claimed as deployed but never wired → relabeled built-but-unwired; (4) tick baseline fix failed outside default params → entry validation + never persist sub-60 baseline; (5) same-second ticks collided/destroyed a durable run → tick-numbered dirs + refuse-overwrite; (6) dashboard defaulted unknown severity to SAFE → DATA_QUALITY fail-safe; (7) seed rmtree'd shared runs/ → deletes only its own dirs; docs: honest audit threat-model + per-tick-realization reframe. Codex confirmed HELD: verifier equivalence, split-plane, byte-reproduction, multi-seed story, MASE/MAPE/CUSUM numbers, 117 offline tests. | 8 findings across pipeline/agent/dashboard/docs; tests 117→127; seed reproduces byte-identically; user chose wire-calibration / relabel-IF / reframe-world. | Plan 7 shipped with calibration & IF unwired but claimed active, an audit not bound to findings, and tick params/seed/severity edge cases unguarded. |
| 2026-08-04 14:08 | `run_tick` now clears each run dir's audit log on write so `runs/latest` mirrors exactly one run | Final Plan-7 review: `_write` appended to `latest/audit.jsonl`, so it accumulated all 6 seeded ticks (19 entries) while `findings.json`/`report.md` showed only the latest — the dashboard's audit panel would show March-dated rows beside an August KPI board. Durable per-run records already live in the timestamped dirs. Regression-locked. | Phase 6 `agent/tick.py` + test; no interface change. | Plan 6 `_write` appended to `latest/audit.jsonl` unconditionally. |
| 2026-08-04 13:50 | `run_tick` baseline window decoupled from detection window (fit baseline on trailing `baseline_days=120`, detect last `days_per_period=30`) | Seeding the demo exposed a Plan-6 defect: baseline was fit on the 30-sample detection window, below CUSUM's 60-sample floor, so every post-bootstrap tick silently degraded to `DATA_QUALITY`. Root-cause fix (not a patch): established-history baseline stays ≥60 samples. Locked with a no-`DATA_QUALITY`-on-remembered-tick regression test. | Phase 6 `agent/tick.py` + test; new `baseline_days` param, default preserves behavior for callers. | Plan 6 fit the next baseline on the just-detected 30-day period. |
| 2026-08-04 13:38 | Plan 7 delivered Phases 7 **and** 8 together (dashboard + seeding + README/diagram/case-study + final review) | Presentation and story layers are one coherent public-facing deliverable and were built in a single plan; controller-direct implementation (subagents hitting session limits, as in Plans 5–6). | Phases 7–8 complete; project build finished. Remaining action is user-side Streamlit Cloud deploy. | Roadmap listed Phases 7 and 8 as separate phases. |
| 2026-08-03 02:10 | Clinical generator's protocol-amendment mechanism changed from "multiply daily rate ×1.6" to "raise enrollment ceiling ×1.6" | Plan defect found during Task 3: by the amendment day the logistic curve has plateaued, so daily increments are ~0 and multiplying stays ~0 — the intended post-amendment surge would not appear. Raising the ceiling re-opens a real surge and is more domain-accurate (amendment loosens inclusion criteria → larger addressable population). | Phase 1 Task 3 only; no interface/schema change, determinism preserved; plan doc + brief corrected to match. | Plan §Task 3 step-3 used `daily *= 1.6` after `PROTOCOL_AMENDMENT_DAY`. |
| 2026-08-03 03:05 | Amendment mechanism redesigned again: from "switch to higher-ceiling curve" to "new capacity ramps in over weeks via its own logistic"; amendment test hardened with a `max < 4×mean` cliff-guard | Final whole-branch review found the ceiling-switch dumped the entire capacity gap into a single 1,937-patient cliff day, and the 30-day-mean test passed only because that one outlier dominated the window — a luck-passing test on the hero feature. Redesign (not another patch) per root-cause policy; validated across seeds 42/7/123/2024 (sustained ~2× surge, busiest day ~1.6× window mean). | Phase 1 Task 3 only; no interface/schema change, determinism preserved; plan doc + brief + test corrected. | Plan §Task 3 used two logistic curves switched at day 120; test compared 30-day before/after means only. |
| 2026-08-03 03:40 | CUSUM severity thresholds changed from (CRITICAL≥h, WARNING≥0.6h) to (CRITICAL≥2h, WARNING≥h) | Implementer hit a genuine BLOCK: a stable in-control series peaks up to ~9 over a 60-sample window, so 0.6h=3.0 false-warned. Empirically validated (20 seeds): stable always <2h; any ≥1σ shift always >15. Threshold correction, not a code/interface change. | Phase 2 Task 3 only; tests unchanged (they pass under corrected mapping). | Plan §P2 Task 3 mapped CRITICAL at h and WARNING at 0.6h. |
| 2026-08-03 06:30 | Added **Phase 2.6 — Robustness Completion** after a 2nd Codex review | Codex re-review: 3 findings fully resolved, but the "bad data never SAFE/crash" contract had holes (n=0/empty-list→SAFE, empty/2-D history→crash, nan params→SAFE, IsolationForest NaN→SAFE) and CUSUM calibration collapsed on short baselines (n=5→46% false-alarm; verified). Recalibrated CUSUM at the worst-case minimum baseline (n=60) + enforce minimum; complete DQ contract. | 5 tasks before Phase 3. | Plan 2.5 fixed specific cases, not the general contract; CUSUM table assumed ~90-pt baseline undocumented. |
| 2026-08-03 05:00 | Added **Phase 2.5 — Hardening** after a Codex adversarial review found must-fix defects in Plans 1–2 | Codex mutation-tested and multi-seed-validated; found (a) clinical holiday-dip inert (0/2920 rows change), (b) amendment integration test fires on the OPPOSITE signal, (c) detectors return SAFE on NaN/empty data, (d) CUSUM 2h/h uncalibrated & window-length-dependent (in-control CRITICAL up to 3.6% at n=365 — my "20-seed validated" claim was too thin), (e) calm-QC test seed-selected, (f) OOS mislabeled as clustered. My internal reviews measured margins, not mutations. | Inserts 5 hardening tasks before Phase 3; Isolation Forest wiring + 0.10 cutoff deferred to Phase 4 (claim corrected: P2 is NOT a complete 3-detector stack). | Plans 1–2 were treated as a complete, trustworthy foundation. |
