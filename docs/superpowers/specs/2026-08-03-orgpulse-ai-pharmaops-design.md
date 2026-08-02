# OrgPulse AI — PharmaOps: Design Spec

**Date:** 2026-08-03
**Status:** Approved (pending written-spec review)
**Author:** Solutions architect (pairing session)

---

## 1. One-line description

An autonomous organizational-intelligence platform for a pharmaceutical
organization: it ingests synthetic-but-realistic data from three operational
streams (clinical trial operations, QC/lab, supply chain), and on a schedule
**detects anomalies, forecasts trends 30/60/90 days ahead, and generates
plain-language, GxP-aware, human-gated recommendations** — replacing passive BI
dashboards with a proactive standing agent.

**Interview/post one-liner:** "It's Palantir AIP's architecture — multi-source
ingestion → continuous anomaly detection → forecasting → LLM course-of-action
recommendations — but open-source, self-hosted, and built to the EU AI Act
high-risk + GxP bar, with every recommendation human-gated and fully
reconstructable from a tamper-evident audit log."

## 2. Goal & audience

- **Purpose:** Portfolio project demonstrating capability to AI Engineer / AI
  Architect / AI Product Owner audiences at Swiss pharma companies (Roche,
  Novartis, Ferring, Lonza). It will be **posted about publicly** (LinkedIn /
  social) — the intended consumer is *someone scrolling a post who clicks a
  link*, not a live interviewer at a terminal.
- **Primary deliverables (the "shop window"):** a live Streamlit Cloud demo, a
  narrative README, an architecture diagram, and a standalone **case-study
  writeup** (problem → architecture → EU AI Act angle → results) suitable for
  posting or adapting directly.
- **Secondary deliverable:** a clonable repo with a one-command local run that
  proves the system is real (not vaporware) for the minority who dig in.

## 3. Differentiation (why this is not a duplicate)

The user already has portfolio projects that overlap on capability:
- `operations-data-agent` — reactive natural-language Q&A over enterprise data
  with visualizations, forecasting, and recommendations.
- A Pharma+Banking GenAI Platform (Azure, RAG) and an LLM Fine-Tuning Platform
  (Dagster + MLflow + KServe).

**OrgPulse's wedge (chosen: "both, tightly fused"):** an *autonomous agent you
can actually run in a regulated pharma environment*. It supersets
`operations-data-agent`'s analytics (forecast/recommend/visualize) but flips the
interaction model from **reactive** (you ask it) to **autonomous** (it wakes on a
cron, remembers baselines across runs, and pushes findings at you), and layers on
**regulated-industry rigor** (GxP/ICH E6(R3) framing + EU AI Act high-risk
compliance) as an inseparable part of the story. The autonomy and the compliance
together are the point.

This project does **not** re-do RAG or fine-tuning — it focuses on operational
intelligence and forecasting.

## 4. Prior-art grounding (research findings)

- **Palantir AIP architecture** is four layers: Ontology → model
  access/governance → **AIP Logic** (continuous background evaluation that flags
  anomalies + generates course-of-action recommendations) → **Actions**. "AIP
  Agent Studio deploys autonomous monitoring agents"; "the LLM is architecturally
  prohibited from kinetic action, and every recommendation requires human
  validation." Our mapping: **Hermes ≈ AIP Agent Studio** (autonomous monitor),
  **LangGraph ≈ AIP Logic** (analysis DAG), recommendations are decision-support
  only, human-gated.
- **EU AI Act** (full high-risk obligations under Annex III mandatory as of
  2026-08-02): **Article 12** — tamper-evident logging, 6-month retention, must
  enable post-hoc reconstruction of each individual AI-assisted decision;
  **Article 14** — human oversight by competent persons; **Articles 11/13** —
  technical documentation + transparency; **Article 26** — deployer obligations.
- **Pharma SPC practice:** CUSUM for detecting shifts in process means, combined
  with a forecast + SPC rules zoning each predicted value **Safe / Warning /
  Critical**. Statistical control charts for batch environments; ML time-series
  for continuous. Our detector stack (CUSUM + Z-score + Isolation Forest) with
  Safe/Warning/Critical zoning is domain-standard, not invented.

Sources: Palantir AIP architecture docs; Towards AI "Inside Palantir AIP"; EU AI
Act Articles 12/13 & 26; arXiv 2509.16431 (proactive SPC via forecasting).

## 5. Architecture

### 5.1 The two-orchestrator split (core)

| Layer | Tech | Role | Palantir analogue |
|---|---|---|---|
| **Autonomic loop** | Hermes (0.19.1, `~/.hermes/`) | Cron wake, baseline memory across runs, invokes the pipeline, decides alerts, writes artifacts + audit log, hosts 3 domain `SKILL.md` files | AIP Agent Studio |
| **Analysis DAG** | LangGraph | Deterministic 4-node pipeline invoked once per run | AIP Logic |

Hermes is the *nervous system* (when to run, what to remember, what to alert).
LangGraph is the *reasoning* (given today's data + baseline: what's wrong, what's
coming, what to do). The boundary is explicit so "why two frameworks?" has a crisp
answer.

### 5.2 Data plane — synthetic generators

Three daily time-series generators. Each Hermes run advances a clock and ingests
"the next day." Generators are seeded/deterministic for reproducibility.

- **Clinical trials (HERO — full forecasting rigor):** multi-site enrollment as
  staggered logistic S-curves, screen-fail rate, dropout/discontinuation,
  per-site performance. Deliberately seeded with real-world structure the
  forecaster is *not* told about: a protocol-amendment step change, a holiday
  enrollment dip, and one chronically underperforming site.
- **QC/lab (solid):** batch pass/fail rate, OOS (out-of-spec) event clusters,
  equipment calibration **drift** (slow mean shift — the CUSUM case).
- **Supply chain (solid):** inventory sawtooth, supplier lead-time spikes,
  supplier reliability seasonality.

Tooling: Faker + custom domain generators (numpy).

### 5.3 Analysis DAG — LangGraph, 4 nodes

1. **Anomaly Detector** — CUSUM + Z-score (statistical) **and** Isolation Forest
   (ML) per KPI, compared against the Hermes-remembered baseline. Emits
   Safe/Warning/Critical zoning per KPI.
2. **Forecaster** — Prophet, 30/60/90-day horizons with confidence bands.
   - **Hero stream (clinical trials):** full rigor — rolling-origin backtest,
     honest error metrics (MAPE/MASE), and a **calibration check** (does the 90%
     interval actually contain ~90% of held-out points?).
   - **Other two streams:** point forecast + interval, lighter treatment.
3. **Domain Analyst (LLM node)** — interprets anomalies + forecasts in pharma-ops
   terms: why this matters, GxP relevance, which deviation thresholds are in play.
4. **Recommendation Writer (LLM node)** — CAPA-framed, ICH E6(R3)-aware
   plain-language recommendations, each stamped decision-support-only with a
   human-oversight disclaimer.

**Output per run:** (a) structured JSON findings object, (b) plain-language
report (Markdown), (c) threshold alert if any KPI is Critical.

### 5.4 Governance layer (cross-cutting — the compliance wedge)

- **EU AI Act self-classification** document: Annex III high-risk, plus Article 11
  technical documentation.
- **Tamper-evident audit log** (Article 12): append-only, **hash-chained** (each
  entry hashes the previous entry). Every recommendation carries `trace_id`,
  timestamp, input-data hash, model + version — any past decision is fully
  **reconstructable**, and tampering is detectable via chain verification.
- **Human oversight** (Article 14): recommendations carry a `review_status` field
  defaulting to `PENDING_HUMAN_REVIEW`; nothing is presented as an automated
  decision.

### 5.5 Presentation — Streamlit (split-plane)

Reads the **latest published run artifacts** (not live compute). Panels: KPI
cards with Safe/Warning/Critical zoning; trend charts with forecast bands +
anomaly highlights; latest-recommendations panel; **audit-log / trace viewer**;
**EU AI Act compliance panel**. Deploys to Streamlit Cloud reading committed
artifacts.

### 5.6 Deployment — the split plane

The tension: a *truly autonomous* Hermes loop (persistent, cron) and a *live
public Streamlit Cloud demo* cannot be the same running process (Streamlit Cloud
is stateless and sleeps). Resolution (**"split plane"**):

- **Compute plane:** Hermes runs the real autonomous loop locally (cron + baseline
  memory + orchestration), writing timestamped run artifacts to a committed store.
- **Presentation plane:** Streamlit Cloud reads the **latest published run** and
  renders it. The public URL is the shop window; the local run is the engine.
- A **pre-seeded run history** (90+ days of daily runs, committed) makes the live
  demo look alive on first landing — populated charts, real anomalies, forecast
  bands.

## 6. Data flow

```
cron → Hermes wakes
     → load baselines from memory
     → 3 domain SKILLs ingest next-day synthetic data
     → invoke LangGraph DAG (detect → forecast → analyze → recommend)
     → findings JSON + recommendations
     → update baselines in Hermes memory
     → write /runs/<timestamp>/ + update /runs/latest
     → append hash-chained audit-log entry
     → emit alert if any KPI Critical
Streamlit Cloud → serves /runs/latest
```

## 7. Offline mode (essential infrastructure, not a fallback)

Because the public demo must always look alive without depending on a live paid
API key on Streamlit Cloud, and because a browsing audience must always see the
full loop:

- `--offline` mode: real generators, real anomaly detection, real forecasts (the
  statistical/ML parts are deterministic and key-free); recommendations come from
  a **deterministic template stub** when no API key is present.
- With an API key: nodes 3 & 4 use the real LLM (Claude Sonnet default,
  configurable) for genuine recommendations.
- The pre-seeded committed run history is generated once (with real LLM output if
  a key is available at seed time) so the live demo is always populated.

## 8. Tech stack & explicit scope cuts

**Stack:** Python 3.11 (Hermes venv), uv (deps), Hermes 0.19.1 (autonomy),
LangGraph (analysis DAG), Prophet (forecasting), scikit-learn (Isolation Forest),
pandas/numpy, Faker (+ custom generators), Streamlit (dashboard), pytest (TDD),
Docker (containerization). LLM: Claude (Sonnet default, configurable).

**YAGNI cuts (deliberately not building):**
- **No FastAPI** — Hermes invokes the pipeline via a **CLI/tool entrypoint**
  (the pipeline is a Python package exposing a `run` command; Hermes calls it as
  a tool / subprocess and reads back the JSON artifact). Streamlit reads files.
  No external HTTP API surface is needed for v1.
- **No database** — file-based run artifacts + Hermes memory suffice.
- **No auth / no multi-user** — single-tenant portfolio demo.
- **No real external data** — synthetic only, by constraint.
- Two of three streams get "solid" (not exhaustive) treatment; forecasting rigor
  concentrated on the hero stream.

## 9. Testing strategy (TDD)

- **Generators:** statistical-property tests — cumulative enrollment monotonic,
  OOS rate within realistic band, seasonality present, injected structural breaks
  land where intended.
- **Detectors:** injected-anomaly tests — inject a step change, assert CUSUM
  fires; inject an outlier, assert Z-score / Isolation Forest fires; assert no
  false-fire on clean series.
- **Forecaster:** backtest harness produces MAPE/MASE on held-out windows; assert
  90% interval coverage ≈ 90% on synthetic holdout (calibration).
- **Pipeline:** golden-path integration test — run DAG on a fixture, assert JSON
  schema validity + recommendation present + human-oversight disclaimer present.
- **Audit log:** hash-chain integrity test (tamper is detectable); trace
  reconstruction test (a `trace_id` reconstructs its full decision context).
- **Governance:** schema-enforced — every output carries `trace_id`,
  `review_status`, and disclaimer.

## 10. Success criteria (revised for public-portfolio target)

1. Live Streamlit Cloud demo is **always populated** (pre-seeded history) and
   shows KPI cards, forecast bands, anomaly highlights, recommendations, audit
   trail, and EU AI Act panel.
2. Hermes runs the autonomous loop on a configurable cron (default daily 07:00),
   remembers baselines, and produces the three outputs (JSON findings,
   plain-language report, threshold alert).
3. Hero-stream forecasting reports honest held-out backtest metrics + interval
   calibration.
4. Governance is real: hash-chained audit log with reconstructable decisions +
   EU AI Act self-classification doc.
5. Repo has a **narrative README**, an **architecture diagram**, a **case-study
   writeup**, and a **one-command local run** (`--offline` works with no API key).
6. The system architecture is explainable in the one-liner (Section 1).

## 11. Rough phasing (feeds the roadmap)

- **P0 — Scaffolding:** repo, uv env, config, project structure, Hermes wiring
  skeleton.
- **P1 — Synthetic generators** (3 streams) + tests.
- **P2 — Anomaly detection** (CUSUM / Z-score / Isolation Forest) + tests.
- **P3 — Forecasting** (Prophet + backtest + calibration, hero stream) + tests.
- **P4 — LangGraph 4-node DAG** + LLM nodes + offline stub + tests.
- **P5 — Governance** (hash-chained audit log, EU AI Act classification,
  disclaimers, schema enforcement) + tests.
- **P6 — Hermes integration** (3× `SKILL.md`, cron, baseline memory, artifact
  writing).
- **P7 — Streamlit dashboard** + pre-seeded run history + Streamlit Cloud deploy.
- **P8 — Story layer:** narrative README, architecture diagram, case-study
  writeup; final verification + code review.

## 12. Open risks

- **R1 — Prophet install weight** (cmdstanpy compiled dep) could complicate the
  clone-and-run and Docker build. *Mitigation:* pin/pre-resolve in uv + Docker;
  `--offline` never blocks on LLM; consider `statsforecast` as a lighter fallback
  if Prophet proves fragile in CI/Streamlit Cloud.
- **R2 — Forecasting on self-generated data proving little.** *Mitigation:* hero
  generator injects breaks/dips/laggards the forecaster isn't told about; report
  held-out backtest + calibration, not in-sample fit.
- **R3 — Streamlit Cloud dependency footprint** (Prophet + LangGraph) may exceed
  comfortable build limits. *Mitigation:* the presentation plane only needs to
  *read* artifacts + plot — keep the Streamlit app's dependency set minimal
  (pandas/plotly/streamlit), not the full pipeline stack.
