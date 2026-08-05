# OrgPulse AI — Architecture

OrgPulse is a **split-plane** system. A **compute plane** (the Hermes-scheduled agent)
does all the heavy work locally and commits small, tamper-evident **run artifacts**; a
**presentation plane** (the Streamlit dashboard) only *reads* those artifacts. The two
planes share nothing but the `runs/` directory — which is why the dashboard deploys to
Streamlit Cloud with three lightweight dependencies and never imports Prophet or LangGraph.

```mermaid
flowchart TB
    subgraph CP["🖥️ Compute plane — Hermes agent (local, scheduled daily 07:00)"]
        direction TB
        MEM[("BaselineMemory<br/>cross-run state")]
        GEN["Synthetic streams<br/>clinical · QC · supply"]
        subgraph DAG["LangGraph analysis DAG"]
            direction LR
            D["detect<br/>Z-score · CUSUM"] --> F["forecast<br/>Prophet 30/60/90d + calibrated interval"]
            F --> A["analyst<br/>LLM interpretation"]
            A --> R["recommend<br/>human-gated CAPA"]
        end
        GEN --> D
        MEM -. "remembered baseline" .-> D
        R --> W["write artifacts"]
        W -. "this period's baseline" .-> MEM
    end

    subgraph ART["📦 Committed run artifacts (runs/)"]
        direction LR
        FJ["findings.json"]
        RM["report.md"]
        AJ["audit.jsonl<br/>hash-chained"]
    end

    subgraph PP["🌐 Presentation plane — Streamlit Cloud (read-only)"]
        DASH["Dashboard<br/>KPIs · forecast fan · recs · audit viewer"]
    end

    subgraph GOV["⚖️ Governance (cross-cutting)"]
        AUD["Tamper-evident audit log<br/>EU AI Act Art. 12"]
        CLS["Self-classification<br/>Annex III high-risk"]
        HG["Human oversight gate<br/>Art. 14 — PENDING_HUMAN_REVIEW"]
    end

    W --> FJ & RM & AJ
    FJ & RM & AJ --> DASH
    R -. logged .-> AUD
    AUD --- AJ
    HG -. gates .-> R

    HERMES["⏰ Hermes cron / SKILL.md ×3"] -. triggers .-> GEN
```

## The planes

| Plane | Runs where | Deps | Responsibility |
|-------|-----------|------|----------------|
| **Compute** | Local / self-hosted, Hermes-scheduled | full stack (numpy, scikit-learn, Prophet, LangGraph, anthropic) | Generate → detect → forecast → reason → recommend → write artifacts; remember baselines |
| **Presentation** | Streamlit Cloud | `streamlit`, `pandas`, `plotly` | Read `runs/latest` + history; render; re-verify the audit chain |

Keeping the planes apart is what makes the public demo both **cheap to host** and **always
alive**: the dashboard has no model to run and no key to hold — it renders committed results.

## The analysis DAG (LangGraph)

`detect → forecast → analyst → recommend`, a typed `StateGraph`:

1. **detect** — Z-score + window-calibrated CUSUM per KPI against the *remembered* baseline.
   Corrupt/insufficient data surfaces as `DATA_QUALITY`, which **dominates** overall
   severity — it never silently passes as "all clear". (Isolation Forest is built but not
   wired — see [Detector status](#detector-status).)
2. **forecast** — deterministic Prophet on the hero KPI (`daily_enrollment`): 30/60/90-day
   point forecast with an **in-sample-calibrated** 90% interval (the calibration factor is
   applied to the published bounds, not just diagnosed). Backtested honestly (it does *not*
   beat naïve; out-of-sample interval coverage is not claimed).
3. **analyst** — an LLM turns the numbers into a GxP-framed interpretation. Offline
   deterministic stub by default (keyless); Claude Sonnet when `ANTHROPIC_API_KEY` is set
   **and** `--online` is passed (the CLI defaults to `--offline`).
4. **recommend** — a human-gated CAPA recommendation per anomaly, each
   `PENDING_HUMAN_REVIEW` with an EU AI Act Art. 14 disclaimer. Data-quality findings get a
   distinct "do not open a CAPA — fix the data" recommendation instead.

## Cross-run memory (the "autonomous" part)

Each tick detects the **recent period** (30 days) against the **established baseline**
(a 120-day trailing window) remembered from the previous tick, then writes the new baseline
back. The baseline window is deliberately longer than the detection window so CUSUM's
false-alarm calibration (which needs ≥60 samples) always holds — otherwise every
post-bootstrap tick would degrade to `DATA_QUALITY`. `run_tick` validates these
preconditions at entry (`baseline_days ≥ 60`, `base_days ≥ max(60, days_per_period)`) and
never persists a baseline below the 60-sample floor.

**What "remembered" means (and doesn't).** The agent remembers baseline **statistics**
(mean/std/n per KPI), not raw data. In this synthetic prototype the generators produce a
**fresh realization at the requested horizon on each tick** — a longer horizon is a new draw
from the same data-generating process, not an append to the previous tick's exact series. So
a tick detects a new realization's recent period against summary statistics carried over from
the previous realization. Detection stays valid because both realizations share the same
underlying process (a structural break lives in the process and appears in every realization
at the same point), but this is deliberately **not** an append-only "one evolving world". A
production deployment would persist and append real observed data (or use counter-based,
date-keyed RNG) so history is byte-stable across horizons; that is out of scope here.

### Detector status

| Detector | Status |
|----------|--------|
| Baseline (mean/std/n) | Live |
| Z-score | Live — wired into the detection node |
| Window-calibrated CUSUM | Live — wired into the detection node |
| Isolation Forest (`detect/isolation.py`) | **Built & unit-tested, NOT wired** into the live path; no multivariate orchestration yet |

The README and diagram reflect the *live* stack (Z-score + CUSUM). Isolation Forest is
retained as a tested component pending a multivariate integration; it does not run in the
current pipeline.

## Governance is not a feature — it's a cross-cut

- **Art. 12 logging** → every run and recommendation is appended to a **hash-chained**
  `audit.jsonl`. The run entry records the input **lineage** — the generation `seed` and
  parameters, an `input_data_hash` of the actual input frames, and (for a remembered-baseline
  tick) a `baseline_input_hash` of the baseline that drove detection — plus output-integrity
  hashes `findings_hash` and `report_hash`, and the model+version. The dashboard's `verify_run`
  checks the chain; that the displayed `findings.json` AND `report.md` still hash to what the
  run entry attests to (so editing either is caught); and that the number of logged
  recommendation entries matches the count in `findings.json` (so deleting trailing records —
  which a forward hash chain can't detect on its own — is caught too). Both verifiers (dashboard
  and compute-plane) fail closed on missing/empty/malformed logs; a proven-equivalence test
  guards against drift between the two implementations. An **offline** decision (the deterministic
  default) can be exactly **regenerated** from the recorded lineage and its outputs checked against
  the logged hashes; an **online** (Claude) run is fully traceable but its LLM output is stochastic
  and not bit-reproducible (no model snapshot / decoding params are captured).

  **Threat model (stated honestly):** an *unkeyed* hash chain stored in the same writable
  file detects **accidental corruption and naïve tampering** (editing a past entry, or the
  findings, breaks verification). It does **not** stop an adversary with write access who
  rewrites the whole file and recomputes the chain forward using the public algorithm — that
  requires a signature under a key the artifact writer doesn't hold, or anchoring the terminal
  hash in a separate trusted store. Keyed signing is deliberately out of scope for this
  keyless-by-default prototype and is noted as future work.
- **Art. 14 human oversight** → the system is *architecturally* prohibited from acting; the
  strongest output it can emit is a recommendation awaiting a qualified human.
- **Annex III self-classification** → machine-readable (`governance/classification.py`) and
  rendered to `docs/EU_AI_ACT_SELF_CLASSIFICATION.md`.

See [CASE_STUDY.md](CASE_STUDY.md) for the reasoning behind these choices and the honest results.
