# 🧬 OrgPulse AI — PharmaOps

**An autonomous organizational-intelligence agent you can actually run in a regulated pharma environment.**

OrgPulse watches three streams of pharma operations data — clinical enrollment, QC/lab, and
supply chain — and every day, on its own: detects anomalies, forecasts 30/60/90 days ahead,
and writes plain-language, **human-gated** recommendations framed for GxP / ICH-E6(R3). Every
decision it makes is logged to a **tamper-evident audit trail**, and it self-classifies as a
**high-risk AI system under the EU AI Act (Annex III)**.

The wedge most "AI ops" demos miss: **autonomy and compliance, fused.** An agent that runs
itself *and* is safe to run where regulators are watching.

> ⚠️ Synthetic data, decision-support prototype. Not a medical device. Every recommendation
> requires human review before any action (EU AI Act Art. 14).

---

## See it

- **Live dashboard:** _not yet deployed — deploy-ready; add your Streamlit Cloud URL here_
- **Case study (the story + honest results):** [docs/CASE_STUDY.md](docs/CASE_STUDY.md)
- **Architecture + diagram:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **EU AI Act self-classification:** [docs/EU_AI_ACT_SELF_CLASSIFICATION.md](docs/EU_AI_ACT_SELF_CLASSIFICATION.md)

## Quickstart

```bash
uv sync

# Run one autonomous tick — generates data, detects, forecasts, reasons,
# writes tamper-evident artifacts to runs/latest, remembers the baseline.
uv run orgpulse tick --tick-number 0

# Regenerate the committed multi-run demo history.
uv run python scripts/seed_runs.py

# Launch the dashboard (reads runs/latest — no model, no API key).
uv run --extra dashboard streamlit run streamlit_app/app.py
```

Keyless by default: the reasoning nodes use a deterministic offline stub. Set
`ANTHROPIC_API_KEY` to route the analyst/recommendation nodes through **Claude Sonnet**
(`claude-sonnet-5`) instead — the audit trail records which model produced each recommendation.

## What it does

```
generate 3 streams ─▶ detect ─▶ forecast ─▶ analyst (LLM) ─▶ recommend (human-gated)
                        │                                          │
                remembered baseline                        hash-chained audit log
```

- **Detect** — Z-score + window-length-calibrated CUSUM against a baseline *remembered from
  the previous run*. Corrupt or insufficient data becomes an explicit `DATA_QUALITY` state
  that **dominates** severity — it never passes as "all clear". (An Isolation Forest detector
  is built and unit-tested in `detect/isolation.py` but is **not yet wired** into the live
  univariate path — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#detector-status).)
- **Forecast** — deterministic Prophet on the hero KPI (clinical enrollment): 30/60/90-day
  forecast with an **in-sample-calibrated** 90% interval (the calibration factor is applied to
  the published bounds), backtested honestly (see results below).
- **Reason** — an LLM turns the numbers into a GxP-framed interpretation and a CAPA
  recommendation, or, for data-quality findings, a "fix the data, don't open a CAPA" note.
- **Govern** — every recommendation is `PENDING_HUMAN_REVIEW` with an Art. 14 disclaimer and
  is logged to a hash-chained, reconstructable audit trail (Art. 12).
- **Run itself** — the whole loop is a Hermes SKILL scheduled daily; baselines persist across
  runs, so the agent has *memory*, not just a cron job.

## Honest results

This project optimizes for **trustworthiness over leaderboard numbers**, and says so:

- The Prophet forecaster **does not beat a naïve baseline** (rolling-origin MASE ≈ 1.4,
  MAPE ≈ 17%). The value delivered is the **calibrated uncertainty interval** and the audit
  trail around the forecast — not point accuracy. Raw Prophet intervals were over-confident
  (~70% measured coverage); a calibration factor is **applied** to widen the published interval
  toward the nominal 90%. That calibration is fit **in-sample**, so out-of-sample coverage is
  not claimed (a production system would use split-conformal) — and the README says so.
- CUSUM thresholds are **calibrated (Monte-Carlo table) to a ~1% in-control false-alarm rate per
  recent-window length at the reference 60-sample baseline** (the enforced minimum; longer
  baselines are *more* conservative, i.e. well below 1%), for the fixed slack `k=0.5` (other `k`
  values are rejected as uncalibrated) — validated with
  mutation-style and multi-seed tests, not just favorable margins.

## Why it exists

A portfolio project demonstrating end-to-end capability for **Swiss-pharma AI Engineer /
Architect / Product-Owner** roles: agentic autonomy, statistically honest ML, LLM reasoning
with a keyless-deterministic fallback, and EU AI Act / GxP governance treated as a first-class
cross-cutting concern rather than a disclaimer bolted on at the end.

## Tech stack

Python 3.11 · uv · LangGraph (analysis DAG) · Prophet (forecasting) · scikit-learn ·
Anthropic Claude (reasoning, optional) · Hermes (autonomy/scheduling) · Streamlit + Plotly
(dashboard) · stdlib hashlib (audit chain).

## Repository map

| Path | What's there |
|------|--------------|
| `src/orgpulse/generators/` | Synthetic clinical / QC / supply streams with injected structural breaks |
| `src/orgpulse/detect/` | Baseline, Z-score, calibrated CUSUM (live); Isolation Forest + orchestrator (built, not yet wired); validation |
| `src/orgpulse/forecast/` | Deterministic Prophet forecaster, rolling-origin backtest, interval calibration |
| `src/orgpulse/pipeline/` | LangGraph `detect→forecast→analyst→recommend` DAG + LLM client (offline/Claude) |
| `src/orgpulse/governance/` | Hash-chained audit log + EU AI Act self-classification |
| `src/orgpulse/agent/` | Cross-run baseline memory + the autonomous `run_tick` loop |
| `hermes/` | 3× domain `SKILL.md` + cron runbook (split-plane) |
| `streamlit_app/` | Self-contained dashboard (reads `runs/`, no heavy deps) |
| `runs/` | Committed demo run history (so the dashboard is populated on first load once deployed) |
| `docs/` | Architecture, case study, EU AI Act classification, specs & plans |

## Tests

```bash
uv run pytest -q      # 145 passing
```

Keyless and offline: no test needs `ANTHROPIC_API_KEY` or network access. The suite was
hardened through six rounds of external adversarial review (the fifth forced a root-cause
redesign of the fail-safe and audit-trust boundaries rather than another patch).
