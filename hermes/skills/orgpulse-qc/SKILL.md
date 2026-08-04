---
name: orgpulse-qc
description: "Monitor QC/lab pass-rate, OOS incidence, and calibration drift via OrgPulse."
version: 1.0.0
author: OrgPulse AI
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [pharma, QC, lab, OOS, calibration, CUSUM, anomaly-detection, GxP]
    related_skills: [orgpulse-clinical, orgpulse-supply]
prerequisites:
  commands: [uv]
---

# OrgPulse — QC / Lab Monitor

Autonomous monitoring of **QC/lab** KPIs: batch pass-rate, drift-dependent
out-of-spec (OOS) incidence, and equipment **calibration drift** (the case CUSUM
is built to catch), against the remembered baseline.

## When to Use

- Scheduled daily QC monitoring.
- The user asks about batch pass-rate, OOS events, or instrument calibration.
- To catch a slow calibration drift before it crosses a specification limit.

## Prerequisites

```bash
cd ~/Programming/orgpulse-ai
```

## 1. Run a tick and read the QC lines

```bash
uv run orgpulse tick --tick-number "$TICK" --offline
grep -E 'pass_rate|calibration_offset' runs/latest/report.md
```

## 2. Interpret

- `calibration_offset` with a high-side CUSUM signal ⇒ a sustained upward drift —
  flag for instrument re-calibration review.
- A declining `pass_rate` ⇒ a real quality signal (it is NOT a calm KPI).
- A `DATA_QUALITY` finding ⇒ investigate the data source/instrumentation first;
  it is a data problem, not a process deviation (do not open a process CAPA).

All recommendations are human-gated; the run is logged to the tamper-evident
`runs/latest/audit.jsonl`.
