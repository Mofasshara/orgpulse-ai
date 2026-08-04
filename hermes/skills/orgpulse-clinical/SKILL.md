---
name: orgpulse-clinical
description: "Monitor clinical-trial enrollment via OrgPulse; run an autonomous tick and read the report."
version: 1.0.0
author: OrgPulse AI
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [pharma, clinical-trials, enrollment, anomaly-detection, forecasting, GxP]
    related_skills: [orgpulse-qc, orgpulse-supply]
prerequisites:
  commands: [uv]
---

# OrgPulse — Clinical Trial Operations Monitor

Run OrgPulse's autonomous analysis for **clinical-trial enrollment** (daily
enrollment, screen-fail, dropout, per-site performance): detect anomalies against
the *remembered* baseline, forecast 30/60/90 days, and produce GxP-aware,
human-gated recommendations.

## When to Use

- Scheduled daily monitoring of trial enrollment (the default cron target).
- The user asks "how is enrollment trending" / "are we behind on recruitment".
- After a protocol amendment, to watch for the expected enrollment change.
- Any request to check clinical-operations KPIs against baseline + forecast.

## Prerequisites

```bash
cd ~/Programming/orgpulse-ai   # the OrgPulse repo (uv-managed)
```

## 1. Run an autonomous tick

Each tick compares the current period against the baseline **remembered from the
previous run** (bootstrapped on the very first run), writes a timestamped run to
`runs/`, and updates the baseline memory. Keyless by default (offline LLM stub);
add `--online` only when `ANTHROPIC_API_KEY` is set.

```bash
uv run orgpulse tick --tick-number "$TICK" --offline
```

## 2. Read the findings for enrollment

```bash
sed -n '/daily_enrollment/p' runs/latest/report.md    # the enrollment KPI line
cat runs/latest/report.md                              # full plain-language report
```

`runs/latest/findings.json` holds the structured findings; every recommendation is
`PENDING_HUMAN_REVIEW` (EU AI Act Art. 14 — decision support, not an automated
decision). `runs/latest/audit.jsonl` is the tamper-evident audit trail (Art. 12).

## 3. Escalate on a critical enrollment signal

If the enrollment finding is `CRITICAL` (e.g. a sustained drop vs. the remembered
baseline), surface the recommendation to the responsible clinical-operations lead
for CAPA/deviation review — do not act automatically.
