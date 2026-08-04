---
name: orgpulse-supply
description: "Monitor supply-chain inventory, supplier lead-time, and reliability via OrgPulse."
version: 1.0.0
author: OrgPulse AI
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [pharma, supply-chain, inventory, lead-time, supplier-reliability, forecasting]
    related_skills: [orgpulse-clinical, orgpulse-qc]
prerequisites:
  commands: [uv]
---

# OrgPulse — Supply Chain Monitor

Autonomous monitoring of **supply-chain** KPIs: inventory (reorder sawtooth),
supplier **lead-time** spikes, and supplier reliability seasonality, against the
remembered baseline.

## When to Use

- Scheduled daily supply-chain monitoring.
- The user asks about inventory levels, supplier lead times, or reliability.
- To catch a supplier lead-time disruption early.

## Prerequisites

```bash
cd ~/Programming/orgpulse-ai
```

## 1. Run a tick and read the supply lines

```bash
uv run orgpulse tick --tick-number "$TICK" --offline
grep -E 'lead_time_days' runs/latest/report.md
```

## 2. Interpret

- A high-side signal on `lead_time_days` vs. the remembered baseline ⇒ a supplier
  lead-time disruption — surface for procurement review and safety-stock decisions.
- Recommendations are decision-support only and require human sign-off.
