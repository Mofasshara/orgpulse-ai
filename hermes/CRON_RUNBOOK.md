# OrgPulse — Autonomous Scheduling Runbook (Hermes)

OrgPulse runs as a **standing agent**: on a schedule it ingests the next period,
detects anomalies against the *remembered* baseline, forecasts, produces
human-gated recommendations, and writes a timestamped run + a tamper-evident audit
log. This runbook wires that loop to Hermes.

## Split-plane model

- **Compute plane (here):** Hermes runs `orgpulse tick` locally. It owns the cron,
  the baseline memory (`runs/baseline_memory.json`, cross-run state), and writes
  every run to `runs/<timestamp>/` and `runs/latest/`.
- **Presentation plane:** the Streamlit dashboard (Plan 7) only *reads* the
  committed `runs/latest/` artifacts — it never runs the pipeline. This is why the
  heavy Prophet/LangGraph stack stays out of the deployed app.

## The tick command

```bash
cd ~/Programming/orgpulse-ai
uv run orgpulse tick --tick-number "$TICK" --offline    # keyless; --online uses Claude if ANTHROPIC_API_KEY is set
```

- Tick 0 **bootstraps** the baseline memory (no alarm on a never-seen KPI).
- Tick N (>0) detects the current period against tick N-1's remembered baseline,
  then saves the current period as the next baseline.
- Every recommendation is `PENDING_HUMAN_REVIEW` (EU AI Act Art. 14); every run is
  logged to a hash-chained, reconstructable audit trail (Art. 12).

## Default schedule: daily 07:00

### Option A — Hermes cron (preferred)

Register a Hermes cron job that runs the tick daily at 07:00, incrementing the tick
counter. Hermes stores the baseline across runs in its memory (`~/.hermes/memories/`);
point `--memory` there for a single shared baseline, or keep it in the repo `runs/`.

```
schedule: "0 7 * * *"          # daily 07:00 local
command:  uv run orgpulse tick --tick-number "$TICK" --offline
```

### Option B — system cron (fallback)

```cron
0 7 * * *  cd $HOME/Programming/orgpulse-ai && TICK=$(cat runs/.tick 2>/dev/null || echo 0) && uv run orgpulse tick --tick-number "$TICK" --offline && echo $((TICK+1)) > runs/.tick
```

## After each run

- `runs/latest/report.md` — plain-language report for the ops team.
- `runs/latest/findings.json` — structured findings (severity, forecast, recommendations).
- `runs/latest/audit.jsonl` — verify integrity any time: it is tamper-evident.
- Domain skills: `hermes/skills/orgpulse-{clinical,qc,supply}/SKILL.md`.
