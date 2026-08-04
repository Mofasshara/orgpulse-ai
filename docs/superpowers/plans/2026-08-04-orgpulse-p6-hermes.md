# OrgPulse P6 — Hermes Autonomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox (`- [ ]`) steps.

**Goal:** Turn the pipeline into a *standing agent* that Hermes runs on a schedule and that **remembers baselines across runs** (the autonomy flex): each tick compares the current period against the *previous* period's remembered per-KPI baseline stored in Hermes memory, writes a timestamped run to a committed run-history, and updates the memory. Plus the authentic Hermes wiring: 3× `SKILL.md` (clinical/QC/supply) and a cron runbook.

**Architecture:** A new `orgpulse.agent` subpackage. `baseline_memory.py` — a JSON-backed `BaselineMemory` persisting `dict[kpi -> Baseline]` (Plan-2 `Baseline`). `tick.py` — `run_tick(...)`: loads remembered baselines, ingests the current period, runs the detectors **against the remembered baseline** (bootstrap on the first tick), forecasts the hero, runs the LLM analyst/recommendation nodes (offline stub by default), writes `runs/<ts>/{findings.json,report.md,audit.jsonl}` + updates `runs/latest`, and saves the current period's baseline back to memory. An `orgpulse tick` CLI. The 3× `SKILL.md` + cron runbook tell Hermes how to monitor each stream and invoke `orgpulse tick` on a cron (default daily 07:00). Split-plane: Hermes runs this locally; the presentation plane (Plan 7 Streamlit) just reads the committed artifacts.

**Tech Stack:** Python 3.11, uv, stdlib json, reuse Plans 2–5. pytest. No new deps.

## Global Constraints
- Python 3.11; deps via uv; **no new deps**. Use `uv run`. Reuse `orgpulse.detect`/`forecast`/`pipeline`/`governance`.
- The remembered baseline is the CROSS-RUN state: tick N (>0) detects the current window against the baseline saved by tick N-1; tick 0 bootstraps (fit from the current window) and does not alarm. Memory persists across process restarts (JSON file).
- Keyless & deterministic (offline LLM stub default). Audit log written per run (Plan 5). JSON-safe.
- SKILL.md files follow the real Hermes format (YAML frontmatter + "When to Use" + bash sections).
- No Co-Authored-By. TDD.

---

### Task 1: Baseline memory + memory-backed tick

**Files:** Create `src/orgpulse/agent/__init__.py`, `baseline_memory.py`, `tick.py`; Test `tests/agent/__init__.py` (empty), `tests/agent/test_tick.py`.

**Interfaces:**
- `baseline_memory.BaselineMemory(path)` — `load() -> dict[str, Baseline]`, `save(dict[str, Baseline])`; JSON serialization of `Baseline(mean,std,n)`. Returns `{}` if the file doesn't exist.
- `tick.run_tick(memory: BaselineMemory, tick_number: int, *, days_per_period=30, base_days=200, seed=42, offline=True, out_root=Path("runs"), now=None) -> Findings` — generates `base_days + tick_number*days_per_period` days of the three streams; the "current period" is the last `days_per_period` days; for each KPI, if `memory` has a remembered baseline, detect the current window against it (cusum_detect/zscore_detect from Plan 2, which take a `Baseline`); else bootstrap (fit_baseline on the current window → SAFE, no alarm). Then forecast the hero + run the analyst/recommendation nodes, write `out_root/<generated_at>/` + `out_root/latest/` artifacts (findings.json, report.md, audit.jsonl), and SAVE the current period's fit_baseline per KPI back to memory. Returns the Findings. `used_bootstrap` is recorded in the run entry.

- [ ] TDD: tests — memory round-trips a Baseline; tick 0 bootstraps (memory populated after, findings SAFE-ish, no crash); tick 1 with the saved memory detects current-vs-remembered (memory updated); `runs/latest/findings.json` written and audit verifies; two ticks leave memory persisted (file exists, has all KPIs). Implement → pass → commit `feat: baseline memory + memory-backed autonomous tick (remembers baselines across runs)`.

---

### Task 2: `orgpulse tick` CLI

**Files:** Modify `src/orgpulse/cli.py`; Test `tests/test_cli_tick.py`.

**Interfaces:** `orgpulse tick` — `tick_number: int = Option(0)`, `seed=42`, `--offline/--online` (default offline), `memory: Path = Option("runs/baseline_memory.json")`, `out: Path = Option("runs")`. Loads `BaselineMemory(memory)`, calls `run_tick(...)`, echoes severity/alert and whether it bootstrapped or used remembered baselines. Keyless default.

- [ ] TDD: test — `orgpulse tick --tick-number 0` then `--tick-number 1` (same memory file) → both exit 0, `runs/latest/findings.json` exists, memory file persists, second tick reports remembered (not bootstrap). Implement → pass → commit.

---

### Task 3: Hermes SKILL.md × 3 + cron runbook

**Files:** Create `hermes/skills/orgpulse-clinical/SKILL.md`, `hermes/skills/orgpulse-qc/SKILL.md`, `hermes/skills/orgpulse-supply/SKILL.md`, `hermes/CRON_RUNBOOK.md`.

**Interfaces:** Each SKILL.md uses the real Hermes frontmatter (`name`, `description`, `version`, `author`, `platforms`, `metadata.hermes.tags`, `prerequisites.commands: [uv]`) + a "When to Use" + a bash section that runs `orgpulse tick` and reads `runs/latest/report.md`, framed for that stream (clinical enrollment / QC pass-rate+OOS / supply lead-time). `CRON_RUNBOOK.md` documents the default daily-07:00 schedule, the split-plane model (Hermes local engine → committed `runs/` → Streamlit Cloud reads latest), Hermes memory usage for baselines, and both the Hermes-cron and a system-cron fallback invocation.

- [ ] Non-code deliverables: write the 4 Markdown files (validate frontmatter parses as YAML). Commit `docs: Hermes SKILL.md x3 + cron runbook (autonomous scheduling)`.

---

## Self-Review
- Hermes runs on a cron (default daily 07:00) → Task 3 runbook + `orgpulse tick`. ✓
- Remembers baselines across runs → Task 1 `BaselineMemory` + memory-backed `run_tick`. ✓
- 3× SKILL.md per domain → Task 3. ✓
- Writes run artifacts (findings/report/audit) + updates baseline → Tasks 1–2. ✓
- Deferred to Plan 7: Streamlit reads `runs/latest`; seeded multi-run history. No new deps; keyless; deterministic.
