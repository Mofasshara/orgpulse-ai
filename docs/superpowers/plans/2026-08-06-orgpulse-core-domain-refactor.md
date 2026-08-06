# OrgPulse Core/Domain Refactor — Implementation Plan

> **For agentic workers:** Behavior-preserving refactor. Execute with tight control (inline),
> not subagents — the golden-master gate must run after every task. Steps use `- [ ]` checkboxes.

**Goal:** Extract OrgPulse's domain-agnostic engine into a `core` layer and move the pharma
specifics behind a single `Domain` interface, so a second domain (banking) can be added without
touching the engine — and so the "configurable platform" claim becomes true. **Pharma behavior
must not change.**

**Architecture:** One repo evolves into `core` (generic: detect/forecast/pipeline/agent/audit/
dashboard-loader, parameterized by a `Domain`) + `domains/pharma` (generators, KPI specs,
LLM framing, compliance). The live pharma Streamlit demo keeps working at the same URL/entry.

**Tech Stack:** unchanged — Python 3.11, uv, LangGraph, Prophet, scikit-learn, Streamlit/Plotly.
No new dependencies.

## Global Constraints (every task obeys these)

- **THE GATE — run after every task, non-negotiable:** `uv run pytest -q` is fully green AND
  `uv run python scripts/seed_runs.py --force` then `git status --short runs` is **empty**
  (byte-identical reproduction). If either fails, STOP and fix (systematic-debugging) before the
  next task. Byte-identical `runs/` is the mechanical proof that pharma behavior is unchanged.
- **Behavior-preserving:** no change to any pharma output, severity, forecast, audit, or dashboard
  render. New abstractions are added; pharma is routed through them producing identical results.
- Determinism, keyless/offline tests, split-plane (dashboard imports no `orgpulse`), no
  Co-Authored-By, git identity name "naloo"/email "mofasshara@hotmail.com" — all unchanged.
- **Honesty:** docs describe a `Domain` *interface* (code: generators + KPI specs + detection
  strategies + framing), NOT a "swap a YAML file" platform. Do not reintroduce that overclaim.
- Commit after each task (the ledger + a clean bisectable history matter for a refactor).

## File Structure (target)

```
src/orgpulse/
  core/                     # domain-AGNOSTIC engine (parameterized by a Domain)
    domain.py               # NEW: Domain, KpiSpec, DetectionMode
    detect/  forecast/  pipeline/  agent/  governance/  config.py   # moved, parameterized
  domains/
    pharma/
      __init__.py           # builds the pharma Domain instance
      generators.py         # clinical/qc/supply (moved)
      framing.py            # GxP/ICH-E6 analyst+recommend prompts, CAPA language
      compliance.py         # EU AI Act mapping content for pharma
  cli.py                    # domain-aware entrypoints
streamlit_app/app.py        # hero derived from artifacts (no hardcoded KPI) — stays split-plane
```

Note: physical file moves happen LAST (Task 7) to keep diffs reviewable; Tasks 1–6 introduce and
wire the abstraction in place first.

---

### Task 1: Golden-master baseline + the gate script

**Files:** Create `scripts/check_no_regression.sh` (or inline commands documented in the plan).

- [ ] **Step 1:** Record the baseline: `uv run pytest -q` (expect 162 passed) and
  `find runs -type f | sort | xargs shasum | shasum` — save the hash. This is the golden master.
- [ ] **Step 2:** Write a one-line gate: pytest green AND `seed_runs.py --force` leaves
  `git status --short runs` empty. Confirm it passes NOW (before any change).
- [ ] **Step 3:** Commit (`chore: document behavior-preserving refactor gate`).

### Task 2: Define the `Domain` interface (pure addition, zero wiring)

**Files:** Create `src/orgpulse/core/domain.py`. Test: `tests/core/test_domain.py`.

**Interfaces (Produces):**
```python
class DetectionMode(str, Enum):
    TIMESERIES = "timeseries"   # z-score + calibrated CUSUM (only mode wired in this refactor)
    THRESHOLD  = "threshold"    # hard min/max rules   (data model only; wired in the banking build)
    DATEMATH   = "datemath"     # days-until-deadline  (data model only; wired in the banking build)

@dataclass(frozen=True)
class KpiSpec:
    name: str
    domain_bounds: tuple[float | None, float | None]   # (lo, hi) inclusive; None = unbounded
    detection_mode: DetectionMode = DetectionMode.TIMESERIES
    is_hero: bool = False
    thresholds: dict | None = None                      # THRESHOLD mode only

@dataclass(frozen=True)
class Domain:
    name: str
    generate: Callable[..., dict[str, "pd.DataFrame"]]  # (start, days, seed) -> {stream: df}
    extract:  Callable[[dict, str], "np.ndarray"]       # (streams, kpi_name) -> 1-D series
    kpi_specs: list[KpiSpec]
    analyst_system: str
    recommend_system: str
    compliance: dict                                    # feeds governance/classification rendering
    def hero(self) -> str: ...                          # the is_hero KPI name
    def bounds(self) -> dict[str, tuple]: ...           # {kpi: (lo,hi)} — replaces hardcoded KPI_DOMAINS
```
- [ ] Test: a Domain with two KpiSpecs (one hero) returns the right `hero()` and `bounds()`;
  exactly one hero is required (raise if zero/multiple). Implement → tests pass → commit.
- [ ] **GATE.**

### Task 3: Build the pharma `Domain` instance (addition; still nothing consumes it)

**Files:** Create `src/orgpulse/domains/pharma/__init__.py` (+ re-export existing generators).
Test: `tests/domains/test_pharma_domain.py`.

- [ ] Construct `PHARMA = Domain(name="pharma", generate=<wraps generate_clinical/qc/supply>,
  extract=<clinical→validated_clinical_daily; qc/supply→column>, kpi_specs=[
  daily_enrollment(bounds=(0,None), hero=True), pass_rate(0,1), calibration_offset(None,None),
  lead_time_days(0,None)], analyst_system=<current GxP text>, recommend_system=<current>,
  compliance=<current EU AI Act dict>)`.
- [ ] Test: `PHARMA.hero()=="daily_enrollment"`; `PHARMA.bounds()` equals the current hardcoded
  `KPI_DOMAINS` exactly; `PHARMA.extract(PHARMA.generate(...), k)` matches today's series for each k.
- [ ] **GATE.**

### Task 4: Route detect/validation through the Domain (behavior identical for pharma)

**Files:** Modify `src/orgpulse/detect/validation.py`, `src/orgpulse/pipeline/nodes.py`.

- [ ] Make `kpi_domain_reason`/`baseline_domain_feasible` read bounds from a passed-in mapping
  (default = `PHARMA.bounds()`), so `KPI_DOMAINS` becomes the pharma Domain's bounds — same values.
- [ ] Add a detection-mode dispatch in the detect path keyed on `KpiSpec.detection_mode`; wire
  ONLY `TIMESERIES` (the existing z-score+CUSUM). `THRESHOLD`/`DATEMATH` raise a clear
  `NotImplementedError("wired in the banking build")` — pharma never hits them.
- [ ] Test: existing detection tests unchanged; add a test that a THRESHOLD KpiSpec raises the
  clear error (documents the seam). **GATE (byte-identical is the key check here).**

### Task 5: Parameterize the pipeline + agent by `Domain` (the core change)

**Files:** Modify `src/orgpulse/pipeline/nodes.py` (detect_node, forecast_node, analyst/recommend
prompts), `src/orgpulse/agent/tick.py` (`_kpi_series`), `src/orgpulse/pipeline/graph.py`
(run_pipeline/render_report), `src/orgpulse/cli.py`.

- [ ] `detect_node`/`_kpi_series` iterate `domain.kpi_specs` + `domain.extract(...)` instead of
  hardcoded KPI names. `forecast_node` forecasts `domain.hero()`. Analyst/recommend nodes pull
  system prompts from `domain.analyst_system`/`recommend_system`. Default domain = `PHARMA` so
  every existing call site is unchanged in behavior.
- [ ] `run_pipeline(clinical, qc, supply, ...)` keeps its signature (pharma), internally building
  the streams dict for `PHARMA`; add an optional `domain=PHARMA` param.
- [ ] Test: full suite green; **GATE — byte-identical `runs/` is the proof this didn't change
  pharma.** This is the highest-risk task; if the seed diff is non-empty, bisect the change.

### Task 6: Domain-agnostic dashboard + governance rendering

**Files:** Modify `streamlit_app/app.py`, `src/orgpulse/pipeline/graph.py` (render_report),
`src/orgpulse/governance/classification.py`.

- [ ] `app.py`: derive the hero from the artifact (`findings["forecast"]` key) instead of the
  hardcoded `HERO="daily_enrollment"`; render whatever KPIs are in `findings` (it already reads
  JSON — keep split-plane, import no orgpulse). Report/classification text sourced from the
  domain's framing/compliance.
- [ ] Test: AppTest renders the committed CRITICAL run identically; `verify_run` still True;
  **GATE.**

### Task 7: Physical layout move (core/ + domains/) + honest docs

**Files:** Move engine modules under `src/orgpulse/core/`, pharma specifics under
`src/orgpulse/domains/pharma/`; fix imports. Update `README.md`, `docs/ARCHITECTURE.md`.

- [ ] Move files; update imports; `uv run pytest -q` green; **GATE.**
- [ ] Docs: describe the `Domain` interface accurately (generators + KPI specs + detection
  strategies + framing/compliance), add a "how to add a domain" section. **Remove any
  "configurable via a YAML file / swap a config" wording** — it's a code interface. Keep the
  live-demo link. Commit + push.

## Self-Review

- Behavior-preserving proven by golden master (byte-identical `runs/` + 162 tests) after **every**
  task — this is the answer to "did the refactor break something that was right." ✓
- Fail-safe spine (nodes.py) changes are gated by the existing fail-safe tests from all 9 rounds. ✓
- `Domain` interface is banking-ready (detection_mode + thresholds fields) but only TIMESERIES is
  wired — THRESHOLD/DATEMATH implemented in the banking build, honestly marked. ✓
- Split-plane preserved (dashboard still imports no orgpulse; hero read from artifacts). ✓
- Overclaim removed; docs match code. ✓
- New surface to review afterward: `core/domain.py` + the mode dispatch + the pharma Domain
  wiring. The MOVED code is unchanged behavior (proven), so the Codex re-review can focus on the
  new abstraction, not re-review the whole engine.
