# OrgPulse P4 — LangGraph Analysis DAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Assemble the per-run analysis pipeline as a LangGraph 4-node DAG — detect → forecast → domain-analyst (LLM) → recommendation-writer (LLM) — that consumes the generators, runs Plan-2 detection and Plan-3 forecasting, produces a structured findings JSON + a plain-language report + a threshold alert, and **always runs keyless** via an offline deterministic LLM stub (the public-demo default).

**Architecture:** A new `orgpulse.pipeline` subpackage. A `PipelineState` TypedDict carries the run through four nodes. The two LLM nodes call an `LLMClient` interface with two implementations: `OfflineLLM` (deterministic templates — the keyless default) and `ClaudeLLM` (Anthropic SDK, `claude-sonnet-5`). Recommendations are CAPA/ICH-E6(R3)-framed and every one carries `review_status="PENDING_HUMAN_REVIEW"` + a human-oversight disclaimer (decision-support, not automated decision). A `run` CLI executes the DAG and writes `findings.json` + `report.md`. **The whole suite runs offline/deterministic — no API key, no network.** The Claude adapter is unit-tested with a mocked client. Governance (hash-chained audit log, EU AI Act self-classification doc) is Plan 5 — this plan only stamps `review_status`/disclaimer into the schema.

**Tech Stack:** Python 3.11, uv, LangGraph, Anthropic SDK, pydantic (bundled via langgraph), pandas/numpy, Prophet + scikit-learn (from Plans 2–3), pytest.

## Global Constraints

- Python **3.11**; deps via **uv** (langgraph + anthropic already added). Use `uv run`. Ignore `VIRTUAL_ENV ... does not match`.
- **No new runtime deps** beyond langgraph/anthropic (already present).
- **Keyless & deterministic tests:** every test runs with `OfflineLLM` (no `Anthropic()` construction, no network). Never require `ANTHROPIC_API_KEY` in a test. The `ClaudeLLM` path is covered only by a unit test with a mocked SDK client.
- Model for `ClaudeLLM` is `claude-sonnet-5` (user-chosen; configurable via `OrgPulseConfig`/env). Do not hardcode a different model.
- Every recommendation MUST carry `review_status="PENDING_HUMAN_REVIEW"` and a human-oversight disclaimer. Findings JSON must be `json.dumps(..., allow_nan=False)`-serializable (reuse the Plan-2.6 `score: float|None` discipline — never emit NaN).
- Reuse Plan-2 detection (`orgpulse.detect`) and Plan-3 forecasting (`orgpulse.forecast`) — do not reimplement them.
- Commit style: `git -c user.name="naloo" -c user.email="mofasshara@hotmail.com" commit -m "<msg>"`; no Co-Authored-By.
- TDD throughout; never weaken a test; report BLOCKED with evidence if stuck.

---

### Task 1: Pipeline state + findings schema

**Files:**
- Create: `src/orgpulse/pipeline/__init__.py`
- Create: `src/orgpulse/pipeline/state.py`
- Test: `tests/pipeline/__init__.py` (empty), `tests/pipeline/test_state.py`

**Interfaces:**
- `orgpulse.pipeline.state.KpiFinding` — frozen dataclass: `kpi: str`, `severity: str` (Severity.name), `detail: str`, `latest_value: float | None`.
- `orgpulse.pipeline.state.Recommendation` — frozen dataclass: `kpi: str`, `text: str`, `capa_frame: str`, `review_status: str = "PENDING_HUMAN_REVIEW"`, `disclaimer: str`.
- `orgpulse.pipeline.state.Findings` — frozen dataclass: `run_id: str`, `generated_at: str` (ISO), `overall_severity: str`, `alert: bool`, `kpi_findings: list[KpiFinding]`, `forecast: dict`, `analysis: str`, `recommendations: list[Recommendation]`. Method `to_json_dict() -> dict` returning a plain, `allow_nan=False`-safe dict.
- `orgpulse.pipeline.state.PipelineState` — a `TypedDict` (total=False) with keys the nodes fill: `run_id`, `generated_at`, `clinical` (pandas DataFrame), `qc`, `supply`, `kpi_findings` (list[KpiFinding]), `overall_severity` (str), `forecast` (dict), `analysis` (str), `recommendations` (list[Recommendation]), `alert` (bool), `offline` (bool).
- `HUMAN_OVERSIGHT_DISCLAIMER: str` constant — a one-line decision-support / human-sign-off statement.

- [ ] **Step 1: Write the failing test**

`tests/pipeline/test_state.py`:
```python
import json

from orgpulse.pipeline.state import (
    KpiFinding, Recommendation, Findings, HUMAN_OVERSIGHT_DISCLAIMER,
)


def test_recommendation_defaults_to_human_review():
    r = Recommendation(kpi="enroll", text="t", capa_frame="CAPA-000", disclaimer=HUMAN_OVERSIGHT_DISCLAIMER)
    assert r.review_status == "PENDING_HUMAN_REVIEW"
    assert "human" in r.disclaimer.lower()


def test_findings_json_serializable_no_nan():
    f = Findings(
        run_id="r1", generated_at="2026-08-04T07:00:00", overall_severity="WARNING", alert=True,
        kpi_findings=[KpiFinding(kpi="enroll", severity="WARNING", detail="d", latest_value=None)],
        forecast={"enroll": {"h30": {"yhat": 12.3, "lower": 8.0, "upper": 17.0}}},
        analysis="a",
        recommendations=[Recommendation(kpi="enroll", text="t", capa_frame="CAPA-001",
                                        disclaimer=HUMAN_OVERSIGHT_DISCLAIMER)],
    )
    d = f.to_json_dict()
    json.dumps(d, allow_nan=False)                       # must not raise
    assert d["recommendations"][0]["review_status"] == "PENDING_HUMAN_REVIEW"
    assert d["alert"] is True and d["kpi_findings"][0]["latest_value"] is None
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/pipeline/test_state.py -v` (ModuleNotFound).

- [ ] **Step 3: Implement**

`src/orgpulse/pipeline/__init__.py`:
```python
"""Analysis pipeline: LangGraph 4-node DAG (detect -> forecast -> analyst -> recommend)."""
```

`src/orgpulse/pipeline/state.py`:
```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TypedDict

import pandas as pd

HUMAN_OVERSIGHT_DISCLAIMER = (
    "Decision support only — not an automated decision. A qualified human must "
    "review and approve before any action (EU AI Act Art. 14 human oversight)."
)


@dataclass(frozen=True)
class KpiFinding:
    kpi: str
    severity: str
    detail: str
    latest_value: float | None


@dataclass(frozen=True)
class Recommendation:
    kpi: str
    text: str
    capa_frame: str
    disclaimer: str
    review_status: str = "PENDING_HUMAN_REVIEW"


@dataclass(frozen=True)
class Findings:
    run_id: str
    generated_at: str
    overall_severity: str
    alert: bool
    kpi_findings: list[KpiFinding]
    forecast: dict
    analysis: str
    recommendations: list[Recommendation]

    def to_json_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "overall_severity": self.overall_severity,
            "alert": self.alert,
            "kpi_findings": [asdict(k) for k in self.kpi_findings],
            "forecast": self.forecast,
            "analysis": self.analysis,
            "recommendations": [asdict(r) for r in self.recommendations],
        }


class PipelineState(TypedDict, total=False):
    run_id: str
    generated_at: str
    offline: bool
    clinical: pd.DataFrame
    qc: pd.DataFrame
    supply: pd.DataFrame
    kpi_findings: list[KpiFinding]
    overall_severity: str
    forecast: dict
    analysis: str
    recommendations: list[Recommendation]
    alert: bool
```
Also create empty `tests/pipeline/__init__.py`.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/pipeline/test_state.py -v` then `uv run pytest -q`.

- [ ] **Step 5: Commit**
```bash
git add src/orgpulse/pipeline/ tests/pipeline/
git commit -m "feat: pipeline state + findings schema (human-oversight defaults, JSON-safe)"
```

---

### Task 2: LLM client interface + offline stub + Claude adapter

**Files:**
- Create: `src/orgpulse/pipeline/llm.py`
- Test: `tests/pipeline/test_llm.py`

**Interfaces:**
- `orgpulse.pipeline.llm.LLMClient` — `typing.Protocol` with `analyze(system: str, prompt: str) -> str` and `recommend(system: str, prompt: str, kpis: list[str]) -> list[dict]` (each dict: `{"kpi","text","capa_frame"}`).
- `orgpulse.pipeline.llm.OfflineLLM` — deterministic, keyless. `analyze` returns a templated paragraph derived from the prompt (no randomness). `recommend` returns one templated CAPA-framed dict per kpi. **This is the default and the only path tests exercise for the DAG.**
- `orgpulse.pipeline.llm.ClaudeLLM` — wraps `anthropic.Anthropic()`; `analyze` calls `client.messages.create(model=..., max_tokens=1024, system=system, messages=[{"role":"user","content":prompt}])` and returns the joined text blocks; `recommend` uses `output_config={"format":{"type":"json_schema","schema":{...}}}` to get a JSON list. Model defaults to `claude-sonnet-5` (constructor arg `model="claude-sonnet-5"`). The client is created lazily in `__init__`; if `anthropic` import or key is missing it raises — callers use `OfflineLLM` unless a key is present.
- `orgpulse.pipeline.llm.make_llm(offline: bool, model: str = "claude-sonnet-5") -> LLMClient` — returns `OfflineLLM()` if `offline` or no `ANTHROPIC_API_KEY` in env; else `ClaudeLLM(model=model)`.

- [ ] **Step 1: Write the failing test**

`tests/pipeline/test_llm.py`:
```python
import os

from orgpulse.pipeline.llm import OfflineLLM, make_llm


def test_offline_analyze_is_deterministic_and_keyless():
    llm = OfflineLLM()
    a = llm.analyze("sys", "enroll dropped 40%")
    b = llm.analyze("sys", "enroll dropped 40%")
    assert a == b and len(a) > 0


def test_offline_recommend_one_per_kpi_with_capa():
    recs = OfflineLLM().recommend("sys", "prompt", kpis=["enroll", "pass_rate"])
    assert len(recs) == 2
    assert {r["kpi"] for r in recs} == {"enroll", "pass_rate"}
    assert all("CAPA" in r["capa_frame"] for r in recs)
    assert all(r["text"] for r in recs)


def test_make_llm_defaults_to_offline_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(make_llm(offline=False), OfflineLLM)   # no key -> offline
    assert isinstance(make_llm(offline=True), OfflineLLM)


def test_claude_llm_uses_sonnet_and_parses(monkeypatch):
    # Mock the Anthropic SDK so no network/key is needed.
    import orgpulse.pipeline.llm as m

    class _Block:
        type = "text"; text = "analysis text"

    class _Msg:
        content = [_Block()]

    calls = {}

    class _Messages:
        def create(self, **kw):
            calls.update(kw); return _Msg()

    class _FakeClient:
        def __init__(self, *a, **k): self.messages = _Messages()

    monkeypatch.setattr(m, "_anthropic_client", lambda: _FakeClient())
    llm = m.ClaudeLLM(model="claude-sonnet-5")
    out = llm.analyze("sys", "prompt")
    assert out == "analysis text"
    assert calls["model"] == "claude-sonnet-5"
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** `src/orgpulse/pipeline/llm.py`
```python
from __future__ import annotations

import json
import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def analyze(self, system: str, prompt: str) -> str: ...
    def recommend(self, system: str, prompt: str, kpis: list[str]) -> list[dict]: ...


class OfflineLLM:
    """Deterministic, keyless LLM stub — the public-demo default."""

    def analyze(self, system: str, prompt: str) -> str:
        head = prompt.strip().splitlines()[0] if prompt.strip() else "the monitored KPIs"
        return (
            "Automated interpretation (offline mode): "
            f"the pipeline observed — {head}. These movements are assessed against "
            "GxP quality expectations; where a KPI is Warning/Critical the trend and "
            "forecast suggest a deviation trigger may apply and should be reviewed by "
            "the quality team."
        )

    def recommend(self, system: str, prompt: str, kpis: list[str]) -> list[dict]:
        out = []
        for i, kpi in enumerate(kpis, start=1):
            out.append({
                "kpi": kpi,
                "capa_frame": f"CAPA-{i:03d}",
                "text": (
                    f"Open a CAPA review for '{kpi}': confirm the signal against source "
                    "data, assess whether an ICH E6(R3) deviation threshold is crossed, "
                    "and, if confirmed, document root cause and corrective action. "
                    "Escalate to the responsible quality owner for sign-off."
                ),
            })
        return out


def _anthropic_client():  # indirection so tests can monkeypatch
    import anthropic

    return anthropic.Anthropic()


class ClaudeLLM:
    def __init__(self, model: str = "claude-sonnet-5") -> None:
        self.model = model
        self._client = _anthropic_client()

    def analyze(self, system: str, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self.model, max_tokens=1024, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

    def recommend(self, system: str, prompt: str, kpis: list[str]) -> list[dict]:
        schema = {
            "type": "object",
            "properties": {
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kpi": {"type": "string"},
                            "capa_frame": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["kpi", "capa_frame", "text"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["recommendations"],
            "additionalProperties": False,
        }
        msg = self._client.messages.create(
            model=self.model, max_tokens=2048, system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return json.loads(text)["recommendations"]


def make_llm(offline: bool, model: str = "claude-sonnet-5") -> LLMClient:
    if offline or not os.environ.get("ANTHROPIC_API_KEY"):
        return OfflineLLM()
    return ClaudeLLM(model=model)
```

- [ ] **Step 4: Run to verify pass** — targeted then `uv run pytest -q`.

- [ ] **Step 5: Commit**
```bash
git add src/orgpulse/pipeline/llm.py tests/pipeline/test_llm.py
git commit -m "feat: LLM client interface + offline stub + Claude(sonnet-5) adapter (keyless default)"
```

---

### Task 3: Detect + forecast nodes

**Files:**
- Create: `src/orgpulse/pipeline/nodes.py`
- Test: `tests/pipeline/test_nodes_analytics.py`

**Interfaces:**
- `orgpulse.pipeline.nodes.detect_node(state: PipelineState) -> dict` — for each monitored KPI (hero: clinical daily enrollment; QC pass_rate & calibration_offset; supply lead_time_days), split the KPI series into a baseline window and a recent window (baseline `>= 60` per Plan 2.6), run `detect_kpi`, and record a `KpiFinding` (severity = `overall_severity(...).name`, detail from the CUSUM/zscore results, `latest_value` = last recent value). Returns `{"kpi_findings": [...], "overall_severity": <max severity name>}`. Uses `series_from_generator` for the clinical hero.
- `orgpulse.pipeline.nodes.forecast_node(state: PipelineState) -> dict` — forecast the hero KPI (clinical daily enrollment) 30/60/90 days with `forecast_kpi`; returns `{"forecast": {"daily_enrollment": {"h30":{...},"h60":{...},"h90":{...}}}}` where each horizon is `{yhat, lower, upper}` floats. (Only the hero stream is forecast — the others are lighter, per the spec.)
- Both nodes are deterministic and keyless (no LLM).

- [ ] **Step 1: Write the failing test**

`tests/pipeline/test_nodes_analytics.py`:
```python
from datetime import date

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical
from orgpulse.generators.qc import generate_qc
from orgpulse.generators.supply import generate_supply
from orgpulse.pipeline.nodes import detect_node, forecast_node


def _state(days=260):
    return {
        "clinical": generate_clinical(date(2026, 1, 1), days, make_rng(42)),
        "qc": generate_qc(date(2026, 1, 1), days, make_rng(43)),
        "supply": generate_supply(date(2026, 1, 1), days, make_rng(44)),
    }


def test_detect_node_produces_findings_and_overall_severity():
    out = detect_node(_state())
    assert out["kpi_findings"] and out["overall_severity"] in {"SAFE", "WARNING", "CRITICAL", "DATA_QUALITY"}
    kpis = {f.kpi for f in out["kpi_findings"]}
    assert "daily_enrollment" in kpis            # hero present
    for f in out["kpi_findings"]:
        assert f.severity in {"SAFE", "WARNING", "CRITICAL", "DATA_QUALITY"}


def test_forecast_node_hero_horizons():
    out = forecast_node(_state())
    fc = out["forecast"]["daily_enrollment"]
    for h in ("h30", "h60", "h90"):
        assert fc[h]["lower"] <= fc[h]["yhat"] <= fc[h]["upper"]


def test_nodes_are_deterministic():
    a = detect_node(_state())["overall_severity"]
    b = detect_node(_state())["overall_severity"]
    assert a == b
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** `src/orgpulse/pipeline/nodes.py` (detect + forecast portion)

Use `orgpulse.detect.orchestrator.detect_kpi`/`overall_severity`, `orgpulse.detect.types.Severity`, `orgpulse.forecast.prophet_forecaster.series_from_generator`/`forecast_kpi`, and `orgpulse.pipeline.state.KpiFinding`. For each KPI build a daily series; baseline = `series[: -30]` (must be ≥60 long — use ≥90 windows), recent = `series[-30:]`. For QC/supply KPIs that are one-row-per-date, use the column directly; for clinical, aggregate via `series_from_generator(df, "date", "daily_enrollment")["y"]`. Guard: if a series is shorter than 90+30, skip that KPI. `forecast_node` builds the hero `ds,y` frame and calls `forecast_kpi(..., horizons=(30,60,90))`, then reads `fc.at_horizon(30/60/90)`.

(Write the full function bodies during implementation, following the interface above; keep them deterministic.)

- [ ] **Step 4: Run to verify pass** — targeted then `uv run pytest -q`.

- [ ] **Step 5: Commit**
```bash
git add src/orgpulse/pipeline/nodes.py tests/pipeline/test_nodes_analytics.py
git commit -m "feat: detect + forecast pipeline nodes (deterministic, hero forecast)"
```

---

### Task 4: LLM nodes (analyst + recommendation writer)

**Files:**
- Modify: `src/orgpulse/pipeline/nodes.py` (add the two LLM nodes)
- Test: `tests/pipeline/test_nodes_llm.py`

**Interfaces:**
- `orgpulse.pipeline.nodes.analyst_node(state, llm: LLMClient) -> dict` — builds a prompt from `kpi_findings` + `forecast`, calls `llm.analyze(...)`, returns `{"analysis": <str>}`.
- `orgpulse.pipeline.nodes.recommendation_node(state, llm: LLMClient) -> dict` — splits flagged KPIs by severity: **anomaly** (WARNING/CRITICAL) go to `llm.recommend(...)` as CAPA-framed recommendations; **DATA_QUALITY** KPIs get a distinct deterministic data-quality recommendation (`capa_frame="DATA-QUALITY"`, "investigate the data source, do NOT open a CAPA on the process") — a data problem is not a process deviation. If neither group is flagged, the hero (`daily_enrollment`) is used for the anomaly path. Every `Recommendation` carries `disclaimer=HUMAN_OVERSIGHT_DISCLAIMER` + `review_status="PENDING_HUMAN_REVIEW"`. `alert = overall_severity in ("CRITICAL", "DATA_QUALITY")` (a systemic data failure is also alert-worthy — you cannot assess). Returns `{"recommendations": [...], "alert": <bool>}`.
- Both take an injected `llm` so tests use `OfflineLLM` (keyless).

- [ ] **Step 1: Write the failing test**

`tests/pipeline/test_nodes_llm.py`:
```python
from orgpulse.pipeline.llm import OfflineLLM
from orgpulse.pipeline.nodes import analyst_node, recommendation_node
from orgpulse.pipeline.state import KpiFinding, HUMAN_OVERSIGHT_DISCLAIMER


def _state():
    return {
        "kpi_findings": [
            KpiFinding(kpi="daily_enrollment", severity="CRITICAL", detail="high CUSUM", latest_value=73.0),
            KpiFinding(kpi="pass_rate", severity="SAFE", detail="stable", latest_value=0.98),
        ],
        "overall_severity": "CRITICAL",
        "forecast": {"daily_enrollment": {"h30": {"yhat": 60.0, "lower": 40.0, "upper": 80.0}}},
    }


def test_analyst_node_returns_text():
    out = analyst_node(_state(), OfflineLLM())
    assert isinstance(out["analysis"], str) and len(out["analysis"]) > 0


def test_recommendation_node_flags_and_gates():
    out = recommendation_node(_state(), OfflineLLM())
    assert out["alert"] is True                       # overall CRITICAL
    recs = out["recommendations"]
    assert recs and all(r.review_status == "PENDING_HUMAN_REVIEW" for r in recs)
    assert all(r.disclaimer == HUMAN_OVERSIGHT_DISCLAIMER for r in recs)
    # only the flagged (non-SAFE) KPI gets a recommendation
    assert {r.kpi for r in recs} == {"daily_enrollment"}
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** the two nodes in `nodes.py` per the interface (flagged = `[f.kpi for f in kpi_findings if f.severity not in ("SAFE",)]`; if empty, fall back to the hero `daily_enrollment`). Build a readable prompt string from findings + forecast.

- [ ] **Step 4: Run to verify pass** — targeted then `uv run pytest -q`.

- [ ] **Step 5: Commit**
```bash
git add src/orgpulse/pipeline/nodes.py tests/pipeline/test_nodes_llm.py
git commit -m "feat: analyst + recommendation LLM nodes (human-gated, CAPA-framed)"
```

---

### Task 5: Assemble the LangGraph DAG + findings/report/alert

**Files:**
- Create: `src/orgpulse/pipeline/graph.py`
- Test: `tests/pipeline/test_graph.py`

**Interfaces:**
- `orgpulse.pipeline.graph.build_graph(llm: LLMClient)` — constructs a `StateGraph(PipelineState)` with nodes `detect` → `forecast` → `analyst` → `recommend` (analyst/recommend close over `llm`), edges `START→detect→forecast→analyst→recommend→END`; returns the compiled graph.
- `orgpulse.pipeline.graph.run_pipeline(clinical, qc, supply, *, offline=True, run_id=None, now=None, model="claude-sonnet-5") -> Findings` — seeds the state, invokes the graph (with `make_llm(offline, model)`), and assembles a `Findings` object from the final state (`generated_at` from `now` or `datetime.now().isoformat(timespec="seconds")`; `run_id` from arg or a timestamp-derived id).
- `orgpulse.pipeline.graph.render_report(findings: Findings) -> str` — plain-language Markdown: title, run id/time, overall severity + alert banner, a KPI table (kpi · severity · detail), the analysis paragraph, forecast (30/60/90 for the hero), and the recommendations each with its CAPA frame + `review_status` + disclaimer.

- [ ] **Step 1: Write the failing test**

`tests/pipeline/test_graph.py`:
```python
import json
from datetime import date

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical
from orgpulse.generators.qc import generate_qc
from orgpulse.generators.supply import generate_supply
from orgpulse.pipeline.graph import run_pipeline, render_report
from orgpulse.pipeline.state import Findings, HUMAN_OVERSIGHT_DISCLAIMER


def _streams(days=260):
    return (
        generate_clinical(date(2026, 1, 1), days, make_rng(42)),
        generate_qc(date(2026, 1, 1), days, make_rng(43)),
        generate_supply(date(2026, 1, 1), days, make_rng(44)),
    )


def test_run_pipeline_offline_end_to_end():
    c, q, s = _streams()
    f = run_pipeline(c, q, s, offline=True, run_id="run-1", now="2026-08-04T07:00:00")
    assert isinstance(f, Findings)
    assert f.run_id == "run-1" and f.overall_severity in {"SAFE", "WARNING", "CRITICAL", "DATA_QUALITY"}
    assert f.kpi_findings and f.recommendations and f.analysis
    # JSON-safe, human-gated:
    json.dumps(f.to_json_dict(), allow_nan=False)
    assert all(r.review_status == "PENDING_HUMAN_REVIEW" for r in f.recommendations)


def test_run_pipeline_is_deterministic_offline():
    c, q, s = _streams()
    a = run_pipeline(c, q, s, offline=True, run_id="r", now="2026-08-04T07:00:00").to_json_dict()
    b = run_pipeline(c, q, s, offline=True, run_id="r", now="2026-08-04T07:00:00").to_json_dict()
    assert a == b


def test_render_report_contains_disclaimer_and_alert():
    c, q, s = _streams()
    f = run_pipeline(c, q, s, offline=True, run_id="r", now="2026-08-04T07:00:00")
    md = render_report(f)
    assert "OrgPulse" in md and "PENDING_HUMAN_REVIEW" in md
    assert HUMAN_OVERSIGHT_DISCLAIMER.split(" — ")[0] in md
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** `graph.py`: `build_graph` wires the four nodes (wrap `analyst_node`/`recommendation_node` in closures that pass `llm`); `run_pipeline` builds `make_llm`, invokes, and maps the final state → `Findings`; `render_report` formats Markdown. Keep `to_json_dict` NaN-free (floats already finite from Prophet; guard any `None`).

- [ ] **Step 4: Run to verify pass** — targeted then `uv run pytest -q`.

- [ ] **Step 5: Commit**
```bash
git add src/orgpulse/pipeline/graph.py tests/pipeline/test_graph.py
git commit -m "feat: assemble LangGraph 4-node DAG + findings/report/alert (offline deterministic)"
```

---

### Task 6: `orgpulse run` CLI (writes findings.json + report.md)

**Files:**
- Modify: `src/orgpulse/cli.py` (add a `run` command)
- Test: `tests/test_cli_run.py`

**Interfaces:**
- `orgpulse.cli.run(days=260, seed=42, offline=True, out=Path("runs/latest"))` — generates the three streams, calls `run_pipeline(..., offline=offline)`, writes `out/findings.json` (`json.dumps(findings.to_json_dict(), indent=2, allow_nan=False)`) and `out/report.md` (`render_report(findings)`), and echoes the overall severity + whether an alert fired. Default `offline=True` so it runs keyless.
- `orgpulse.cli.write_run(out: Path, days: int, seed: int, offline: bool) -> Findings` — the testable core.

- [ ] **Step 1: Write the failing test**

`tests/test_cli_run.py`:
```python
import json
from pathlib import Path

from typer.testing import CliRunner

from orgpulse.cli import app, write_run


def test_write_run_offline_writes_artifacts(tmp_path: Path):
    f = write_run(out=tmp_path, days=200, seed=42, offline=True)
    findings = json.loads((tmp_path / "findings.json").read_text())
    assert findings["run_id"] and "recommendations" in findings
    assert (tmp_path / "report.md").exists()
    assert f.overall_severity in {"SAFE", "WARNING", "CRITICAL", "DATA_QUALITY"}


def test_cli_run_offline(tmp_path: Path):
    result = CliRunner().invoke(app, ["run", "--days", "200", "--out", str(tmp_path), "--offline"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "findings.json").exists() and (tmp_path / "report.md").exists()
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** the `run` command + `write_run` in `cli.py` (import `run_pipeline`/`render_report` lazily inside the function so the heavy pipeline import doesn't slow `orgpulse generate`). Typer option `--offline/--online` (default offline). `days` keeps the `min=1` guard.

- [ ] **Step 4: Run + smoke** — `uv run pytest tests/test_cli_run.py -v`; `uv run pytest -q` (full suite); `uv run orgpulse run --days 120 --out /tmp/orgpulse-run --offline` (prints severity; writes both files).

- [ ] **Step 5: Commit**
```bash
git add src/orgpulse/cli.py tests/test_cli_run.py
git commit -m "feat: orgpulse run CLI — execute DAG, write findings.json + report.md (offline default)"
```

---

## Self-Review

**Spec coverage (P4):**
- 4-node LangGraph DAG (detect → forecast → analyst → recommend) → Tasks 3–5. ✓
- Domain-analyst LLM node → Task 4. ✓
- Recommendation-writer LLM node (CAPA/ICH E6(R3), human-gated) → Task 4. ✓
- Offline deterministic stub (keyless demo path) → Task 2, default everywhere. ✓
- Findings JSON + report + threshold alert → Tasks 1 & 5. ✓
- `orgpulse run` one-command execution → Task 6. ✓
- `review_status` + human-oversight disclaimer on every recommendation → Tasks 1 & 4 (schema default + node). ✓
- Deferred to Plan 5 (governance): hash-chained audit log, EU AI Act self-classification doc, trace reconstruction. Deferred to Plan 5/6: wiring Isolation Forest into a multivariate node (still un-orchestrated), forecasting the non-hero streams.

**Placeholder scan:** Tasks 3 & 4 leave node *bodies* to the implementer but fully specify the interface, inputs, outputs, and determinism — no hidden logic. Everything else has real code. **Type consistency:** `KpiFinding`/`Recommendation`/`Findings`/`PipelineState` defined once (Task 1) and consumed by nodes/graph/CLI; `LLMClient` protocol implemented by `OfflineLLM`/`ClaudeLLM`; model string `claude-sonnet-5` used consistently; `to_json_dict()` is the single JSON surface (NaN-free). **Keyless:** every test uses `OfflineLLM`; `ClaudeLLM` is mocked; no test needs `ANTHROPIC_API_KEY`.
