"""Proof that the engine is genuinely domain-agnostic: a minimal NON-pharma domain must run the
whole stack (tick, forecast, verify, offline framing, and the LangGraph pipeline) with no engine
changes. This is the guard against re-overclaiming "banking can be added without touching the
engine" — it exercises exactly what Codex's BANK probe exposed."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

from orgpulse.agent.baseline_memory import BaselineMemory
from orgpulse.agent.tick import run_tick
from orgpulse.core.domain import DetectionMode, Domain, KpiSpec
from orgpulse.pipeline.graph import run_pipeline
from orgpulse.pipeline.nodes import detect_node
from streamlit_app import data


def _demo_generate(start, days, seed):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days)
    return {"metrics": pd.DataFrame({
        "date": dates,
        "widget_rate": np.clip(0.9 + rng.normal(0, 0.02, days), 0.0, 1.0),   # in [0,1]
        "queue_len": np.abs(50 + rng.normal(0, 5, days)),                    # >= 0
    })}


def _demo_extract(streams, kpi):
    return streams["metrics"][kpi].to_numpy()


def _demo_hero_frame(streams):
    m = streams["metrics"]
    return pd.DataFrame({"ds": pd.to_datetime(m["date"]), "y": m["widget_rate"].to_numpy(dtype=float)})


DEMO = Domain(
    name="demo",
    generate=_demo_generate,
    extract=_demo_extract,
    kpi_specs=(KpiSpec("widget_rate", (0.0, 1.0), is_hero=True), KpiSpec("queue_len", (0.0, None))),
    analyst_system="You are an operations analyst.",
    recommend_system="You are an operations reviewer.",
    hero_frame=_demo_hero_frame,
)


def test_second_domain_runs_the_full_tick_stack(tmp_path):
    f = run_tick(BaselineMemory(tmp_path / "m.json"), 0, out_root=tmp_path / "runs",
                 now="2026-01-01T00:00:00", offline=True, domain=DEMO)

    kpis = {k.kpi for k in f.kpi_findings}
    assert kpis == {"widget_rate", "queue_len"}                 # domain KPIs, not pharma
    assert "daily_enrollment" not in kpis
    assert f.overall_severity in {"SAFE", "WARNING", "CRITICAL", "DATA_QUALITY"}

    latest = tmp_path / "runs" / "latest"
    fj = json.loads((latest / "findings.json").read_text())
    assert "widget_rate" in fj["forecast"]                       # forecast uses the domain hero
    assert "daily_enrollment" not in fj["forecast"]

    ok, reason = data.verify_run(latest)                         # verifier is domain-agnostic
    assert ok is True, reason

    # Offline framing is the DOMAIN's, not pharma GxP/ICH/CAPA language.
    blob = (latest / "report.md").read_text() + fj["analysis"] + " ".join(r["text"] for r in fj["recommendations"])
    for pharma_word in ("GxP", "ICH E6", "quality team"):
        assert pharma_word not in blob, pharma_word


def test_second_domain_traverses_the_langgraph_pipeline(tmp_path):
    """Codex's exact failure: build_graph(...).invoke({"loans": ...}) dropped the undeclared stream.
    With the generic `streams` state key, a non-pharma domain's streams now traverse the graph."""
    from orgpulse.pipeline.graph import build_graph
    from orgpulse.pipeline.llm import OfflineLLM

    streams = DEMO.generate(date(2026, 1, 1), 260, 0)
    final = build_graph(OfflineLLM(DEMO), DEMO).invoke({"streams": streams, "offline": True})
    sev = {f.kpi: f.severity for f in final["kpi_findings"]}
    assert set(sev) == {"widget_rate", "queue_len"}             # the demo streams reached detection
    assert "DATA_QUALITY" not in sev.values()                   # valid data assessed, not "missing stream"


def test_second_domain_run_pipeline_public_api():
    """r11 #1: the public run_pipeline must accept a GENERIC streams dict (was pharma-only, so a
    second domain's streams were relabelled and lost). Mutation: revert to (clinical,qc,supply)."""
    streams = DEMO.generate(date(2026, 1, 1), 260, 0)
    f = run_pipeline(streams, offline=True, run_id="d", now="2026-01-01T00:00:00", domain=DEMO)
    sev = {k.kpi: k.severity for k in f.kpi_findings}
    assert set(sev) == {"widget_rate", "queue_len"} and "DATA_QUALITY" not in sev.values()
    assert "widget_rate" in f.forecast


def test_parity_detection_dq_suppresses_forecast(tmp_path):
    """Mutation-constrains the detect/forecast PARITY guard: extract() returns NaN for the hero
    (detection -> DATA_QUALITY) while hero_frame() stays clean. NO forecast may be published."""
    def bad_extract(streams, kpi):
        arr = _demo_extract(streams, kpi).astype(float)
        return np.full_like(arr, np.nan) if kpi == "widget_rate" else arr

    dom = replace(DEMO, extract=bad_extract)   # hero_frame unchanged (clean)
    f = run_tick(BaselineMemory(tmp_path / "m.json"), 0, out_root=tmp_path / "r",
                 now="2026-01-01T00:00:00", offline=True, domain=dom)
    assert {k.kpi: k.severity for k in f.kpi_findings}["widget_rate"] == "DATA_QUALITY"
    assert "widget_rate" not in f.forecast     # parity: a DQ hero is never forecast


def test_forecast_rejects_corrupt_hero_dates(tmp_path):
    """r11 #2: hero_frame with NaT or duplicate ds must yield NO forecast (never a crash or a
    bogus one). Mutation: drop the ds validation in forecast_node."""
    def nat(streams):
        fr = _demo_hero_frame(streams); fr.loc[0, "ds"] = pd.NaT; return fr

    def dup(streams):
        fr = _demo_hero_frame(streams); fr.loc[1, "ds"] = fr.loc[0, "ds"]; return fr

    for builder in (nat, dup):
        dom = replace(DEMO, hero_frame=builder)
        f = run_tick(BaselineMemory(tmp_path / f"m_{builder.__name__}.json"), 0,
                     out_root=tmp_path / f"r_{builder.__name__}", now="2026-01-01T00:00:00",
                     offline=True, domain=dom)                  # must not raise
        assert "widget_rate" not in f.forecast


def test_recommendation_fallback_and_prompts_use_domain_hero(tmp_path):
    """r11 #2/#3: the SAFE hero-fallback recommendation is for the DOMAIN hero; the engine's
    analyst prompt names the domain hero and the recommend prompt is domain-neutral (no
    'daily_enrollment'/'CAPA'). Mutation: revert either to the hardcoded pharma value."""
    from orgpulse.pipeline.nodes import analyst_node, recommendation_node

    f = run_tick(BaselineMemory(tmp_path / "m.json"), 0, out_root=tmp_path / "r",
                 now="2026-01-01T00:00:00", offline=True, domain=DEMO)
    assert any(r.kpi == "widget_rate" for r in f.recommendations)
    assert all(r.kpi != "daily_enrollment" for r in f.recommendations)

    class Recorder:
        def __init__(self): self.prompts = []
        def analyze(self, system, prompt): self.prompts.append(prompt); return "x"
        def recommend(self, system, prompt, kpis): self.prompts.append(prompt); return DEMO.offline_recommend(kpis)

    r = Recorder()
    state = {"kpi_findings": [], "overall_severity": "CRITICAL",
             "forecast": {"widget_rate": {"h30": {"yhat": 1, "lower": 0, "upper": 2}}}}
    analyst_node(state, r, DEMO)
    recommendation_node(state, r, DEMO)
    joined = " ".join(r.prompts)
    assert "widget_rate" in joined
    assert "daily_enrollment" not in joined and "CAPA" not in joined


# --- A multi-mode domain: a TIMESERIES hero + a THRESHOLD KPI + a DATEMATH KPI ---------------

def _mm_generate(start, days, seed):
    rng = np.random.default_rng(seed)
    return {"book": pd.DataFrame({
        "date": pd.date_range(start, periods=days),
        "npl_ratio": np.abs(2.0 + rng.normal(0, 0.05, days)),   # timeseries hero (>= 0)
        "cet1": np.full(days, 10.0),                            # THRESHOLD: 10.0 <= critical 10.5 -> CRITICAL
        "icaap_days": np.full(days, 20.0),                      # DATEMATH: 20 days > 14 -> SAFE
    })}


def _mm_extract(streams, kpi):
    return streams["book"][kpi].to_numpy()


def _mm_hero_frame(streams):
    b = streams["book"]
    return pd.DataFrame({"ds": pd.to_datetime(b["date"]), "y": b["npl_ratio"].to_numpy(dtype=float)})


MULTIMODE = Domain(
    name="multimode", generate=_mm_generate, extract=_mm_extract, hero_frame=_mm_hero_frame,
    kpi_specs=(
        KpiSpec("npl_ratio", (0.0, None), DetectionMode.TIMESERIES, is_hero=True),
        KpiSpec("cet1", (0.0, None), DetectionMode.THRESHOLD,
                thresholds={"warn": 11.5, "critical": 10.5, "direction": "below"}),
        KpiSpec("icaap_days", (None, None), DetectionMode.DATEMATH,
                thresholds={"warn_days": 14, "critical_days": 3}),
    ),
    analyst_system="a", recommend_system="r",
)


def test_threshold_and_datemath_modes_end_to_end():
    """The two new modes assess the LATEST value: a breached capital floor is CRITICAL, a distant
    deadline is SAFE. Mutation: revert the mode dispatch -> these become the timeseries path."""
    det = detect_node({"streams": MULTIMODE.generate(date(2026, 1, 1), 200, 0)}, MULTIMODE)
    sev = {f.kpi: f.severity for f in det["kpi_findings"]}
    assert sev["cet1"] == "CRITICAL"                 # 10.0 <= critical 10.5 (direction below)
    assert sev["icaap_days"] == "SAFE"               # 20 days remaining > 14
    assert sev["npl_ratio"] in {"SAFE", "WARNING", "CRITICAL"}
    assert det["overall_severity"] == "CRITICAL"


def test_threshold_mode_needs_no_full_history():
    """min-history is mode-aware: a THRESHOLD KPI with only 5 points is still assessed (not
    DATA_QUALITY-for-short), while the TIMESERIES hero still requires a full baseline."""
    det = detect_node({"streams": MULTIMODE.generate(date(2026, 1, 1), 5, 0)}, MULTIMODE)
    sev = {f.kpi: f.severity for f in det["kpi_findings"]}
    assert sev["cet1"] == "CRITICAL"                 # threshold assessed on 5 rows
    assert sev["npl_ratio"] == "DATA_QUALITY"        # timeseries still needs 120


def test_multimode_tick_produces_verifiable_artifact(tmp_path):
    f = run_tick(BaselineMemory(tmp_path / "m.json"), 0, out_root=tmp_path / "r",
                 now="2026-01-01T00:00:00", offline=True, domain=MULTIMODE)
    sev = {k.kpi: k.severity for k in f.kpi_findings}
    assert sev["cet1"] == "CRITICAL" and sev["icaap_days"] == "SAFE"
    assert f.overall_severity == "CRITICAL"
    ok, reason = data.verify_run(tmp_path / "r" / "latest")
    assert ok is True, reason
