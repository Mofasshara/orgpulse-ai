"""Proof that the engine is genuinely domain-agnostic: a minimal NON-pharma domain must run the
whole stack (tick, forecast, verify, offline framing, and the LangGraph pipeline) with no engine
changes. This is the guard against re-overclaiming "banking can be added without touching the
engine" — it exercises exactly what Codex's BANK probe exposed."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from orgpulse.agent.baseline_memory import BaselineMemory
from orgpulse.agent.tick import run_tick
from orgpulse.core.domain import Domain, KpiSpec
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
