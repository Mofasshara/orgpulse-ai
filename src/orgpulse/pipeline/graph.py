"""LangGraph 4-node DAG: detect → forecast → analyst → recommend → Findings & report."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from langgraph.graph import StateGraph, START, END

from orgpulse.pipeline.llm import LLMClient, make_llm
from orgpulse.pipeline.nodes import detect_node, forecast_node, analyst_node, recommendation_node
from orgpulse.pipeline.state import Findings, PipelineState, HUMAN_OVERSIGHT_DISCLAIMER


def build_graph(llm: LLMClient):
    """
    Construct a StateGraph(PipelineState) with nodes:
    detect → forecast → analyst → recommend

    Analyst and recommend nodes are wrapped in closures to pass llm.
    Returns the compiled graph.
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("detect", detect_node)
    graph.add_node("forecast", forecast_node)

    # Wrap analyst and recommend to pass llm
    def analyst_node_with_llm(state: PipelineState) -> dict:
        return analyst_node(state, llm)

    def recommendation_node_with_llm(state: PipelineState) -> dict:
        return recommendation_node(state, llm)

    graph.add_node("analyst", analyst_node_with_llm)
    graph.add_node("recommend", recommendation_node_with_llm)

    # Add edges
    graph.add_edge(START, "detect")
    graph.add_edge("detect", "forecast")
    graph.add_edge("forecast", "analyst")
    graph.add_edge("analyst", "recommend")
    graph.add_edge("recommend", END)

    return graph.compile()


def run_pipeline(
    clinical: pd.DataFrame,
    qc: pd.DataFrame,
    supply: pd.DataFrame,
    *,
    offline: bool = True,
    run_id: str | None = None,
    now: str | None = None,
    model: str = "claude-sonnet-5",
) -> Findings:
    """
    Run the end-to-end offline-deterministic pipeline.

    - Build llm using make_llm(offline, model)
    - Seed initial state with the three DataFrames
    - Invoke the compiled graph
    - Assemble Findings from the final state
    - generated_at = now if given else datetime.now().isoformat(timespec="seconds")
    - run_id = arg or a timestamp-derived id
    """
    # Prepare LLM
    llm = make_llm(offline, model)

    # Build and compile graph
    compiled_graph = build_graph(llm)

    # Seed initial state
    initial_state: PipelineState = {
        "clinical": clinical,
        "qc": qc,
        "supply": supply,
        "offline": offline,
    }

    # Invoke graph
    final_state = compiled_graph.invoke(initial_state)

    # Resolve generated_at first, then derive run_id from it so a caller
    # pinning `now` gets a deterministic run_id too.
    generated_at = now if now is not None else datetime.now().isoformat(timespec="seconds")
    run_id = run_id or f"run-{generated_at}"

    # Assemble Findings from final state
    findings = Findings(
        run_id=run_id,
        generated_at=generated_at,
        overall_severity=final_state.get("overall_severity", "SAFE"),
        alert=final_state.get("alert", False),
        kpi_findings=final_state.get("kpi_findings", []),
        forecast=final_state.get("forecast", {}),
        analysis=final_state.get("analysis", ""),
        recommendations=final_state.get("recommendations", []),
    )

    return findings


def render_report(findings: Findings) -> str:
    """
    Render a Findings object as a human-readable Markdown report.

    Contains:
    - Title with "OrgPulse"
    - Run ID and generated timestamp
    - Overall severity + alert banner
    - KPI findings table
    - Analysis paragraph
    - Forecast (30/60/90 for hero KPI daily_enrollment)
    - Recommendations (each with CAPA frame, status, and disclaimer)
    """
    lines = []

    # Title
    lines.append("# OrgPulse Data Quality Report")
    lines.append("")

    # Run metadata
    lines.append(f"**Run ID:** {findings.run_id}")
    lines.append(f"**Generated:** {findings.generated_at}")
    lines.append("")

    # Alert banner
    severity_emoji = {
        "SAFE": "✅",
        "WARNING": "⚠️",
        "CRITICAL": "🚨",
        "DATA_QUALITY": "⚠️",
    }
    emoji = severity_emoji.get(findings.overall_severity, "ℹ️")
    lines.append(f"{emoji} **Overall Severity:** {findings.overall_severity}")
    if findings.alert:
        lines.append("**🔔 ALERT TRIGGERED** — Human review required.")
    lines.append("")

    # KPI findings table
    if findings.kpi_findings:
        lines.append("## KPI Findings")
        lines.append("")
        lines.append("| KPI | Severity | Detail | Latest Value |")
        lines.append("|-----|----------|--------|--------------|")
        for kpi_f in findings.kpi_findings:
            latest_val = kpi_f.latest_value if kpi_f.latest_value is not None else "N/A"
            lines.append(f"| {kpi_f.kpi} | {kpi_f.severity} | {kpi_f.detail} | {latest_val} |")
        lines.append("")

    # Analysis
    if findings.analysis:
        lines.append("## Analysis")
        lines.append("")
        lines.append(findings.analysis)
        lines.append("")

    # Forecast
    if findings.forecast and "daily_enrollment" in findings.forecast:
        lines.append("## Forecast (Daily Enrollment)")
        lines.append("")
        hero_forecast = findings.forecast["daily_enrollment"]
        for horizon in ["h30", "h60", "h90"]:
            if horizon in hero_forecast:
                data = hero_forecast[horizon]
                yhat = data.get("yhat", "N/A")
                lower = data.get("lower", "N/A")
                upper = data.get("upper", "N/A")
                days = horizon[1:]
                lines.append(f"- **{days} days:** {yhat} (90% interval: [{lower}, {upper}])")
        lines.append("")

    # Recommendations
    if findings.recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for i, rec in enumerate(findings.recommendations, start=1):
            lines.append(f"### Recommendation {i}: {rec.kpi}")
            lines.append("")
            lines.append(f"**CAPA Frame:** {rec.capa_frame}")
            lines.append(f"**Status:** {rec.review_status}")
            lines.append("")
            lines.append(rec.text)
            lines.append("")
            lines.append(f"> {rec.disclaimer}")
            lines.append("")

    # Disclaimer summary at end
    lines.append("---")
    lines.append("")
    disclaimer_opening = HUMAN_OVERSIGHT_DISCLAIMER.split(" — ")[0]
    lines.append(f"**{disclaimer_opening}**")
    lines.append("")
    lines.append(f"> {HUMAN_OVERSIGHT_DISCLAIMER}")
    lines.append("")

    return "\n".join(lines)
