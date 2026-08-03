from __future__ import annotations

import math
import random
from typing import TypedDict

import numpy as np

from orgpulse.detect.orchestrator import detect_kpi, overall_severity
from orgpulse.detect.types import Severity
from orgpulse.forecast.prophet_forecaster import forecast_kpi, series_from_generator
from orgpulse.pipeline.llm import LLMClient
from orgpulse.pipeline.state import HUMAN_OVERSIGHT_DISCLAIMER, KpiFinding, PipelineState, Recommendation


class DetectNodeOutput(TypedDict):
    kpi_findings: list[KpiFinding]
    overall_severity: str


class ForecastNodeOutput(TypedDict):
    forecast: dict


def _kpi_finding(kpi: str, series) -> KpiFinding | None:
    """
    Build a KpiFinding for one KPI series.

    - Requires >= 120 points (>=90 baseline + 30 recent); returns None if shorter.
    - detail comes from the *driving* detector (the one whose severity == the
      overall/max severity), not always the zscore result.
    - latest_value is a plain Python float (NaN/inf -> None) for JSON safety.
    """
    s = np.asarray(series, dtype=float)
    if len(s) < 120:  # need >=90 baseline + 30 recent
        return None
    baseline, recent = s[:-30], s[-30:]
    results = detect_kpi(baseline, recent, kpi=kpi)
    sev = overall_severity(results)
    driving = next((r for r in results if r.severity == sev), results[0])  # detail from the driving detector
    v = float(recent[-1])
    latest = v if math.isfinite(v) else None
    return KpiFinding(kpi=kpi, severity=sev.name, detail=driving.detail, latest_value=latest)


def detect_node(state: PipelineState) -> DetectNodeOutput:
    """
    Detect anomalies for each monitored KPI.

    Hero KPI: clinical daily_enrollment (aggregated across sites)
    QC KPIs: pass_rate, calibration_offset
    Supply KPI: lead_time_days

    Each KPI series is split into a >=90-point baseline and a 30-point recent
    window and run through detect_kpi. Series shorter than 120 are skipped.
    """
    series_by_kpi: list[tuple[str, np.ndarray]] = []

    if "clinical" in state:
        clinical_series = series_from_generator(state["clinical"], "date", "daily_enrollment")["y"].to_numpy()
        series_by_kpi.append(("daily_enrollment", clinical_series))

    if "qc" in state:
        qc_df = state["qc"]
        series_by_kpi.append(("pass_rate", qc_df["pass_rate"].to_numpy()))
        series_by_kpi.append(("calibration_offset", qc_df["calibration_offset"].to_numpy()))

    if "supply" in state:
        series_by_kpi.append(("lead_time_days", state["supply"]["lead_time_days"].to_numpy()))

    findings: list[KpiFinding] = []
    max_severity = Severity.DATA_QUALITY
    have_finding = False
    for kpi, series in series_by_kpi:
        finding = _kpi_finding(kpi, series)
        if finding is None:
            continue
        findings.append(finding)
        sev = Severity[finding.severity]
        if not have_finding or sev > max_severity:
            max_severity = sev
            have_finding = True

    return {
        "kpi_findings": findings,
        "overall_severity": max_severity.name,
    }


def forecast_node(state: PipelineState) -> ForecastNodeOutput:
    """
    Forecast the hero KPI (clinical daily_enrollment) at horizons 30, 60, 90 days.

    Returns forecast with keys: yhat, lower (yhat_lower), upper (yhat_upper).
    Sets seed=42 for determinism in Prophet's uncertainty sampling.
    """
    # Set seed for determinism (Prophet uses Monte Carlo for uncertainty intervals)
    np.random.seed(42)
    random.seed(42)

    forecast_result = {}

    if "clinical" in state:
        clinical_df = state["clinical"]
        # Build daily series
        daily_df = series_from_generator(clinical_df, "date", "daily_enrollment")

        # Forecast at horizons 30, 60, 90
        fc = forecast_kpi(daily_df, "daily_enrollment", horizons=(30, 60, 90))

        # Extract forecasts at each horizon
        forecast_result["daily_enrollment"] = {}
        for h in (30, 60, 90):
            fc_at_h = fc.at_horizon(h)
            forecast_result["daily_enrollment"][f"h{h}"] = {
                "yhat": float(fc_at_h["yhat"]),
                "lower": float(fc_at_h["yhat_lower"]),
                "upper": float(fc_at_h["yhat_upper"]),
            }

    return {"forecast": forecast_result}


def analyst_node(state: PipelineState, llm: LLMClient) -> dict:
    """
    Analyze findings and forecast via LLM.

    Builds a prompt from kpi_findings + forecast, calls llm.analyze,
    returns {"analysis": <str>}.
    """
    findings = state.get("kpi_findings", [])
    forecast = state.get("forecast", {})

    # Build a readable prompt from findings and forecast
    prompt_lines = ["KPI Findings:"]
    for f in findings:
        prompt_lines.append(f"- {f.kpi} ({f.severity}): {f.detail} [latest={f.latest_value}]")

    prompt_lines.append("\nForecast (hero KPI daily_enrollment):")
    if "daily_enrollment" in forecast:
        for horizon, data in forecast["daily_enrollment"].items():
            yhat = data.get("yhat", "N/A")
            lower = data.get("lower", "N/A")
            upper = data.get("upper", "N/A")
            prompt_lines.append(f"- {horizon}: yhat={yhat}, [{lower}, {upper}]")
    else:
        prompt_lines.append("- No forecast available")

    prompt = "\n".join(prompt_lines)

    # Call LLM analyze
    analysis = llm.analyze("You are a clinical data quality analyst.", prompt)

    return {"analysis": analysis}


def recommendation_node(state: PipelineState, llm: LLMClient) -> dict:
    """
    Generate recommendations for flagged KPIs.

    Splits flagged findings by severity:
    - anomaly_kpis (WARNING/CRITICAL) get an LLM-authored CAPA recommendation.
    - dq_kpis (DATA_QUALITY) get a deterministic data-quality recommendation —
      a data problem, not a process deviation, so no CAPA on the process.
    Hero fallback: if both groups are empty, anomaly_kpis = ["daily_enrollment"].

    Every recommendation carries HUMAN_OVERSIGHT_DISCLAIMER and defaults to
    review_status="PENDING_HUMAN_REVIEW". Alert fires on a critical anomaly OR a
    systemic data failure: overall_severity in ("CRITICAL", "DATA_QUALITY").
    Returns {"recommendations": [...], "alert": <bool>}.
    """
    findings = state.get("kpi_findings", [])
    overall_severity = state.get("overall_severity", "SAFE")

    # Split flagged findings by severity
    anomaly_kpis = [f.kpi for f in findings if f.severity in ("WARNING", "CRITICAL")]
    dq_kpis = [f.kpi for f in findings if f.severity == "DATA_QUALITY"]

    # Hero fallback: only when nothing at all is flagged
    if not anomaly_kpis and not dq_kpis:
        anomaly_kpis = ["daily_enrollment"]

    recommendations = []

    # Anomaly KPIs -> LLM-authored CAPA recommendations
    if anomaly_kpis:
        prompt_lines = ["Generate CAPA recommendations for the following flagged KPIs:"]
        for kpi in anomaly_kpis:
            prompt_lines.append(f"- {kpi}")
        prompt = "\n".join(prompt_lines)

        rec_dicts = llm.recommend("You are a quality assurance CAPA expert.", prompt, kpis=anomaly_kpis)
        for rec_dict in rec_dicts:
            recommendations.append(
                Recommendation(
                    kpi=rec_dict["kpi"],
                    text=rec_dict["text"],
                    capa_frame=rec_dict["capa_frame"],
                    disclaimer=HUMAN_OVERSIGHT_DISCLAIMER,
                    review_status="PENDING_HUMAN_REVIEW",
                )
            )

    # Data-quality KPIs -> deterministic data-quality recommendation (no LLM)
    for kpi in dq_kpis:
        recommendations.append(
            Recommendation(
                kpi=kpi,
                capa_frame="DATA-QUALITY",
                text=(
                    f"Data-quality issue on '{kpi}': the detector could not assess this KPI "
                    "(missing, corrupt, or insufficient data). Investigate the data source and "
                    "instrumentation before drawing any conclusion; do NOT treat this as a process "
                    "deviation or open a CAPA on the process."
                ),
                disclaimer=HUMAN_OVERSIGHT_DISCLAIMER,
            )
        )

    # Alert fires on a critical anomaly OR a systemic data failure
    alert = overall_severity in ("CRITICAL", "DATA_QUALITY")

    return {"recommendations": recommendations, "alert": alert}
