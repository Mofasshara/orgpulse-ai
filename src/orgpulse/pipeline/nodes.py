from __future__ import annotations

import math
from typing import TypedDict

import numpy as np

from orgpulse.detect.orchestrator import detect_kpi, overall_severity
from orgpulse.detect.types import Severity
from orgpulse.forecast.calibration import interval_calibration
from orgpulse.forecast.prophet_forecaster import forecast_kpi, series_from_generator
from orgpulse.pipeline.llm import LLMClient
from orgpulse.pipeline.state import (
    HUMAN_OVERSIGHT_DISCLAIMER,
    KpiFinding,
    PipelineState,
    Recommendation,
    data_quality_finding,
)


class DetectNodeOutput(TypedDict):
    kpi_findings: list[KpiFinding]
    overall_severity: str


class ForecastNodeOutput(TypedDict):
    forecast: dict


def _kpi_finding(kpi: str, series) -> KpiFinding:
    """
    Assess one KPI series and ALWAYS return a finding. This is the single fail-safe gate:
    the default is DATA_QUALITY, and SAFE/WARNING/CRITICAL are returned only after the data
    is affirmatively validated (1-D, numeric, >=120 points) AND detected. A short, missing,
    ragged, or non-numeric series can never be silently omitted or read as SAFE.

    - detail comes from the *driving* detector (the one whose severity == the max severity).
    - latest_value is a plain Python float (NaN/inf -> None) for JSON safety.
    """
    try:
        s = np.asarray(series, dtype=float)
    except (ValueError, TypeError):
        return data_quality_finding(kpi, "unusable input (non-numeric or ragged)")
    if s.ndim != 1:
        return data_quality_finding(kpi, f"unusable input ({s.ndim}-D, expected 1-D)")
    if len(s) < 120:  # need >=90 baseline + 30 recent
        return data_quality_finding(kpi, f"insufficient data ({len(s)} points < 120 required)")
    baseline, recent = s[:-30], s[-30:]
    results = detect_kpi(baseline, recent, kpi=kpi)  # returns DATA_QUALITY on non-finite data
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
    # Each extractor pulls one KPI series from the (possibly malformed) input. Extraction and
    # array conversion happen INSIDE the guard below so that a missing column, a 2-D/ragged
    # column, or any schema corruption becomes a DATA_QUALITY finding rather than crashing the
    # run — upholding the "bad data never crashes, never reads SAFE" contract at the node level.
    extractors: list[tuple[str, callable]] = []
    if "clinical" in state:
        extractors.append(
            ("daily_enrollment",
             lambda: series_from_generator(state["clinical"], "date", "daily_enrollment")["y"].to_numpy())
        )
    if "qc" in state:
        extractors.append(("pass_rate", lambda: state["qc"]["pass_rate"].to_numpy()))
        extractors.append(("calibration_offset", lambda: state["qc"]["calibration_offset"].to_numpy()))
    if "supply" in state:
        extractors.append(("lead_time_days", lambda: state["supply"]["lead_time_days"].to_numpy()))

    findings: list[KpiFinding] = []
    for kpi, extract in extractors:
        try:
            series = extract()
        except (KeyError, ValueError, TypeError, IndexError):
            findings.append(data_quality_finding(kpi, "unusable input (missing column / malformed data)"))
            continue
        findings.append(_kpi_finding(kpi, series))  # always returns a finding (never None)

    # DATA_QUALITY dominates (mirrors detect.orchestrator.overall_severity): if any KPI
    # could not be assessed — or nothing was analyzable at all — the run is DATA_QUALITY,
    # not a lower severity. Otherwise take the max of SAFE/WARNING/CRITICAL. (A bare
    # numeric max is wrong here because Severity.DATA_QUALITY = -1, the lowest value.)
    if not findings or any(f.severity == "DATA_QUALITY" for f in findings):
        overall = Severity.DATA_QUALITY
    else:
        overall = max(Severity[f.severity] for f in findings)

    return {"kpi_findings": findings, "overall_severity": overall.name}


def forecast_node(state: PipelineState) -> ForecastNodeOutput:
    """
    Forecast the hero KPI (clinical daily_enrollment) at horizons 30, 60, 90 days,
    with a calibration factor APPLIED to the published interval (not just diagnosed).

    Raw Prophet intervals are overconfident (~70% coverage against a nominal 90%). We
    fit a widening scale via rolling-origin folds (`interval_calibration`) and apply it
    to the production bounds so the published interval reflects that correction. The
    scale is fit IN-SAMPLE, so `calibration.in_sample` is recorded and the out-of-sample
    coverage is explicitly not claimed. If the history is too short to calibrate, the raw
    interval is published unchanged with `calibration.applied=False`.

    Returns forecast[kpi] with per-horizon {yhat, lower, upper} plus a `calibration` block.
    """
    forecast_result = {}

    if "clinical" in state:
        clinical_df = state["clinical"]
        # Build daily series. Guard corrupt/unusable input: a non-finite or too-short hero
        # series must not be fed to Prophet (it would crash or emit a meaningless forecast).
        # Detection already flags such a stream DATA_QUALITY; here we simply publish no forecast.
        try:
            daily_df = series_from_generator(clinical_df, "date", "daily_enrollment")
            y = daily_df["y"].to_numpy(dtype=float)
        except (KeyError, ValueError, TypeError):
            return {"forecast": forecast_result}
        if len(y) < 120 or not np.isfinite(y).all():
            return {"forecast": forecast_result}

        # Forecast at horizons 30, 60, 90
        fc = forecast_kpi(daily_df, "daily_enrollment", horizons=(30, 60, 90))

        # Fit the interval-calibration scale on this history and APPLY it to the
        # published bounds. Guard short histories: fall back to raw, uncalibrated.
        try:
            cal = interval_calibration(daily_df)
            scale, raw_cov, applied = cal.scale_factor, cal.raw_coverage, True
            cal_meta = {
                "applied": True, "in_sample": True, "scale_factor": float(scale),
                "raw_coverage": float(raw_cov), "nominal": float(cal.nominal),
                "note": "interval widened by an in-sample calibration factor; "
                        "out-of-sample coverage not validated",
            }
        except ValueError:
            scale = 1.0
            cal_meta = {
                "applied": False, "in_sample": True, "scale_factor": 1.0,
                "raw_coverage": None, "nominal": 0.9,
                "note": "history too short to calibrate; raw Prophet interval published",
            }

        forecast_result["daily_enrollment"] = {"calibration": cal_meta}
        for h in (30, 60, 90):
            fc_at_h = fc.at_horizon(h)
            yhat = float(fc_at_h["yhat"])
            lower = float(fc_at_h["yhat_lower"])
            upper = float(fc_at_h["yhat_upper"])
            forecast_result["daily_enrollment"][f"h{h}"] = {
                "yhat": yhat,
                "lower": yhat - scale * (yhat - lower),  # widen by the calibration scale
                "upper": yhat + scale * (upper - yhat),
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

    # Hero fallback when nothing is flagged: if the run is DATA_QUALITY (nothing was
    # analyzable), treat the hero as a data-quality case (not a process CAPA); otherwise
    # it is a genuinely calm run and the hero gets a routine CAPA-style review rec.
    if not anomaly_kpis and not dq_kpis:
        if overall_severity == "DATA_QUALITY":
            dq_kpis = ["daily_enrollment"]
        else:
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
