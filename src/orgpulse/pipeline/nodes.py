from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
import pandas as pd

from orgpulse.detect.orchestrator import detect_kpi, overall_severity
from orgpulse.detect.types import Severity
from orgpulse.detect.validation import kpi_domain_reason, kpi_series_reason
from orgpulse.forecast.calibration import interval_calibration
from orgpulse.forecast.prophet_forecaster import forecast_kpi, series_from_generator


def validated_clinical_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Validate RAW clinical rows before aggregating to a daily series — the single builder
    used by BOTH detection and forecasting. Aggregation hides per-row corruption (a negative
    site total sums into a plausible positive), so per-site validity is checked here, first.

    Raises ValueError if: a date is missing/NaT; (date, site_id) is not unique; or any present
    per-site daily_enrollment is negative or non-integer (enrollment is a nonnegative count).
    NaN per-site values are allowed — they propagate to a NaN daily total (→ DATA_QUALITY).
    Returns the daily (ds, y) frame with any-NaN-in-a-day → NaN.
    """
    if "date" not in df.columns or "daily_enrollment" not in df.columns:
        raise ValueError("clinical frame missing required columns")
    if df["date"].isna().any():
        raise ValueError("clinical frame has missing/NaT dates")
    if "site_id" in df.columns and df.duplicated(["date", "site_id"]).any():
        raise ValueError("duplicate (date, site_id) clinical records")
    vals = pd.to_numeric(df["daily_enrollment"], errors="coerce")
    present = vals[df["daily_enrollment"].notna()]
    if ((present < 0) | (present != present.round())).any():
        raise ValueError("impossible per-site enrollment (negative or non-integer count)")
    return series_from_generator(df, "date", "daily_enrollment")
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
    - a series shorter than 120 points, non-1-D, non-numeric, non-finite, or with any value
      outside the KPI's real-world domain becomes DATA_QUALITY (never silently skipped).
    """
    reason = kpi_series_reason(kpi, series, min_len=120)  # 1-D, numeric, >=120, finite, in-domain
    if reason is not None:
        return data_quality_finding(kpi, reason)
    s = np.asarray(series, dtype=float)
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
    window and run through detect_kpi. Every expected KPI ALWAYS yields a finding: an
    unusable series (short, non-numeric, non-finite, or out of domain) becomes DATA_QUALITY
    rather than being skipped.
    """
    # Each extractor pulls one KPI series from the (possibly malformed) input. Extraction and
    # array conversion happen INSIDE the guard below so that a missing column, a 2-D/ragged
    # column, or any schema corruption becomes a DATA_QUALITY finding rather than crashing the
    # run — upholding the "bad data never crashes, never reads SAFE" contract at the node level.
    extractors: list[tuple[str, callable]] = []
    if "clinical" in state:
        extractors.append(
            ("daily_enrollment", lambda: validated_clinical_daily(state["clinical"])["y"].to_numpy())
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
        # Build the daily series through the SAME validated builder + series check that
        # detection uses — so a hero KPI that detection rules DATA_QUALITY (out-of-domain,
        # duplicate rows, NaT dates, non-integer/negative per-site counts, non-finite, too
        # short) never yields a numeric forecast. One validator, both consumers.
        try:
            daily_df = validated_clinical_daily(state["clinical"])
            y = daily_df["y"].to_numpy(dtype=float)
        except (KeyError, ValueError, TypeError):
            return {"forecast": forecast_result}
        if kpi_series_reason("daily_enrollment", y, min_len=120) is not None:
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
