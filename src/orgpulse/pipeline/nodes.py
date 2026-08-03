from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
import pandas as pd

from orgpulse.detect.orchestrator import detect_kpi, overall_severity
from orgpulse.detect.types import Severity
from orgpulse.forecast.prophet_forecaster import forecast_kpi, series_from_generator
from orgpulse.pipeline.state import KpiFinding, PipelineState


class DetectNodeOutput(TypedDict):
    kpi_findings: list[KpiFinding]
    overall_severity: str


class ForecastNodeOutput(TypedDict):
    forecast: dict


def detect_node(state: PipelineState) -> DetectNodeOutput:
    """
    Detect anomalies for each monitored KPI.

    Hero KPI: clinical daily_enrollment (aggregated across sites)
    QC KPIs: pass_rate, calibration_offset
    Supply KPI: lead_time_days

    For each KPI:
    - Build daily series (aggregate for clinical)
    - Split into baseline (>= 90) and recent (30)
    - Run detect_kpi for each
    - Record KpiFinding with severity, detail, latest_value
    - Skip if series < 120
    """
    findings: list[KpiFinding] = []
    all_severities: list[Severity] = []

    # Clinical hero: daily_enrollment
    if "clinical" in state:
        clinical_df = state["clinical"]
        # Aggregate via series_from_generator
        clinical_series_df = series_from_generator(clinical_df, "date", "daily_enrollment")
        y = clinical_series_df["y"].to_numpy()

        if len(y) >= 120:
            baseline = y[:-30]
            recent = y[-30:]

            results = detect_kpi(baseline, recent, "daily_enrollment")
            severity = overall_severity(results)

            # Extract detail from the most relevant result (preferring flagged)
            detail = results[0].detail if results else "no results"

            # Latest value: last recent value
            latest_val = float(recent[-1])
            if not math.isfinite(latest_val):
                latest_val = None

            findings.append(
                KpiFinding(
                    kpi="daily_enrollment",
                    severity=severity.name,
                    detail=detail,
                    latest_value=latest_val,
                )
            )
            all_severities.append(severity)

    # QC: pass_rate
    if "qc" in state:
        qc_df = state["qc"]
        y = qc_df["pass_rate"].to_numpy()

        if len(y) >= 120:
            baseline = y[:-30]
            recent = y[-30:]

            results = detect_kpi(baseline, recent, "pass_rate")
            severity = overall_severity(results)

            detail = results[0].detail if results else "no results"

            latest_val = float(recent[-1])
            if not math.isfinite(latest_val):
                latest_val = None

            findings.append(
                KpiFinding(
                    kpi="pass_rate",
                    severity=severity.name,
                    detail=detail,
                    latest_value=latest_val,
                )
            )
            all_severities.append(severity)

    # QC: calibration_offset
    if "qc" in state:
        qc_df = state["qc"]
        y = qc_df["calibration_offset"].to_numpy()

        if len(y) >= 120:
            baseline = y[:-30]
            recent = y[-30:]

            results = detect_kpi(baseline, recent, "calibration_offset")
            severity = overall_severity(results)

            detail = results[0].detail if results else "no results"

            latest_val = float(recent[-1])
            if not math.isfinite(latest_val):
                latest_val = None

            findings.append(
                KpiFinding(
                    kpi="calibration_offset",
                    severity=severity.name,
                    detail=detail,
                    latest_value=latest_val,
                )
            )
            all_severities.append(severity)

    # Supply: lead_time_days
    if "supply" in state:
        supply_df = state["supply"]
        y = supply_df["lead_time_days"].to_numpy()

        if len(y) >= 120:
            baseline = y[:-30]
            recent = y[-30:]

            results = detect_kpi(baseline, recent, "lead_time_days")
            severity = overall_severity(results)

            detail = results[0].detail if results else "no results"

            latest_val = float(recent[-1])
            if not math.isfinite(latest_val):
                latest_val = None

            findings.append(
                KpiFinding(
                    kpi="lead_time_days",
                    severity=severity.name,
                    detail=detail,
                    latest_value=latest_val,
                )
            )
            all_severities.append(severity)

    # Overall severity: max across all KPIs
    if all_severities:
        overall_sev = max(all_severities).name
    else:
        overall_sev = Severity.DATA_QUALITY.name

    return {
        "kpi_findings": findings,
        "overall_severity": overall_sev,
    }


def forecast_node(state: PipelineState) -> ForecastNodeOutput:
    """
    Forecast the hero KPI (clinical daily_enrollment) at horizons 30, 60, 90 days.

    Returns forecast with keys: yhat, lower (yhat_lower), upper (yhat_upper).
    """
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
