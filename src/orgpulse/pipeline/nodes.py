from __future__ import annotations

import math
from typing import TypedDict

import numpy as np

from orgpulse.detect.orchestrator import detect_kpi, overall_severity
from orgpulse.detect.types import Severity
from orgpulse.forecast.prophet_forecaster import forecast_kpi, series_from_generator
from orgpulse.pipeline.state import KpiFinding, PipelineState


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
