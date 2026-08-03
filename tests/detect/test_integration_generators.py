from datetime import date

import numpy as np

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical, PROTOCOL_AMENDMENT_DAY
from orgpulse.generators.qc import generate_qc
from orgpulse.detect.baseline import fit_baseline
from orgpulse.detect.cusum import cusum_detect
from orgpulse.detect.orchestrator import detect_kpi, overall_severity
from orgpulse.detect.types import Severity


def test_detects_qc_calibration_drift_upward():
    df = generate_qc(date(2026, 1, 1), 300, make_rng(42))
    series = df["calibration_offset"].to_numpy()
    baseline = fit_baseline(series[:60])
    recent = series[-60:]
    r = cusum_detect(recent, baseline, kpi="calibration_offset")
    assert r.severity is Severity.CRITICAL
    assert "high" in r.detail                       # DIRECTION: upward drift
    assert recent.mean() > series[:60].mean()       # sanity: it really drifted up


def test_detects_clinical_amendment_surge_upward():
    df = generate_clinical(date(2026, 1, 1), 200, make_rng(42))
    per = df.groupby("date")["daily_enrollment"].sum().sort_index().to_numpy()
    history = per[PROTOCOL_AMENDMENT_DAY - 40 : PROTOCOL_AMENDMENT_DAY - 10]
    recent = per[PROTOCOL_AMENDMENT_DAY : PROTOCOL_AMENDMENT_DAY + 30]
    results = detect_kpi(history, recent, kpi="daily_enrollment")
    cusum = [r for r in results if r.method == "cusum"][0]
    assert overall_severity(results) >= Severity.WARNING
    assert recent.mean() > history.mean()           # DIRECTION: surge is UP
    assert "high" in cusum.detail                    # detector saw an upward shift


def test_stable_clinical_kpi_has_bounded_false_alarm_rate():
    # QC pass_rate is NOT calm (it declines with calibration drift), so it is a real
    # signal, not a false-alarm control. screen_fail_rate is genuinely iid/stationary,
    # so a benign stable KPI must rarely CRITICAL.
    crit = 0
    N = 200
    for s in range(N):
        df = generate_clinical(date(2026, 1, 1), 200, make_rng(s))
        series = df.groupby("date")["screen_fail_rate"].mean().to_numpy()
        sev = overall_severity(detect_kpi(series[:120], series[120:150], kpi="screen_fail_rate"))
        crit += sev is Severity.CRITICAL
    assert crit / N <= 0.05
