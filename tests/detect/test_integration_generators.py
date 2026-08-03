from datetime import date

import numpy as np

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical, PROTOCOL_AMENDMENT_DAY
from orgpulse.generators.qc import generate_qc
from orgpulse.detect.baseline import fit_baseline
from orgpulse.detect.cusum import cusum_detect
from orgpulse.detect.orchestrator import detect_kpi, overall_severity
from orgpulse.detect.types import Severity


def test_detects_qc_calibration_drift():
    # QC calibration_offset drifts upward over time -> CUSUM should fire.
    df = generate_qc(date(2026, 1, 1), 300, make_rng(42))
    series = df["calibration_offset"].to_numpy()
    baseline = fit_baseline(series[:60])          # early, near-zero window
    recent = series[-60:]                          # late, drifted window
    r = cusum_detect(recent, baseline, kpi="calibration_offset")
    assert r.severity is Severity.CRITICAL and "high" in r.detail


def test_detects_clinical_amendment_surge():
    # Post-amendment daily enrollment surges vs the pre-amendment plateau.
    df = generate_clinical(date(2026, 1, 1), 200, make_rng(42))
    per_day = df.groupby("date")["daily_enrollment"].sum().sort_index().to_numpy()
    history = per_day[PROTOCOL_AMENDMENT_DAY - 40 : PROTOCOL_AMENDMENT_DAY - 10]
    recent = per_day[PROTOCOL_AMENDMENT_DAY : PROTOCOL_AMENDMENT_DAY + 30]
    sev = overall_severity(detect_kpi(history, recent, kpi="daily_enrollment"))
    assert sev >= Severity.WARNING


def test_calm_baseline_stays_safe():
    # A stable stretch of the QC pass_rate should not raise a critical alarm.
    df = generate_qc(date(2026, 1, 1), 200, make_rng(7))
    series = df["pass_rate"].to_numpy()
    history = series[:120]
    recent = series[120:150]
    sev = overall_severity(detect_kpi(history, recent, kpi="pass_rate"))
    assert sev <= Severity.WARNING     # no false CRITICAL on a calm stretch
