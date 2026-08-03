import numpy as np

from orgpulse.detect.types import Severity
from orgpulse.detect.orchestrator import detect_kpi, overall_severity


def test_detect_kpi_returns_zscore_and_cusum():
    rng = np.random.default_rng(0)
    history = rng.normal(50, 2, 120)
    recent = rng.normal(50, 2, 30)
    results = detect_kpi(history, recent, kpi="pass_rate")
    methods = {r.method for r in results}
    assert methods == {"zscore", "cusum"}


def test_overall_severity_is_max():
    rng = np.random.default_rng(0)
    history = rng.normal(50, 2, 120)
    recent = rng.normal(50, 2, 30)
    calm = overall_severity(detect_kpi(history, recent, kpi="k"))
    assert calm is Severity.SAFE


def test_shift_raises_overall_severity():
    rng = np.random.default_rng(1)
    history = rng.normal(50, 2, 120)
    recent = rng.normal(56, 2, 30)          # +3σ shift
    sev = overall_severity(detect_kpi(history, recent, kpi="k"))
    assert sev >= Severity.WARNING


def test_detect_kpi_bad_history_is_data_quality_not_crash():
    for hist in (np.array([]), np.array([1.0, np.nan, 2.0]), np.ones((5, 2))):
        res = detect_kpi(hist, np.arange(30.0), kpi="k")
        assert all(r.severity is Severity.DATA_QUALITY for r in res)


def test_detect_kpi_bad_recent_is_data_quality_not_crash():
    for rec in (np.array([]), np.ones((5, 2)), np.array([1.0, np.inf])):
        res = detect_kpi(np.random.default_rng(0).normal(50, 2, 90), rec, kpi="k")
        assert all(r.severity is Severity.DATA_QUALITY for r in res)


def test_overall_severity_empty_is_data_quality():
    assert overall_severity([]) is Severity.DATA_QUALITY
