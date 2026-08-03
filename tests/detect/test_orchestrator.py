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
    assert overall_severity([]) is Severity.SAFE


def test_shift_raises_overall_severity():
    rng = np.random.default_rng(1)
    history = rng.normal(50, 2, 120)
    recent = rng.normal(56, 2, 30)          # +3σ shift
    sev = overall_severity(detect_kpi(history, recent, kpi="k"))
    assert sev >= Severity.WARNING
