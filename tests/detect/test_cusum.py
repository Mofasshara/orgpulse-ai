import numpy as np

from orgpulse.detect.baseline import Baseline, fit_baseline
from orgpulse.detect.types import Severity
from orgpulse.detect.cusum import cusum_detect


def test_stable_series_is_safe():
    rng = np.random.default_rng(0)
    base_window = rng.normal(50, 2, 100)
    baseline = fit_baseline(base_window)
    stable = rng.normal(50, 2, 60)          # same process
    r = cusum_detect(stable, baseline, kpi="pass_rate")
    assert r.severity is Severity.SAFE and r.flagged is False


def test_upward_shift_is_detected():
    rng = np.random.default_rng(1)
    baseline = fit_baseline(rng.normal(50, 2, 100))
    shifted = rng.normal(54, 2, 60)         # +2σ sustained shift
    r = cusum_detect(shifted, baseline, kpi="calibration")
    assert r.severity is Severity.CRITICAL and r.flagged is True
    assert "high" in r.detail


def test_downward_shift_is_detected():
    rng = np.random.default_rng(2)
    baseline = fit_baseline(rng.normal(50, 2, 100))
    dropped = rng.normal(46, 2, 60)         # -2σ sustained shift
    r = cusum_detect(dropped, baseline, kpi="reliability")
    assert r.severity is Severity.CRITICAL
    assert "low" in r.detail
