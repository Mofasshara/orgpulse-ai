import numpy as np
import pytest
from orgpulse.detect.baseline import fit_baseline, Baseline
from orgpulse.detect.cusum import (
    cusum_detect, calibrated_thresholds, simulate_incontrol_quantiles, _CALIB_TABLE, _MIN_BASELINE_N,
)
from orgpulse.detect.types import Severity


def test_calibration_table_matches_montecarlo():
    for n in (30, 60, 120):
        warn, crit = _CALIB_TABLE[n]
        sim_warn, sim_crit = simulate_incontrol_quantiles(n)
        assert abs(warn - sim_warn) < 0.6 and abs(crit - sim_crit) < 0.6


def test_short_baseline_is_data_quality():
    b = Baseline(mean=50.0, std=2.0, n=_MIN_BASELINE_N - 1)
    assert cusum_detect(np.random.default_rng(0).normal(50, 2, 60), b, kpi="k").severity is Severity.DATA_QUALITY


@pytest.mark.parametrize("n_recent", [20, 30, 60, 120, 365])
def test_incontrol_false_alarm_bounded_across_windows(n_recent):
    crit = 0
    N = 800
    for s in range(N):
        rng = np.random.default_rng(50000 + s)
        base = fit_baseline(rng.normal(50, 2, 60))     # minimum baseline
        rec = rng.normal(50, 2, n_recent)
        crit += cusum_detect(rec, base, kpi="k").severity is Severity.CRITICAL
    assert crit / N <= 0.03                              # ~1% target + slack, at the worst-case baseline


def test_detection_power_for_real_shift():
    hits, N = 0, 800
    for s in range(N):
        rng = np.random.default_rng(60000 + s)
        base = fit_baseline(rng.normal(50, 2, 90))
        rec = rng.normal(50 + 2, 2, 60)
        hits += cusum_detect(rec, base, kpi="k").severity is Severity.CRITICAL
    assert hits / N >= 0.95


def test_detector_actually_uses_window_dependent_thresholds(monkeypatch):
    # Prove cusum_detect selects thresholds by len(values), not a hard-coded value.
    seen = {}
    import orgpulse.detect.cusum as cz
    real = cz.calibrated_thresholds
    def spy(n):
        seen["n"] = n
        return real(n)
    monkeypatch.setattr(cz, "calibrated_thresholds", spy)
    base = fit_baseline(np.random.default_rng(0).normal(50, 2, 90))
    cz.cusum_detect(np.random.default_rng(1).normal(50, 2, 123), base, kpi="k")
    assert seen["n"] == 123
