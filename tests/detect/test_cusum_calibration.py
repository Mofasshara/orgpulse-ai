import numpy as np

from orgpulse.detect.baseline import fit_baseline
from orgpulse.detect.cusum import (
    cusum_detect, calibrated_thresholds, simulate_incontrol_quantiles, _CALIB_TABLE,
)
from orgpulse.detect.types import Severity


def test_calibration_table_matches_montecarlo():
    for n in (30, 60, 120):
        warn, crit = _CALIB_TABLE[n]
        sim_warn, sim_crit = simulate_incontrol_quantiles(n)
        assert abs(warn - sim_warn) < 0.5
        assert abs(crit - sim_crit) < 0.5


def test_thresholds_increase_with_window_length():
    assert calibrated_thresholds(30)[1] < calibrated_thresholds(365)[1]


def test_incontrol_false_alarm_rate_within_target():
    # Over many in-control seeds, CRITICAL rate must stay near the 1% target.
    n, crit_hits, warn_hits, N = 60, 0, 0, 1000
    for s in range(N):
        rng = np.random.default_rng(50000 + s)
        base = fit_baseline(rng.normal(50, 2, 90))
        rec = rng.normal(50, 2, n)
        sev = cusum_detect(rec, base, kpi="k").severity
        crit_hits += sev is Severity.CRITICAL
        warn_hits += sev is Severity.WARNING
    assert crit_hits / N <= 0.03          # ~1% target + Monte-Carlo slack
    assert warn_hits / N <= 0.16          # ~10% target + slack


def test_detection_power_for_real_shift():
    # A sustained +1σ shift must be caught (CRITICAL) almost always.
    n, hits, N = 60, 0, 1000
    for s in range(N):
        rng = np.random.default_rng(60000 + s)
        base = fit_baseline(rng.normal(50, 2, 90))
        rec = rng.normal(50 + 2, 2, n)     # +1σ (σ=2)
        hits += cusum_detect(rec, base, kpi="k").severity is Severity.CRITICAL
    assert hits / N >= 0.95
