from orgpulse.detect.baseline import Baseline
from orgpulse.detect.types import Severity
from orgpulse.detect.zscore import zscore_detect


BASE = Baseline(mean=100.0, std=10.0, n=50)


def test_normal_value_is_safe():
    r = zscore_detect(105.0, BASE, kpi="enroll")
    assert r.severity is Severity.SAFE and r.flagged is False
    assert r.score == 0.5


def test_warning_at_three_sigma():
    r = zscore_detect(131.0, BASE, kpi="enroll")   # z = 3.1
    assert r.severity is Severity.WARNING and r.flagged is True


def test_critical_beyond_4_5_sigma():
    r = zscore_detect(146.0, BASE, kpi="enroll")   # z = 4.6
    assert r.severity is Severity.CRITICAL


def test_negative_direction_uses_absolute_z():
    r = zscore_detect(54.0, BASE, kpi="enroll")    # z = -4.6
    assert r.severity is Severity.CRITICAL and r.score < 0
