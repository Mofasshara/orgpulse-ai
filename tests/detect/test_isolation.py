import numpy as np

from orgpulse.detect.types import Severity
from orgpulse.detect.isolation import isolation_detect


def test_normal_point_is_safe():
    rng = np.random.default_rng(0)
    history = rng.normal([100, 5], [8, 0.5], size=(200, 2))
    current = np.array([101.0, 5.1])          # squarely inside the cloud
    r = isolation_detect(history, current, kpi="supply")
    assert r.severity is Severity.SAFE and r.flagged is False


def test_outlier_point_is_flagged():
    rng = np.random.default_rng(0)
    history = rng.normal([100, 5], [8, 0.5], size=(200, 2))
    current = np.array([300.0, 25.0])         # far outside on both features
    r = isolation_detect(history, current, kpi="supply")
    assert r.flagged is True and r.severity in (Severity.WARNING, Severity.CRITICAL)


def test_deterministic_given_random_state():
    rng = np.random.default_rng(0)
    history = rng.normal([100, 5], [8, 0.5], size=(200, 2))
    current = np.array([300.0, 25.0])
    a = isolation_detect(history, current, kpi="supply")
    b = isolation_detect(history, current, kpi="supply")
    assert a.score == b.score and a.severity == b.severity
