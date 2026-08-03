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


def test_isolation_nonfinite_is_data_quality_not_safe():
    hist = np.random.default_rng(0).normal([100, 5], [8, 0.5], size=(200, 2))
    assert isolation_detect(hist, np.array([np.nan, np.nan]), kpi="k").severity is Severity.DATA_QUALITY
    assert isolation_detect(hist, np.array([np.inf, 5.0]), kpi="k").severity is Severity.DATA_QUALITY
    bad_hist = hist.copy(); bad_hist[0, 0] = np.nan
    assert isolation_detect(bad_hist, np.array([100.0, 5.0]), kpi="k").severity is Severity.DATA_QUALITY


def test_isolation_empty_or_shape_mismatch_is_data_quality():
    hist = np.random.default_rng(0).normal([100, 5], [8, 0.5], size=(200, 2))
    assert isolation_detect(np.empty((0, 2)), np.array([1.0, 2.0]), kpi="k").severity is Severity.DATA_QUALITY
    assert isolation_detect(hist, np.array([1.0]), kpi="k").severity is Severity.DATA_QUALITY  # wrong dims
