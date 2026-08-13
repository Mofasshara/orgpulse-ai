import numpy as np

from orgpulse.core.domain import DetectionMode
from orgpulse.detect.rules import datemath_detect, min_history_for, threshold_detect
from orgpulse.detect.types import Severity


def _arr(v):
    return np.array([v] * 5, dtype=float)


def test_threshold_below_direction_with_inclusive_boundaries():
    th = {"warn": 11.5, "critical": 10.5, "direction": "below"}   # e.g. a CET1 capital floor
    assert threshold_detect(_arr(12.0), th, "cet1").severity is Severity.SAFE
    assert threshold_detect(_arr(11.5), th, "cet1").severity is Severity.WARNING     # boundary inclusive
    assert threshold_detect(_arr(11.0), th, "cet1").severity is Severity.WARNING
    assert threshold_detect(_arr(10.5), th, "cet1").severity is Severity.CRITICAL    # boundary inclusive
    assert threshold_detect(_arr(9.9), th, "cet1").severity is Severity.CRITICAL


def test_threshold_above_direction():
    th = {"warn": 80.0, "critical": 95.0, "direction": "above"}   # e.g. a VaR-utilization ceiling
    assert threshold_detect(_arr(50.0), th, "var").severity is Severity.SAFE
    assert threshold_detect(_arr(85.0), th, "var").severity is Severity.WARNING
    assert threshold_detect(_arr(96.0), th, "var").severity is Severity.CRITICAL


def test_datemath_deadline_bands_and_overdue():
    assert datemath_detect(_arr(30), None, "icaap").severity is Severity.SAFE
    assert datemath_detect(_arr(14), None, "icaap").severity is Severity.WARNING     # <= 14
    assert datemath_detect(_arr(3), None, "icaap").severity is Severity.CRITICAL     # <= 3
    overdue = datemath_detect(_arr(-5), None, "icaap")
    assert overdue.severity is Severity.CRITICAL and "OVERDUE" in overdue.detail


import json
from dataclasses import asdict

import pytest


def _assert_dq(r):
    """A DATA_QUALITY result must match the convention shared by every other detector: severity
    DATA_QUALITY, score=None (NOT a non-finite float — that breaks allow_nan=False serialization),
    flagged=True, and JSON-serializable under allow_nan=False."""
    assert r.severity is Severity.DATA_QUALITY
    assert r.score is None
    assert r.flagged is True
    json.dumps(asdict(r), allow_nan=False)   # must not raise


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("direction", ["below", "above"])
def test_threshold_detect_fails_safe_on_non_finite_latest(bad, direction):
    """A non-finite latest value is unassessable (DATA_QUALITY), never SAFE and never a wrong
    severity — consistent with the timeseries detectors. Mutation: drop the isfinite guard and
    -inf silently reads CRITICAL while NaN/inf read SAFE. Also pins score=None/flagged=True so a
    non-finite score can never reach an allow_nan=False serializer."""
    th = {"warn": 95.0, "critical": 11.5, "direction": direction} if direction == "above" \
        else {"warn": 11.5, "critical": 10.5, "direction": direction}
    _assert_dq(threshold_detect(_arr(bad), th, "cet1"))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_datemath_detect_fails_safe_on_non_finite_latest(bad):
    """Same fail-safe contract and result-field convention for the deadline detector."""
    _assert_dq(datemath_detect(_arr(bad), None, "icaap"))


# Malformed SHAPE/DTYPE, not just non-finite values. The standalone detectors must fail safe on
# their own (like cusum/isolation), not only when the pipeline's series gate intercepts first.
_MALFORMED = [
    ([], "empty series"),
    (["bad"], "non-numeric series"),
    ([[12.0, 10.0]], "2-D array"),
    (10.0, "0-D scalar"),
    (None, "None"),
    ([10 ** 10000], "int too large to convert to float (OverflowError)"),
]


@pytest.mark.parametrize("bad,_desc", _MALFORMED)
def test_threshold_detect_fails_safe_on_malformed_input(bad, _desc):
    """Empty/ragged/2-D/non-numeric/0-D input must return DATA_QUALITY, never raise. Mutation:
    index-and-convert before validating (the old `float(np.asarray(values)[-1])`) -> IndexError/
    ValueError/TypeError."""
    th = {"warn": 11.5, "critical": 10.5, "direction": "below"}
    _assert_dq(threshold_detect(bad, th, "cet1"))


@pytest.mark.parametrize("bad,_desc", _MALFORMED)
def test_datemath_detect_fails_safe_on_malformed_input(bad, _desc):
    """Same standalone malformed-input fail-safe for the deadline detector."""
    _assert_dq(datemath_detect(bad, None, "icaap"))


def test_scalar_numeric_gates_fail_safe_on_overflow():
    """The scalar validators and the detectors that consume a remembered baseline must treat an
    int-too-large-for-float as unusable (False / DATA_QUALITY), never crash standalone. Mutation:
    drop OverflowError from is_finite_scalar -> these raise."""
    from orgpulse.detect.validation import (
        baseline_domain_feasible, finite_nonneg, finite_positive_order, is_finite_scalar,
    )
    from orgpulse.detect.zscore import zscore_detect
    from orgpulse.detect.cusum import cusum_detect
    from orgpulse.detect.baseline import Baseline

    huge = 10 ** 10000
    assert is_finite_scalar(huge) is False
    assert finite_nonneg(huge) is False
    assert finite_positive_order(1.0, huge) is False
    assert baseline_domain_feasible("x", huge, 1.0, {"x": (0.0, None)}) is False
    assert zscore_detect(1.0, Baseline(mean=huge, std=1.0, n=120), kpi="x").severity is Severity.DATA_QUALITY
    assert cusum_detect([1.0] * 120, Baseline(mean=huge, std=1.0, n=120), kpi="x").severity is Severity.DATA_QUALITY


def test_baseline_memory_overflow_value_is_clear_operator_error(tmp_path):
    """Valid JSON whose baseline mean is an int too large for float must produce the clear
    operator-facing 'corrupt or unreadable' ValueError, not a raw OverflowError. Mutation: drop
    OverflowError from the load() except."""
    from orgpulse.agent.baseline_memory import BaselineMemory

    p = tmp_path / "m.json"
    p.write_text('{"x": {"mean": 1' + "0" * 1001 + ', "std": 1, "n": 120}}')  # ~10**1001
    with pytest.raises(ValueError, match="corrupt or unreadable"):
        BaselineMemory(p).load()


def test_detect_kpi_fails_safe_on_overflow_standalone():
    """The timeseries orchestrator must fail to DATA_QUALITY on an int-too-large-for-float, standalone
    too (via the shared ensure_1d_finite gate converting OverflowError -> ValueError, which every
    caller already catches). Mutation: drop that conversion -> OverflowError propagates uncaught."""
    from orgpulse.detect.orchestrator import detect_kpi
    res = detect_kpi([10 ** 10000] * 120, [10 ** 10000] * 30, "x")
    assert all(r.severity is Severity.DATA_QUALITY for r in res)


def test_min_history_is_mode_aware():
    assert min_history_for(DetectionMode.TIMESERIES) == 120
    assert min_history_for(DetectionMode.THRESHOLD) == 1
    assert min_history_for(DetectionMode.DATEMATH) == 1
