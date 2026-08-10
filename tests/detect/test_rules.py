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


def test_min_history_is_mode_aware():
    assert min_history_for(DetectionMode.TIMESERIES) == 120
    assert min_history_for(DetectionMode.THRESHOLD) == 1
    assert min_history_for(DetectionMode.DATEMATH) == 1
