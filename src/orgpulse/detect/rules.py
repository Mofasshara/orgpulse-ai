"""Non-timeseries detection modes: hard THRESHOLD rules and DATEMATH deadlines.

Unlike the timeseries detectors (which need a baseline and history), these assess the LATEST
value of a KPI directly — a capital ratio breaching its regulatory floor is a breach regardless
of trend, and a filing deadline is arithmetic, not statistics. Both return the same
``AnomalyResult`` shape as the timeseries detectors so the pipeline is uniform.
"""
from __future__ import annotations

import math

import numpy as np

from orgpulse.core.domain import DetectionMode
from orgpulse.detect.types import AnomalyResult, Severity
from orgpulse.detect.validation import MIN_ASSESSMENT_DAYS


def min_history_for(mode: DetectionMode) -> int:
    """How many observations a KPI needs to be assessable, by mode. Timeseries needs a full
    baseline (120); a threshold/deadline check needs only the latest value."""
    return MIN_ASSESSMENT_DAYS if mode is DetectionMode.TIMESERIES else 1


def _latest_finite(values) -> tuple[float | None, str | None]:
    """Extract the LATEST value as a finite float, or return a reason it is unassessable.

    Standalone-safe (never raises): empty, ragged/2-D, non-numeric, 0-D-scalar, and non-finite
    inputs all yield a reason rather than an exception — so the rule detectors return DATA_QUALITY
    on malformed input just like cusum/isolation do on their OWN, not only when the pipeline's
    series gate happens to intercept first. Returns (value, None) on success, (None, reason) otherwise.
    """
    try:
        arr = np.asarray(values, dtype=float)
    except (ValueError, TypeError, OverflowError):
        # OverflowError: an int too large to convert to float (e.g. 10**10000) is corrupt numeric
        # data, not a valid latest value — fail safe like any other malformed input.
        return None, "value is non-numeric, ragged, or out-of-range; cannot assess"
    if arr.ndim != 1:
        return None, f"expected a 1-D series, got a {arr.ndim}-D array; cannot assess"
    if arr.size == 0:
        return None, "empty series; cannot assess"
    v = float(arr[-1])
    # Fail SAFE-is-not-the-default: a non-finite latest value is unassessable, not "not breached".
    if not math.isfinite(v):
        return None, f"non-finite latest value ({v}); cannot assess"
    return v, None


def _dq(kpi: str, method: str, reason: str) -> AnomalyResult:
    """A DATA_QUALITY result in the shared convention: score=None (never a non-finite float that
    breaks allow_nan=False serialization), flagged=True (surfaced, not silently passed)."""
    return AnomalyResult(kpi=kpi, method=method, severity=Severity.DATA_QUALITY, score=None,
                         flagged=True, detail=f"{kpi}: {reason}")


def threshold_detect(values, thresholds: dict, kpi: str) -> AnomalyResult:
    """Hard min/max rule on the LATEST value. ``thresholds`` = {"warn", "critical", "direction"}
    where direction is "below" (breach when value <= threshold, e.g. a capital floor) or "above"
    (breach when value >= threshold, e.g. a utilization ceiling)."""
    v, reason = _latest_finite(values)
    if reason is not None:
        return _dq(kpi, "threshold", reason)
    warn, crit = float(thresholds["warn"]), float(thresholds["critical"])
    direction = thresholds.get("direction", "below")
    if direction == "below":
        sev = Severity.CRITICAL if v <= crit else (Severity.WARNING if v <= warn else Severity.SAFE)
        rel = "<="
    else:
        sev = Severity.CRITICAL if v >= crit else (Severity.WARNING if v >= warn else Severity.SAFE)
        rel = ">="
    detail = f"{kpi}={v:.4g} (warn {rel} {warn:g}, critical {rel} {crit:g})"
    return AnomalyResult(kpi=kpi, method="threshold", severity=sev, score=v,
                         flagged=sev is not Severity.SAFE, detail=detail)


def datemath_detect(values, thresholds: dict | None, kpi: str) -> AnomalyResult:
    """Deadline arithmetic on the LATEST value, interpreted as DAYS REMAINING (negative = overdue).
    CRITICAL if overdue or within `critical_days` (default 3); WARNING within `warn_days`
    (default 14); else SAFE."""
    days, reason = _latest_finite(values)
    if reason is not None:
        return _dq(kpi, "datemath", reason)
    t = thresholds or {}
    warn_days, crit_days = float(t.get("warn_days", 14)), float(t.get("critical_days", 3))
    if days < 0:
        sev, note = Severity.CRITICAL, f"OVERDUE by {abs(days):.0f} day(s)"
    elif days <= crit_days:
        sev, note = Severity.CRITICAL, f"due in {days:.0f} day(s) (<= {crit_days:.0f})"
    elif days <= warn_days:
        sev, note = Severity.WARNING, f"due in {days:.0f} day(s) (<= {warn_days:.0f})"
    else:
        sev, note = Severity.SAFE, f"due in {days:.0f} day(s)"
    return AnomalyResult(kpi=kpi, method="datemath", severity=sev, score=days,
                         flagged=sev is not Severity.SAFE, detail=f"{kpi}: {note}")
