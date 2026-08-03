from __future__ import annotations

from orgpulse.detect.baseline import Baseline
from orgpulse.detect.types import AnomalyResult, Severity
from orgpulse.detect.validation import is_finite_scalar, baseline_is_usable, finite_positive_order


def zscore_detect(
    value: float, baseline: Baseline, kpi: str, warn: float = 3.0, crit: float = 4.5
) -> AnomalyResult:
    def _dq(detail):
        return AnomalyResult(kpi=kpi, method="zscore", severity=Severity.DATA_QUALITY, score=None, flagged=True, detail=detail)

    if not baseline_is_usable(baseline.mean, baseline.std, baseline.n):
        return _dq("unusable baseline")
    if not finite_positive_order(warn, crit):
        return _dq("invalid thresholds")
    if not is_finite_scalar(value):
        return _dq("non-finite value")

    z = (value - baseline.mean) / baseline.std
    if not is_finite_scalar(z):
        return _dq("non-finite standardized value")

    az = abs(z)
    if az >= crit:
        sev = Severity.CRITICAL
    elif az >= warn:
        sev = Severity.WARNING
    else:
        sev = Severity.SAFE
    return AnomalyResult(
        kpi=kpi,
        method="zscore",
        severity=sev,
        score=float(z),
        flagged=sev > Severity.SAFE,
        detail=f"z={z:.2f} (warn>={warn}, crit>={crit})",
    )
