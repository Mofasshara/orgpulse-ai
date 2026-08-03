from __future__ import annotations

from orgpulse.detect.baseline import Baseline
from orgpulse.detect.types import AnomalyResult, Severity


def zscore_detect(
    value: float, baseline: Baseline, kpi: str, warn: float = 3.0, crit: float = 4.5
) -> AnomalyResult:
    z = (value - baseline.mean) / baseline.std
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
