from __future__ import annotations

import numpy as np

from orgpulse.detect.baseline import Baseline
from orgpulse.detect.types import AnomalyResult, Severity
from orgpulse.detect.validation import is_finite_scalar


def zscore_detect(
    value: float, baseline: Baseline, kpi: str, warn: float = 3.0, crit: float = 4.5
) -> AnomalyResult:
    if not is_finite_scalar(value) or not np.isfinite(baseline.std) or baseline.std <= 0:
        return AnomalyResult(kpi=kpi, method="zscore", severity=Severity.DATA_QUALITY,
                             score=float("nan"), flagged=True, detail="non-finite value or baseline")
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
