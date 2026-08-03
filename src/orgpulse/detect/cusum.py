from __future__ import annotations

import numpy as np

from orgpulse.detect.baseline import Baseline
from orgpulse.detect.types import AnomalyResult, Severity


def cusum_detect(
    values: np.ndarray, baseline: Baseline, kpi: str, k: float = 0.5, h: float = 5.0
) -> AnomalyResult:
    z = (np.asarray(values, dtype=float) - baseline.mean) / baseline.std
    s_hi = 0.0
    s_lo = 0.0
    max_hi = 0.0
    max_lo = 0.0
    cross_idx = -1
    for i, zi in enumerate(z):
        s_hi = max(0.0, s_hi + zi - k)
        s_lo = max(0.0, s_lo - zi - k)
        max_hi = max(max_hi, s_hi)
        max_lo = max(max_lo, s_lo)
        if cross_idx == -1 and (s_hi >= h or s_lo >= h):
            cross_idx = i
    score = max(max_hi, max_lo)
    side = "high" if max_hi >= max_lo else "low"
    if score >= 2.0 * h:
        sev = Severity.CRITICAL
    elif score >= h:
        sev = Severity.WARNING
    else:
        sev = Severity.SAFE
    return AnomalyResult(
        kpi=kpi,
        method="cusum",
        severity=sev,
        score=float(score),
        flagged=sev > Severity.SAFE,
        detail=f"{side} CUSUM peak={score:.2f} (h={h}); first cross at idx {cross_idx}",
    )
