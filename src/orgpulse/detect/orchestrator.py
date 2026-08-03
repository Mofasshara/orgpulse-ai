from __future__ import annotations

import numpy as np

from orgpulse.detect.baseline import fit_baseline
from orgpulse.detect.cusum import cusum_detect
from orgpulse.detect.types import AnomalyResult, Severity
from orgpulse.detect.zscore import zscore_detect


def detect_kpi(history: np.ndarray, recent: np.ndarray, kpi: str) -> list[AnomalyResult]:
    baseline = fit_baseline(np.asarray(history, dtype=float))
    recent = np.asarray(recent, dtype=float)
    return [
        zscore_detect(float(recent[-1]), baseline, kpi=kpi),
        cusum_detect(recent, baseline, kpi=kpi),
    ]


def overall_severity(results: list[AnomalyResult]) -> Severity:
    if not results:
        return Severity.SAFE
    return max(r.severity for r in results)
