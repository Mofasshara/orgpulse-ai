from __future__ import annotations

import numpy as np

from orgpulse.detect.baseline import fit_baseline
from orgpulse.detect.cusum import cusum_detect
from orgpulse.detect.types import AnomalyResult, Severity
from orgpulse.detect.validation import ensure_1d_finite
from orgpulse.detect.zscore import zscore_detect


def detect_kpi(history: np.ndarray, recent: np.ndarray, kpi: str) -> list[AnomalyResult]:
    def _dq(detail):
        return [AnomalyResult(kpi, "detect_kpi", Severity.DATA_QUALITY, None, True, detail)]

    try:
        hist = ensure_1d_finite(history, "history")
        rec = ensure_1d_finite(recent, "recent")
    except (ValueError, TypeError) as exc:
        return _dq(f"data quality: {exc}")
    try:
        baseline = fit_baseline(hist)
    except (ValueError, TypeError) as exc:
        return _dq(f"data quality (baseline): {exc}")
    return [zscore_detect(float(rec[-1]), baseline, kpi=kpi), cusum_detect(rec, baseline, kpi=kpi)]


def overall_severity(results: list[AnomalyResult]) -> Severity:
    if not results:
        return Severity.DATA_QUALITY
    if any(r.severity is Severity.DATA_QUALITY for r in results):
        return Severity.DATA_QUALITY
    return max(r.severity for r in results)
