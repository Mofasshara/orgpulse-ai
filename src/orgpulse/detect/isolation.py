from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from orgpulse.detect.types import AnomalyResult, Severity


def isolation_detect(
    history: np.ndarray,
    current: np.ndarray,
    kpi: str,
    contamination: float = 0.05,
    random_state: int = 42,
) -> AnomalyResult:
    def _dq(detail): return AnomalyResult(kpi, "isolation_forest", Severity.DATA_QUALITY, None, True, detail)
    try:
        hist = np.asarray(history, dtype=float)
        cur = np.asarray(current, dtype=float)
    except (ValueError, TypeError, OverflowError):
        # Non-numeric (e.g. [["bad"]]) or too-large-for-float (e.g. 10**10000) input is corrupt,
        # not a crash — same standalone-safe contract as the shared validation gates.
        return _dq("unusable input (non-numeric or out-of-range)")
    if hist.ndim != 2 or hist.shape[0] < 1 or not np.all(np.isfinite(hist)):
        return _dq("unusable history")
    if cur.ndim != 1 or cur.shape[0] != hist.shape[1] or not np.all(np.isfinite(cur)):
        return _dq("unusable current vector")
    cur = cur.reshape(1, -1)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(hist)
    pred = int(model.predict(cur)[0])                 # -1 anomaly, +1 normal
    neg_score = float(-model.decision_function(cur)[0])  # higher = more anomalous
    if pred == -1 and neg_score >= 0.10:
        sev = Severity.CRITICAL
    elif pred == -1:
        sev = Severity.WARNING
    else:
        sev = Severity.SAFE
    return AnomalyResult(
        kpi=kpi,
        method="isolation_forest",
        severity=sev,
        score=neg_score,
        flagged=sev > Severity.SAFE,
        detail=f"iforest neg_score={neg_score:.3f}, predict={pred}",
    )
