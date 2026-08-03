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
    hist = np.asarray(history, dtype=float)
    cur = np.asarray(current, dtype=float).reshape(1, -1)
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
