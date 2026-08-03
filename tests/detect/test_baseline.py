import numpy as np
import pytest

from orgpulse.detect.types import Severity, AnomalyResult
from orgpulse.detect.baseline import Baseline, fit_baseline


def test_severity_is_ordered():
    assert Severity.CRITICAL > Severity.WARNING > Severity.SAFE
    assert max(Severity.SAFE, Severity.CRITICAL) is Severity.CRITICAL


def test_anomaly_result_fields():
    r = AnomalyResult(kpi="k", method="zscore", severity=Severity.WARNING,
                      score=3.1, flagged=True, detail="d")
    assert r.kpi == "k" and r.flagged is True and r.severity is Severity.WARNING


def test_fit_baseline_mean_std():
    b = fit_baseline(np.array([10.0, 12.0, 14.0]))
    assert b.mean == pytest.approx(12.0)
    assert b.std == pytest.approx(np.std([10, 12, 14]))  # population std
    assert b.n == 3


def test_fit_baseline_zero_std_floored():
    b = fit_baseline(np.array([5.0, 5.0, 5.0]))
    assert b.std >= 1e-9 and b.std < 1e-6


def test_fit_baseline_empty_raises():
    with pytest.raises(ValueError):
        fit_baseline(np.array([]))
