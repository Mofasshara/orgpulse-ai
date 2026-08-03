import numpy as np
import pytest

from orgpulse.detect.validation import ensure_1d_finite, is_finite_scalar
from orgpulse.detect.baseline import fit_baseline
from orgpulse.detect.zscore import zscore_detect
from orgpulse.detect.cusum import cusum_detect
from orgpulse.detect.baseline import Baseline
from orgpulse.detect.orchestrator import overall_severity, detect_kpi
from orgpulse.detect.types import Severity, AnomalyResult


def test_ensure_1d_finite_rejects_bad_input():
    with pytest.raises(ValueError):
        ensure_1d_finite(np.array([]), "x")
    with pytest.raises(ValueError):
        ensure_1d_finite(np.array([1.0, np.nan]), "x")
    with pytest.raises(ValueError):
        ensure_1d_finite(np.array([[1.0, 2.0]]), "x")   # 2-D
    assert ensure_1d_finite(np.array([1.0, 2.0]), "x").tolist() == [1.0, 2.0]


def test_fit_baseline_rejects_nan_and_2d():
    with pytest.raises(ValueError):
        fit_baseline(np.array([1.0, np.nan, 2.0]))
    with pytest.raises(ValueError):
        fit_baseline(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_nonfinite_value_is_data_quality_not_safe():
    b = Baseline(mean=100.0, std=10.0, n=50)
    r = zscore_detect(float("nan"), b, kpi="k")
    assert r.severity is Severity.DATA_QUALITY and r.flagged is True
    assert r.severity is not Severity.SAFE


def test_cusum_nonfinite_series_is_data_quality_not_safe():
    b = Baseline(mean=100.0, std=10.0, n=50)
    r = cusum_detect(np.array([np.nan, np.nan]), b, kpi="k")
    assert r.severity is Severity.DATA_QUALITY and r.severity is not Severity.SAFE


def test_overall_severity_data_quality_dominates():
    dq = AnomalyResult("k", "zscore", Severity.DATA_QUALITY, float("nan"), True, "")
    safe = AnomalyResult("k", "cusum", Severity.SAFE, 0.0, False, "")
    assert overall_severity([dq, safe]) is Severity.DATA_QUALITY
