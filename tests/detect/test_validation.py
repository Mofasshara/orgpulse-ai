import json
from dataclasses import asdict

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


def test_corrupt_baseline_is_data_quality_not_safe():
    for b in (Baseline(mean=float("nan"), std=10.0, n=5),
              Baseline(mean=100.0, std=float("nan"), n=5),
              Baseline(mean=100.0, std=0.0, n=5)):
        rz = zscore_detect(50.0, b, kpi="k")
        rc = cusum_detect(np.array([1.0, 2.0, 3.0]), b, kpi="k")
        assert rz.severity is Severity.DATA_QUALITY, f"zscore leaked {rz.severity} for {b}"
        assert rc.severity is Severity.DATA_QUALITY, f"cusum leaked {rc.severity} for {b}"


def test_baseline_zero_n_is_data_quality():
    b = Baseline(mean=0.0, std=1.0, n=0)
    assert zscore_detect(0.0, b, kpi="k").severity is Severity.DATA_QUALITY
    assert cusum_detect(np.array([1.0, 2.0, 3.0]), b, kpi="k").severity is Severity.DATA_QUALITY


def test_invalid_params_are_data_quality_not_safe():
    b = Baseline(mean=100.0, std=10.0, n=90)
    assert zscore_detect(0.0, b, kpi="k", warn=float("nan"), crit=float("nan")).severity is Severity.DATA_QUALITY
    assert zscore_detect(0.0, b, kpi="k", warn=5.0, crit=1.0).severity is Severity.DATA_QUALITY  # warn>=crit
    assert cusum_detect(np.array([1.0, 2.0, 3.0]), b, kpi="k", k=float("nan")).severity is Severity.DATA_QUALITY


def test_overflow_standardization_is_data_quality_not_critical():
    b = Baseline(mean=-1e308, std=1.0, n=90)
    r = zscore_detect(1e308, b, kpi="k")
    assert r.severity is Severity.DATA_QUALITY   # not a fabricated CRITICAL with inf score


def test_data_quality_score_is_none_and_json_serializable():
    b = Baseline(mean=100.0, std=10.0, n=90)
    r = zscore_detect(float("nan"), b, kpi="k")
    assert r.score is None
    json.dumps(asdict(r) | {"severity": int(r.severity)}, allow_nan=False)  # must not raise
