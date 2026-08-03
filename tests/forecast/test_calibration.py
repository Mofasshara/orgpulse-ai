from datetime import date

import numpy as np

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical
from orgpulse.forecast.prophet_forecaster import series_from_generator
from orgpulse.forecast.calibration import interval_calibration
from orgpulse.forecast.types import CalibrationResult


def _series(days=260):
    df = generate_clinical(date(2026, 1, 1), days, make_rng(42))
    return series_from_generator(df, "date", "daily_enrollment")


def test_calibration_detects_and_corrects_overconfidence():
    r = interval_calibration(_series(), horizon=30, n_folds=4, nominal=0.9)
    assert isinstance(r, CalibrationResult)
    assert r.nominal == 0.9 and r.n_points == 120
    # Prophet's raw intervals are overconfident here (measured ~0.70):
    assert r.raw_coverage < r.nominal
    # The empirical scale widens them (>= 1) and brings coverage up to ~nominal:
    assert r.scale_factor >= 1.0
    assert r.adjusted_coverage >= r.raw_coverage
    assert abs(r.adjusted_coverage - r.nominal) <= 0.08     # within 8pts of nominal
