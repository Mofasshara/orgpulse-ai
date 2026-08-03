from datetime import date

import numpy as np

from orgpulse.generators.base import date_series, logistic_curve, is_holiday_dip


def test_date_series_length_and_spacing():
    ds = date_series(date(2026, 1, 1), 5)
    assert ds[0] == date(2026, 1, 1)
    assert ds[-1] == date(2026, 1, 5)
    assert len(ds) == 5
    assert (ds[1] - ds[0]).days == 1


def test_logistic_curve_is_monotonic_and_bounded():
    curve = logistic_curve(days=100, ceiling=500.0, midpoint=50.0, steepness=0.1)
    assert len(curve) == 100
    assert np.all(np.diff(curve) >= -1e-9)          # non-decreasing
    assert curve[-1] <= 500.0 + 1e-6                # bounded by ceiling
    assert curve[-1] > curve[0]                     # actually grows


def test_holiday_dip_reduces_late_december():
    assert is_holiday_dip(date(2026, 12, 24)) < 1.0
    assert is_holiday_dip(date(2026, 6, 15)) == 1.0
