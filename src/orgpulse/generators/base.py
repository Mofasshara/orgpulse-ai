from __future__ import annotations

from datetime import date, timedelta

import numpy as np


def date_series(start: date, days: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(days)]


def logistic_curve(days: int, ceiling: float, midpoint: float, steepness: float) -> np.ndarray:
    t = np.arange(days)
    return ceiling / (1.0 + np.exp(-steepness * (t - midpoint)))


def is_holiday_dip(d: date) -> float:
    # Enrollment slows in the winter-holiday window (last 10 days of December).
    if d.month == 12 and d.day >= 22:
        return 0.4
    return 1.0
