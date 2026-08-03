from __future__ import annotations

import numpy as np


def ensure_1d_finite(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {arr.ndim}-D")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN/inf)")
    return arr


def is_finite_scalar(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def baseline_is_usable(mean, std, n) -> bool:
    try:
        return is_finite_scalar(mean) and is_finite_scalar(std) and float(std) > 0.0 and int(n) >= 1
    except (TypeError, ValueError):
        return False


def finite_positive_order(warn, crit) -> bool:
    try:
        return is_finite_scalar(warn) and is_finite_scalar(crit) and 0.0 < float(warn) < float(crit)
    except (TypeError, ValueError):
        return False


def finite_nonneg(x) -> bool:
    try:
        return is_finite_scalar(x) and float(x) >= 0.0
    except (TypeError, ValueError):
        return False
