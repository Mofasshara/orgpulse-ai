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


# Per-KPI DOMAIN constraints (lo, hi) — inclusive; None means unbounded on that side.
# A value outside its domain is impossible real-world data (a negative enrollment count, a
# pass rate above 1) and must be DATA_QUALITY, not a statistically "stable" SAFE. Offsets are
# unbounded (calibration drift can be negative), so calibration_offset is intentionally absent.
KPI_DOMAINS: dict[str, tuple[float | None, float | None]] = {
    "daily_enrollment": (0.0, None),   # nonnegative counts
    "pass_rate": (0.0, 1.0),           # a proportion
    "lead_time_days": (0.0, None),     # nonnegative duration
}


def kpi_domain_reason(kpi: str, arr: np.ndarray) -> str | None:
    """Return a DATA_QUALITY reason if any value violates the KPI's domain, else None.
    Assumes `arr` is already finite (call after the finite check)."""
    bounds = KPI_DOMAINS.get(kpi)
    if bounds is None:
        return None
    lo, hi = bounds
    a = np.asarray(arr, dtype=float)
    if lo is not None and np.any(a < lo):
        return f"value below domain minimum {lo} for {kpi}"
    if hi is not None and np.any(a > hi):
        return f"value above domain maximum {hi} for {kpi}"
    return None


def kpi_series_reason(kpi: str, series, *, min_len: int | None = None) -> str | None:
    """THE single "is this KPI series usable?" check, shared by every consumer (detection,
    forecasting, the tick). Returns a DATA_QUALITY reason or None. Checks: numeric &
    convertible, 1-D, optional minimum length, all-finite, and in-domain. Centralizing this
    is deliberate — round after round, a second consumer that re-validated on its own forgot
    a case and let bad data through.
    """
    try:
        a = np.asarray(series, dtype=float)
    except (ValueError, TypeError):
        return f"unusable input (non-numeric or ragged) for {kpi}"
    if a.ndim != 1:
        return f"unusable input ({a.ndim}-D, expected 1-D) for {kpi}"
    if min_len is not None and len(a) < min_len:
        return f"insufficient data ({len(a)} points < {min_len} required) for {kpi}"
    if a.size == 0 or not np.isfinite(a).all():
        return f"non-finite values (NaN/inf) for {kpi}"
    return kpi_domain_reason(kpi, a)
