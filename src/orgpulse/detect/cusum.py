from __future__ import annotations

import numpy as np

from orgpulse.detect.baseline import Baseline
from orgpulse.detect.types import AnomalyResult, Severity
from orgpulse.detect.validation import baseline_is_usable, ensure_1d_finite, finite_nonneg


_MIN_BASELINE_N: int = 60

_CALIB_TABLE: dict[int, tuple[float, float]] = {
    20: (4.47, 7.0), 30: (5.12, 7.96), 45: (5.68, 9.26), 60: (6.13, 9.71),
    90: (6.75, 11.1), 120: (7.25, 12.06), 180: (8.01, 13.37), 365: (9.19, 15.2),
}


def _cusum_peak(z: np.ndarray, k: float, warn_ref: float) -> tuple[float, str, int]:
    s_hi = s_lo = max_hi = max_lo = 0.0
    cross_hi = cross_lo = -1
    for i, zi in enumerate(z):
        s_hi = max(0.0, s_hi + zi - k)
        s_lo = max(0.0, s_lo - zi - k)
        if s_hi > max_hi:
            max_hi = s_hi
        if s_lo > max_lo:
            max_lo = s_lo
        if cross_hi == -1 and s_hi >= warn_ref:
            cross_hi = i
        if cross_lo == -1 and s_lo >= warn_ref:
            cross_lo = i
    if max_hi >= max_lo:
        return max_hi, "high", cross_hi
    return max_lo, "low", cross_lo


def calibrated_thresholds(n_recent: int) -> tuple[float, float]:
    """(warn, crit) peak thresholds for a recent-window length; interpolated from the hist=60 calibration table, clamped outside [20, 365]."""
    keys = sorted(_CALIB_TABLE)
    if n_recent <= keys[0]:
        return _CALIB_TABLE[keys[0]]
    if n_recent >= keys[-1]:
        return _CALIB_TABLE[keys[-1]]
    lo = max(k for k in keys if k <= n_recent)
    hi = min(k for k in keys if k >= n_recent)
    if lo == hi:
        return _CALIB_TABLE[lo]
    frac = (n_recent - lo) / (hi - lo)
    w = _CALIB_TABLE[lo][0] + frac * (_CALIB_TABLE[hi][0] - _CALIB_TABLE[lo][0])
    c = _CALIB_TABLE[lo][1] + frac * (_CALIB_TABLE[hi][1] - _CALIB_TABLE[lo][1])
    return w, c


def simulate_incontrol_quantiles(
    n_recent: int, history_len: int = 60, k: float = 0.5, n_sims: int = 6000, seed: int = 10000
) -> tuple[float, float]:
    peaks = np.empty(n_sims)
    for s in range(n_sims):
        rng = np.random.default_rng(seed + s)
        base = rng.normal(50, 2, history_len)
        m, sd = base.mean(), max(base.std(), 1e-9)
        z = (rng.normal(50, 2, n_recent) - m) / sd
        peaks[s] = _cusum_peak(z, k, warn_ref=5.0)[0]
    return float(np.percentile(peaks, 90)), float(np.percentile(peaks, 99))


def cusum_detect(
    values: np.ndarray, baseline: Baseline, kpi: str, k: float = 0.5
) -> AnomalyResult:
    """Two-sided standardized CUSUM with severity calibrated to window length.

    Severity thresholds achieve an in-control false-alarm target of ~1% CRITICAL /
    ~10% WARNING. They are calibrated for a baseline of at least _MIN_BASELINE_N (60)
    observations — shorter baselines return DATA_QUALITY — and for a recent-window
    length in [20, 365]; beyond that range thresholds are clamped and false-alarm
    control degrades. Returns DATA_QUALITY (never SAFE) on unusable baseline/params/data.
    """
    def _dq(detail):
        return AnomalyResult(kpi=kpi, method="cusum", severity=Severity.DATA_QUALITY, score=None, flagged=True, detail=detail)

    try:
        arr = ensure_1d_finite(values, "cusum values")
    except ValueError as exc:
        return _dq(f"data quality: {exc}")

    if not baseline_is_usable(baseline.mean, baseline.std, baseline.n):
        return _dq("unusable baseline")
    if baseline.n < _MIN_BASELINE_N:
        return _dq("insufficient baseline (<60)")
    if not finite_nonneg(k):
        return _dq("invalid k")

    z = (arr - baseline.mean) / baseline.std
    warn, crit = calibrated_thresholds(len(arr))
    score, side, cross_idx = _cusum_peak(z, k, warn)
    if not np.isfinite(score):
        return _dq("non-finite score")

    if score >= crit:
        sev = Severity.CRITICAL
    elif score >= warn:
        sev = Severity.WARNING
    else:
        sev = Severity.SAFE
    detail = f"{side} CUSUM peak={score:.2f} (warn={warn:.2f}, crit={crit:.2f}, n={len(arr)}); first cross idx {cross_idx}"
    return AnomalyResult(
        kpi=kpi,
        method="cusum",
        severity=sev,
        score=float(score),
        flagged=sev > Severity.SAFE,
        detail=detail,
    )
