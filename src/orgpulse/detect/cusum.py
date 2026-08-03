from __future__ import annotations

import numpy as np

from orgpulse.detect.baseline import Baseline
from orgpulse.detect.types import AnomalyResult, Severity
from orgpulse.detect.validation import baseline_is_usable, ensure_1d_finite


_CALIB_TABLE: dict[int, tuple[float, float]] = {
    20: (4.31, 6.77), 30: (4.85, 7.76), 45: (5.47, 8.49), 60: (5.78, 9.11),
    90: (6.36, 9.97), 120: (6.77, 10.56), 180: (7.43, 11.39), 365: (8.41, 12.76),
}


def _cusum_peak(z: np.ndarray, k: float) -> tuple[float, str, int]:
    s_hi = s_lo = max_hi = max_lo = 0.0
    cross_hi = cross_lo = -1
    warn_ref = 5.0  # reference level only for the informational first-cross index
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
    n_recent: int, history_len: int = 90, k: float = 0.5, n_sims: int = 4000, seed: int = 10000
) -> tuple[float, float]:
    peaks = np.empty(n_sims)
    for s in range(n_sims):
        rng = np.random.default_rng(seed + s)
        base = rng.normal(50, 2, history_len)
        m, sd = base.mean(), max(base.std(), 1e-9)
        z = (rng.normal(50, 2, n_recent) - m) / sd
        peaks[s] = _cusum_peak(z, k)[0]
    return float(np.percentile(peaks, 90)), float(np.percentile(peaks, 99))


def cusum_detect(
    values: np.ndarray, baseline: Baseline, kpi: str, k: float = 0.5
) -> AnomalyResult:
    try:
        arr = ensure_1d_finite(values, "cusum values")
    except ValueError as exc:
        return AnomalyResult(kpi=kpi, method="cusum", severity=Severity.DATA_QUALITY,
                             score=float("nan"), flagged=True, detail=f"data quality: {exc}")
    if not baseline_is_usable(baseline.mean, baseline.std):
        return AnomalyResult(kpi=kpi, method="cusum", severity=Severity.DATA_QUALITY,
                             score=float("nan"), flagged=True, detail="unusable baseline (non-finite or std<=0)")
    z = (arr - baseline.mean) / baseline.std
    score, side, cross_idx = _cusum_peak(z, k)
    warn, crit = calibrated_thresholds(len(arr))
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
