from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orgpulse.detect.validation import ensure_1d_finite

_EPS = 1e-9


@dataclass(frozen=True)
class Baseline:
    mean: float
    std: float
    n: int


def fit_baseline(values: np.ndarray) -> Baseline:
    arr = ensure_1d_finite(values, "baseline values")
    std = float(np.std(arr, ddof=0))
    return Baseline(mean=float(np.mean(arr)), std=max(std, _EPS), n=int(arr.size))
