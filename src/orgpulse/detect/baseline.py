from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-9


@dataclass(frozen=True)
class Baseline:
    mean: float
    std: float
    n: int


def fit_baseline(values: np.ndarray) -> Baseline:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("fit_baseline requires at least one value")
    std = float(np.std(arr, ddof=0))
    return Baseline(mean=float(np.mean(arr)), std=max(std, _EPS), n=int(arr.size))
