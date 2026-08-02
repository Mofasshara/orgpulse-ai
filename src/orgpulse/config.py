from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OrgPulseConfig:
    seed: int = 42
    start_date: date = date(2026, 1, 1)
    n_sites: int = 8
    runs_dir: Path = field(default_factory=lambda: Path("runs"))


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
