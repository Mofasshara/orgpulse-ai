from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Severity(IntEnum):
    DATA_QUALITY = -1
    SAFE = 0
    WARNING = 1
    CRITICAL = 2


@dataclass(frozen=True)
class AnomalyResult:
    kpi: str
    method: str
    severity: Severity
    score: float | None
    flagged: bool
    detail: str
