from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypedDict

import pandas as pd

HUMAN_OVERSIGHT_DISCLAIMER = (
    "Decision support only — not an automated decision. A qualified human must "
    "review and approve before any action (EU AI Act Art. 14 human oversight)."
)


@dataclass(frozen=True)
class KpiFinding:
    kpi: str
    severity: str
    detail: str
    latest_value: float | None


@dataclass(frozen=True)
class Recommendation:
    kpi: str
    text: str
    capa_frame: str
    disclaimer: str
    review_status: str = "PENDING_HUMAN_REVIEW"


@dataclass(frozen=True)
class Findings:
    run_id: str
    generated_at: str
    overall_severity: str
    alert: bool
    kpi_findings: list[KpiFinding]
    forecast: dict
    analysis: str
    recommendations: list[Recommendation]

    def to_json_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "overall_severity": self.overall_severity,
            "alert": self.alert,
            "kpi_findings": [asdict(k) for k in self.kpi_findings],
            "forecast": self.forecast,
            "analysis": self.analysis,
            "recommendations": [asdict(r) for r in self.recommendations],
        }


class PipelineState(TypedDict, total=False):
    run_id: str
    generated_at: str
    offline: bool
    clinical: pd.DataFrame
    qc: pd.DataFrame
    supply: pd.DataFrame
    kpi_findings: list[KpiFinding]
    overall_severity: str
    forecast: dict
    analysis: str
    recommendations: list[Recommendation]
    alert: bool
