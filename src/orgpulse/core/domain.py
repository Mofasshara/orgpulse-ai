"""The single domain-abstraction seam for OrgPulse.

The engine (detection, forecasting, the LangGraph DAG, the autonomous tick, the audit trail, the
dashboard loader) is domain-agnostic. Everything that differs between pharma and banking — which
data streams exist, which KPIs they carry, each KPI's real-world domain and detection mode, the
hero KPI to forecast, the analyst/recommendation framing, and the compliance content — lives in a
``Domain`` value. Adding a new regulated domain = constructing a new ``Domain``, not editing the
engine.

This is a *code interface* (generators and extractors are functions), not a "swap-a-YAML-file"
platform — describe it as such.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class DetectionMode(str, Enum):
    """How a KPI is assessed. Only TIMESERIES is wired in the core→domain refactor; THRESHOLD and
    DATEMATH are part of the data model now and get their engine handlers in the banking build."""

    TIMESERIES = "timeseries"   # z-score + window-calibrated CUSUM against a remembered baseline
    THRESHOLD = "threshold"     # hard min/max rules (e.g. regulatory capital floors)
    DATEMATH = "datemath"       # days-until-deadline arithmetic (e.g. filing due dates)


@dataclass(frozen=True)
class KpiSpec:
    """One monitored KPI: its name, real-world domain bounds, how it's detected, whether it's the
    forecast hero, and (for THRESHOLD mode) its warn/critical thresholds."""

    name: str
    domain_bounds: tuple[float | None, float | None] = (None, None)  # (lo, hi) inclusive; None = unbounded
    detection_mode: DetectionMode = DetectionMode.TIMESERIES
    is_hero: bool = False
    thresholds: dict | None = None  # THRESHOLD mode: e.g. {"warn": 11.5, "critical": 10.5, "direction": "below"}


@dataclass(frozen=True)
class Domain:
    """A complete, self-consistent domain configuration for the engine."""

    name: str
    generate: Callable[..., dict]          # (start, days, seed) -> {stream_name: DataFrame}
    extract: Callable[[dict, str], object]  # (streams, kpi_name) -> 1-D np.ndarray (raises on unusable input)
    kpi_specs: tuple[KpiSpec, ...]
    analyst_system: str                     # LLM system prompt for the analyst node
    recommend_system: str                   # LLM system prompt for the recommend node
    compliance: dict = field(default_factory=dict)  # content for governance/classification rendering

    def __post_init__(self) -> None:
        heroes = [k for k in self.kpi_specs if k.is_hero]
        if len(heroes) != 1:
            raise ValueError(
                f"domain {self.name!r} must declare exactly one hero KPI (is_hero=True), got {len(heroes)}"
            )
        names = [k.name for k in self.kpi_specs]
        if len(names) != len(set(names)):
            raise ValueError(f"domain {self.name!r} has duplicate KPI names: {names}")

    def hero(self) -> str:
        """The KPI to forecast."""
        return next(k.name for k in self.kpi_specs if k.is_hero)

    def bounds(self) -> dict[str, tuple[float | None, float | None]]:
        """Per-KPI domain bounds — replaces the hardcoded KPI_DOMAINS table."""
        return {k.name: k.domain_bounds for k in self.kpi_specs}

    def spec(self, name: str) -> KpiSpec:
        return next(k for k in self.kpi_specs if k.name == name)
