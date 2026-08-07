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

import math
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


def _default_offline_recommend(kpis: list) -> list[dict]:
    """Domain-neutral offline recommendations. Each domain overrides this with its own language."""
    return [
        {"kpi": k, "capa_frame": f"REC-{i:03d}",
         "text": f"Review '{k}' against source data and escalate to the responsible owner for sign-off."}
        for i, k in enumerate(kpis, start=1)
    ]


@dataclass(frozen=True)
class Domain:
    """A complete, self-consistent domain configuration for the engine."""

    name: str
    generate: Callable[..., dict]          # (start, days, seed) -> {stream_name: DataFrame}
    extract: Callable[[dict, str], object]  # (streams, kpi_name) -> 1-D np.ndarray (raises on unusable input)
    kpi_specs: tuple[KpiSpec, ...]
    analyst_system: str                     # LLM system prompt for the analyst node (online/Claude)
    recommend_system: str                   # LLM system prompt for the recommend node (online/Claude)
    compliance: dict = field(default_factory=dict)  # content for governance/classification rendering
    # (streams) -> DataFrame(ds, y) for the hero KPI — Prophet needs the dated frame, not just the
    # value array `extract` returns. Optional; if absent, no forecast is published.
    hero_frame: Callable[[dict], object] | None = None
    # OFFLINE (keyless) framing: the deterministic stub can't be prompted, so the domain supplies
    # its own language. `offline_framing` completes the analyst sentence; `offline_recommend`
    # builds the recommendation dicts. Defaults are domain-neutral; each domain overrides them.
    offline_framing: str = (
        "A qualified owner should review any Warning/Critical KPI against source data before acting."
    )
    offline_recommend: Callable[[list], list] = _default_offline_recommend

    def __post_init__(self) -> None:
        # A Domain is a self-consistent VALUE: a malformed one must fail at CONSTRUCTION, never
        # degrade or fail open at run time. Validate hero count, unique names, detection-mode
        # type, and physically-sensible bounds (finite-or-None, lo <= hi).
        if not self.kpi_specs:
            raise ValueError(f"domain {self.name!r} has no KPIs")
        heroes = [k for k in self.kpi_specs if k.is_hero]
        if len(heroes) != 1:
            raise ValueError(
                f"domain {self.name!r} must declare exactly one hero KPI (is_hero=True), got {len(heroes)}"
            )
        names = [k.name for k in self.kpi_specs]
        if len(names) != len(set(names)):
            raise ValueError(f"domain {self.name!r} has duplicate KPI names: {names}")
        for k in self.kpi_specs:
            if not isinstance(k.detection_mode, DetectionMode):
                raise ValueError(f"KPI {k.name!r}: detection_mode must be a DetectionMode, got {k.detection_mode!r}")
            lo, hi = k.domain_bounds
            for edge, v in (("lower", lo), ("upper", hi)):
                if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)):
                    raise ValueError(f"KPI {k.name!r}: {edge} bound must be None or a finite number, got {v!r}")
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(f"KPI {k.name!r}: lower bound {lo} exceeds upper bound {hi}")

    def hero(self) -> str:
        """The KPI to forecast."""
        return next(k.name for k in self.kpi_specs if k.is_hero)

    def bounds(self) -> dict[str, tuple[float | None, float | None]]:
        """Per-KPI domain bounds — replaces the hardcoded KPI_DOMAINS table."""
        return {k.name: k.domain_bounds for k in self.kpi_specs}

    def spec(self, name: str) -> KpiSpec:
        try:
            return next(k for k in self.kpi_specs if k.name == name)
        except StopIteration:
            raise KeyError(f"domain {self.name!r} has no KPI named {name!r}") from None
