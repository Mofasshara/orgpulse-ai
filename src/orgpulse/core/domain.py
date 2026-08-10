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
    """How a KPI is assessed. All three modes are wired into the engine (detection dispatches on
    the KPI's mode in both the run pipeline and the autonomous tick) and mutation-tested."""

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


def _default_dq_recommendation(kpi: str) -> str:
    """Domain-neutral text for a data-quality finding. Each domain overrides with its own language."""
    return (
        f"Data-quality issue on '{kpi}': the detector could not assess this KPI (missing, corrupt, "
        "or insufficient data). Investigate the data source and instrumentation before drawing any "
        "conclusion; do not treat this as a genuine anomaly until the data is fixed."
    )


@dataclass(frozen=True)
class Domain:
    """A complete, self-consistent domain configuration for the engine."""

    name: str
    # (start, days, seed) -> {stream_name: DataFrame}. CONTRACT: return streams in a STABLE order
    # (Python dicts preserve construction order) — the tick's input_data_hash concatenates
    # streams.values(), so a nondeterministic order would produce unstable lineage hashes.
    generate: Callable[..., dict]
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
    dq_recommendation: Callable[[str], str] = _default_dq_recommendation  # (kpi) -> data-quality rec text

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
            if k.detection_mode is DetectionMode.THRESHOLD:
                # THRESHOLD/DATEMATH thresholds drive run-time float() arithmetic, so a malformed
                # one must fail HERE, at construction — never crash a run, silently return the
                # wrong severity, or hide a typo. Reject: non-dict configs (including falsy ones
                # like [] / "" / False); unknown keys (a misspelled 'warn_day' would otherwise
                # silently fall back to the default, masking the intended value); non-finite/
                # non-numeric cutoffs; and any band ordering that makes a severity UNREACHABLE
                # (WARNING must be attainable — see the strict comparisons below).
                th = k.thresholds
                if not isinstance(th, dict):
                    raise ValueError(f"KPI {k.name!r}: THRESHOLD mode requires a thresholds dict, got {th!r}")
                allowed = {"warn", "critical", "direction"}
                unknown = set(th) - allowed
                if unknown:
                    raise ValueError(f"KPI {k.name!r}: THRESHOLD thresholds has unknown key(s) {sorted(unknown)}; "
                                     f"allowed keys are {sorted(allowed)}")
                nums = {}
                for x in ("warn", "critical"):
                    v = th.get(x)
                    if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)):
                        raise ValueError(f"KPI {k.name!r}: THRESHOLD '{x}' must be a finite number, got {v!r}")
                    nums[x] = float(v)
                direction = th.get("direction", "below")
                if direction not in ("below", "above"):
                    raise ValueError(f"KPI {k.name!r}: threshold 'direction' must be 'below' or 'above'")
                # CRITICAL must sit STRICTLY further in the breach direction than WARNING; otherwise a
                # band is unreachable (e.g. below with critical >= warn: every value <= warn is also
                # <= critical, so WARNING never fires — a collapsed two-state config, not a supported mode).
                if direction == "below" and nums["critical"] >= nums["warn"]:
                    raise ValueError(f"KPI {k.name!r}: for direction 'below', critical ({nums['critical']:g}) "
                                     f"must be < warn ({nums['warn']:g}) so WARNING is reachable")
                if direction == "above" and nums["critical"] <= nums["warn"]:
                    raise ValueError(f"KPI {k.name!r}: for direction 'above', critical ({nums['critical']:g}) "
                                     f"must be > warn ({nums['warn']:g}) so WARNING is reachable")
            if k.detection_mode is DetectionMode.DATEMATH:
                # Only None means "use the default 14/3-day bands". A falsy non-dict ([] / "" / False)
                # is a malformed config, not a request for defaults — reject it. Unknown keys are
                # rejected for the same typo-masking reason as THRESHOLD.
                th = {} if k.thresholds is None else k.thresholds
                if not isinstance(th, dict):
                    raise ValueError(f"KPI {k.name!r}: DATEMATH thresholds must be a dict or None, got {k.thresholds!r}")
                allowed = {"warn_days", "critical_days"}
                unknown = set(th) - allowed
                if unknown:
                    raise ValueError(f"KPI {k.name!r}: DATEMATH thresholds has unknown key(s) {sorted(unknown)}; "
                                     f"allowed keys are {sorted(allowed)}")
                day_vals = {}
                for x, default in (("warn_days", 14.0), ("critical_days", 3.0)):
                    v = th.get(x, default)
                    if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)) or v < 0:
                        raise ValueError(f"KPI {k.name!r}: DATEMATH '{x}' must be a finite number >= 0, got {v!r}")
                    day_vals[x] = float(v)
                # WARNING band = (critical_days, warn_days]; require critical_days STRICTLY < warn_days
                # so it is non-empty (same reachability rule as THRESHOLD).
                if day_vals["critical_days"] >= day_vals["warn_days"]:
                    raise ValueError(f"KPI {k.name!r}: DATEMATH critical_days ({day_vals['critical_days']:g}) "
                                     f"must be < warn_days ({day_vals['warn_days']:g}) so WARNING is reachable")
        for cb in ("generate", "extract", "offline_recommend", "dq_recommendation"):
            if not callable(getattr(self, cb)):
                raise ValueError(f"domain {self.name!r}: {cb} must be callable, got {getattr(self, cb)!r}")
        if self.hero_frame is not None and not callable(self.hero_frame):
            raise ValueError(f"domain {self.name!r}: hero_frame must be callable or None")
        # The framing fields are consumed as strings (online LLM system prompts; the offline stub's
        # sentence). A non-string would crash the online analyst/recommend call — reject at construction.
        for sfield in ("analyst_system", "recommend_system", "offline_framing"):
            if not isinstance(getattr(self, sfield), str):
                raise ValueError(f"domain {self.name!r}: {sfield} must be a string, got {type(getattr(self, sfield)).__name__}")

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
