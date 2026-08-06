"""The PHARMA domain — the first configuration of the OrgPulse engine.

Streams: clinical (patient enrollment), QC/lab, supply chain. Hero KPI: daily_enrollment.
All detection is TIMESERIES. Framing is GxP / ICH-E6(R3) / CAPA. This module constructs the
``Domain`` value; the engine consumes it. Values here reproduce the previously-hardcoded pharma
behavior exactly (bounds match the old KPI_DOMAINS; prompts match the old node system strings).
"""
from __future__ import annotations

from datetime import date

from orgpulse.config import make_rng
from orgpulse.core.domain import Domain, KpiSpec
from orgpulse.generators.clinical import generate_clinical
from orgpulse.generators.qc import generate_qc
from orgpulse.generators.supply import generate_supply


def _generate(start: date, days: int, seed: int) -> dict:
    # Same seeding as the CLI/tick: seed, seed+1, seed+2 for the three streams.
    return {
        "clinical": generate_clinical(start, days, make_rng(seed)),
        "qc": generate_qc(start, days, make_rng(seed + 1)),
        "supply": generate_supply(start, days, make_rng(seed + 2)),
    }


def _extract(streams: dict, kpi: str):
    # Lazy import: validated_clinical_daily currently lives in pipeline.nodes; importing it lazily
    # avoids a module-load cycle once the pipeline imports this domain.
    if kpi == "daily_enrollment":
        from orgpulse.pipeline.nodes import validated_clinical_daily
        return validated_clinical_daily(streams["clinical"])["y"].to_numpy()
    if kpi in ("pass_rate", "calibration_offset"):
        return streams["qc"][kpi].to_numpy()
    if kpi == "lead_time_days":
        return streams["supply"]["lead_time_days"].to_numpy()
    raise KeyError(f"unknown pharma KPI: {kpi}")


PHARMA = Domain(
    name="pharma",
    generate=_generate,
    extract=_extract,
    kpi_specs=(
        KpiSpec("daily_enrollment", (0.0, None), is_hero=True),  # nonnegative counts; forecast hero
        KpiSpec("pass_rate", (0.0, 1.0)),                        # a proportion
        KpiSpec("calibration_offset", (None, None)),             # unbounded offset
        KpiSpec("lead_time_days", (0.0, None)),                  # nonnegative duration
    ),
    analyst_system="You are a clinical data quality analyst.",
    recommend_system="You are a quality assurance CAPA expert.",
    compliance={
        "context": "GxP-regulated pharmaceutical operations (clinical trials, QC/lab, supply chain)",
    },
)
