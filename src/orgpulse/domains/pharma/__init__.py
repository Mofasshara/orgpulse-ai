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


def _hero_frame(streams: dict):
    # The hero (daily_enrollment) as a validated daily (ds, y) frame — what Prophet needs.
    from orgpulse.pipeline.nodes import validated_clinical_daily
    return validated_clinical_daily(streams["clinical"])


def _lineage_extract(streams: dict):
    # One (date, source_tier) pair per contributing clinical record, aligned row-for-row with
    # the raw stream (not the daily-aggregated hero frame): lineage_confidence does its own
    # per-day reduction over these raw records.
    clinical = streams["clinical"]
    return clinical["date"].to_numpy(), clinical["source_tier"].to_numpy()


def _extract(streams: dict, kpi: str):
    # Lazy import: validated_clinical_daily currently lives in pipeline.nodes; importing it lazily
    # avoids a module-load cycle once the pipeline imports this domain.
    if kpi == "daily_enrollment":
        return _hero_frame(streams)["y"].to_numpy()
    if kpi in ("pass_rate", "calibration_offset"):
        return streams["qc"][kpi].to_numpy()
    if kpi == "lead_time_days":
        return streams["supply"]["lead_time_days"].to_numpy()
    raise KeyError(f"unknown pharma KPI: {kpi}")


def _pharma_dq_recommendation(kpi: str) -> str:
    return (
        f"Data-quality issue on '{kpi}': the detector could not assess this KPI "
        "(missing, corrupt, or insufficient data). Investigate the data source and "
        "instrumentation before drawing any conclusion; do NOT treat this as a process "
        "deviation or open a CAPA on the process."
    )


def _pharma_offline_recommend(kpis: list) -> list[dict]:
    return [
        {
            "kpi": kpi,
            "capa_frame": f"CAPA-{i:03d}",
            "text": (
                f"Open a CAPA review for '{kpi}': confirm the signal against source "
                "data, assess whether an ICH E6(R3) deviation threshold is crossed, "
                "and, if confirmed, document root cause and corrective action. "
                "Escalate to the responsible quality owner for sign-off."
            ),
        }
        for i, kpi in enumerate(kpis, start=1)
    ]


def _pharma_lineage_caveat(kpi: str) -> str:
    return (
        f"The clinical records behind '{kpi}' include site(s) whose data did not come from a "
        "verified system of record (scanned/manual case report forms rather than EDC entry). "
        "Treat this forecast as lower-trust: do not act on it without confirming the underlying "
        "site data with the responsible quality owner, and note the widened interval already "
        "published reflects this genealogy risk, not just statistical uncertainty."
    )


def make_pharma_domain(
    paper_site_ids: frozenset[int] = frozenset(),
    missing_source_site_ids: frozenset[int] = frozenset(),
) -> Domain:
    """Build the pharma Domain. The default (no degraded sites) is the historical PHARMA
    singleton, byte-identical to before this capability existed. Passing paper_site_ids /
    missing_source_site_ids tags those clinical sites with degraded lineage, for a demo
    scenario or a test, without touching any numeric generator behavior."""

    def _generate(start: date, days: int, seed: int) -> dict:
        # Same seeding as the CLI/tick: seed, seed+1, seed+2 for the three streams.
        return {
            "clinical": generate_clinical(
                start, days, make_rng(seed),
                paper_site_ids=paper_site_ids, missing_source_site_ids=missing_source_site_ids,
            ),
            "qc": generate_qc(start, days, make_rng(seed + 1)),
            "supply": generate_supply(start, days, make_rng(seed + 2)),
        }

    return Domain(
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
        hero_frame=_hero_frame,
        lineage_extract=_lineage_extract,
        offline_framing=(
            "These movements are assessed against GxP quality expectations; where a KPI is "
            "Warning/Critical the trend and forecast suggest a deviation trigger may apply and "
            "should be reviewed by the quality team."
        ),
        offline_recommend=_pharma_offline_recommend,
        dq_recommendation=_pharma_dq_recommendation,
        lineage_caveat=_pharma_lineage_caveat,
    )


PHARMA = make_pharma_domain()
