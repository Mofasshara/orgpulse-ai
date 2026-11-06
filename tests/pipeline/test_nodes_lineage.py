from datetime import date

from orgpulse.domains.pharma import PHARMA, make_pharma_domain
from orgpulse.pipeline.nodes import detect_node, forecast_node


def _streams(domain, days=260, seed=42):
    return domain.generate(date(2026, 1, 1), days, seed)


def _state(domain, days=260, seed=42):
    streams = _streams(domain, days, seed)
    kfs = detect_node({"streams": streams}, domain)["kpi_findings"]
    return {"streams": streams, "kpi_findings": kfs}


def test_forecast_node_publishes_full_confidence_lineage_for_clean_pharma():
    out = forecast_node(_state(PHARMA), PHARMA)
    lineage = out["forecast"]["daily_enrollment"]["lineage"]
    assert lineage["confidence"] == 1.0
    assert lineage["low_trust"] is False
    assert lineage["widen_factor"] == 1.0


def test_forecast_node_widens_interval_when_lineage_confidence_is_low():
    clean_fc = forecast_node(_state(PHARMA), PHARMA)["forecast"]["daily_enrollment"]
    degraded = make_pharma_domain(paper_site_ids=frozenset({0, 1, 2}))
    degraded_fc = forecast_node(_state(degraded), degraded)["forecast"]["daily_enrollment"]
    assert degraded_fc["lineage"]["low_trust"] is True
    clean_width = clean_fc["h30"]["upper"] - clean_fc["h30"]["lower"]
    degraded_width = degraded_fc["h30"]["upper"] - degraded_fc["h30"]["lower"]
    assert degraded_width > clean_width   # direction, not a brittle exact magnitude


def test_forecast_node_omits_lineage_block_when_domain_has_no_lineage_extract():
    from dataclasses import replace
    no_lineage_domain = replace(PHARMA, lineage_extract=None)
    out = forecast_node(_state(no_lineage_domain), no_lineage_domain)
    assert "lineage" not in out["forecast"]["daily_enrollment"]


def test_forecast_node_falls_back_to_unsafe_lineage_when_lineage_extract_raises():
    from dataclasses import replace
    def _boom(streams):
        raise RuntimeError("boom")
    broken = replace(PHARMA, lineage_extract=_boom)
    out = forecast_node(_state(broken), broken)
    lineage = out["forecast"]["daily_enrollment"]["lineage"]
    assert lineage["confidence"] == 0.0
    assert lineage["low_trust"] is True


def test_forecast_node_publishes_no_lineage_key_when_hero_is_data_quality():
    # A hero series too short to assess (< MIN_ASSESSMENT_DAYS) is DATA_QUALITY; no forecast,
    # therefore no lineage claim either.
    streams = _streams(PHARMA, days=10)
    kfs = detect_node({"streams": streams}, PHARMA)["kpi_findings"]
    out = forecast_node({"streams": streams, "kpi_findings": kfs}, PHARMA)
    assert out["forecast"] == {}


def test_forecast_node_falls_back_to_unsafe_lineage_when_tiers_are_unrecognized():
    # Integration-level twin of the module-level unknown-tier test in Task 2: a domain whose
    # lineage_extract hands back a tier value outside the three known ones must still reach
    # forecast_node as the unsafe default, not raise and not silently pass through.
    from dataclasses import replace
    streams = _streams(PHARMA)
    def _bad_tiers(streams):
        dates, _ = PHARMA.lineage_extract(streams)
        return dates, ["not_a_real_tier"] * len(dates)
    bad_domain = replace(PHARMA, lineage_extract=_bad_tiers)
    kfs = detect_node({"streams": streams}, bad_domain)["kpi_findings"]
    out = forecast_node({"streams": streams, "kpi_findings": kfs}, bad_domain)
    lineage = out["forecast"]["daily_enrollment"]["lineage"]
    assert lineage["confidence"] == 0.0
    assert lineage["low_trust"] is True
