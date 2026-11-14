from datetime import date

from orgpulse.domains.pharma import PHARMA, make_pharma_domain
from orgpulse.pipeline.llm import make_llm
from orgpulse.pipeline.nodes import detect_node, forecast_node, recommendation_node
from orgpulse.pipeline.state import KpiFinding


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


def test_forecast_node_rejects_lineage_that_does_not_cover_the_forecast_window():
    # Codex r1 Critical: a domain's lineage_extract is untrusted (same class as extract/
    # hero_frame) and could return data from an unrelated window while the real history
    # contains a degraded site. A single clean record dated decades before the actual
    # forecast window must not be treated as evidence ABOUT that window.
    from dataclasses import replace
    import numpy as np
    degraded = make_pharma_domain(paper_site_ids=frozenset({0, 1, 2}))
    streams = _streams(degraded)

    def _stale_single_record(streams):
        return np.array(["1900-01-01"], dtype="datetime64[ns]"), ["system_of_record"]

    lying_domain = replace(degraded, lineage_extract=_stale_single_record)
    kfs = detect_node({"streams": streams}, lying_domain)["kpi_findings"]
    out = forecast_node({"streams": streams, "kpi_findings": kfs}, lying_domain)
    lineage = out["forecast"]["daily_enrollment"]["lineage"]
    assert lineage["confidence"] == 0.0
    assert lineage["low_trust"] is True


def test_forecast_node_rejects_lineage_missing_some_forecast_window_dates():
    # Partial coverage (most dates present, a chunk of the window missing) must also fail
    # unsafe, not just total mismatch -- the forecast used those missing dates' data too.
    from dataclasses import replace
    degraded = make_pharma_domain(paper_site_ids=frozenset({0, 1, 2}))
    streams = _streams(degraded)

    def _partial_coverage(streams):
        dates, tiers = degraded.lineage_extract(streams)
        # Drop every record from the first 30 days -- a real but incomplete lineage report.
        cutoff = sorted(set(dates))[30]
        keep = [i for i, d in enumerate(dates) if d >= cutoff]
        return dates[keep], [tiers[i] for i in keep]

    partial_domain = replace(degraded, lineage_extract=_partial_coverage)
    kfs = detect_node({"streams": streams}, partial_domain)["kpi_findings"]
    out = forecast_node({"streams": streams, "kpi_findings": kfs}, partial_domain)
    lineage = out["forecast"]["daily_enrollment"]["lineage"]
    assert lineage["confidence"] == 0.0
    assert lineage["low_trust"] is True


def test_forecast_node_accepts_lineage_that_fully_covers_the_forecast_window():
    # Full-coverage lineage (the normal production case) must NOT be rejected by the
    # coverage check -- confirms the check isn't over-strict.
    out = forecast_node(_state(PHARMA), PHARMA)
    lineage = out["forecast"]["daily_enrollment"]["lineage"]
    assert lineage["confidence"] == 1.0
    assert lineage["low_trust"] is False


def _calm_state_with_lineage(low_trust: bool):
    return {
        "kpi_findings": [KpiFinding(kpi="daily_enrollment", severity="SAFE", detail="d", latest_value=1.0)],
        "overall_severity": "SAFE",
        "forecast": {"daily_enrollment": {"lineage": {"low_trust": low_trust, "confidence": 0.3 if low_trust else 1.0}}},
    }


def test_low_trust_hero_forces_alert_and_adds_lineage_risk_recommendation():
    llm = make_llm(offline=True, domain=PHARMA)
    out = recommendation_node(_calm_state_with_lineage(True), llm, PHARMA)
    assert out["alert"] is True
    matches = [r for r in out["recommendations"] if r.capa_frame == "LINEAGE-RISK" and r.kpi == "daily_enrollment"]
    assert len(matches) == 1 and matches[0].text.strip()


def test_high_trust_hero_does_not_add_lineage_risk_recommendation():
    llm = make_llm(offline=True, domain=PHARMA)
    out = recommendation_node(_calm_state_with_lineage(False), llm, PHARMA)
    assert out["alert"] is False
    assert not any(r.capa_frame == "LINEAGE-RISK" for r in out["recommendations"])


def test_llm_recommendation_cannot_collide_with_the_reserved_lineage_risk_frame():
    # Codex r1 Minor: an LLM-authored (or offline-stub) recommendation for the hero's
    # statistical anomaly could itself claim capa_frame="LINEAGE-RISK", colliding with the
    # deterministic entry recommendation_node appends for low-trust lineage -- two ambiguous
    # duplicate (kpi, capa_frame) cards for the same finding.
    class _CollidingLLM:
        def recommend(self, system, prompt, kpis):
            return [{"kpi": kpis[0], "text": "an LLM-authored note",
                      "capa_frame": "LINEAGE-RISK"}]

    state = _calm_state_with_lineage(True)
    state["kpi_findings"] = [KpiFinding(kpi="daily_enrollment", severity="WARNING", detail="d", latest_value=1.0)]
    state["overall_severity"] = "WARNING"
    out = recommendation_node(state, _CollidingLLM(), PHARMA)
    matches = [r for r in out["recommendations"]
               if r.kpi == "daily_enrollment" and r.capa_frame == "LINEAGE-RISK"]
    assert len(matches) == 1


def test_lineage_risk_recommendation_coexists_with_an_anomaly_recommendation():
    llm = make_llm(offline=True, domain=PHARMA)
    state = _calm_state_with_lineage(True)
    state["kpi_findings"] = [KpiFinding(kpi="daily_enrollment", severity="WARNING", detail="d", latest_value=1.0)]
    state["overall_severity"] = "WARNING"
    out = recommendation_node(state, llm, PHARMA)
    kinds = [r.capa_frame for r in out["recommendations"] if r.kpi == "daily_enrollment"]
    assert "LINEAGE-RISK" in kinds and len(kinds) == 2   # both the anomaly rec and the lineage rec
