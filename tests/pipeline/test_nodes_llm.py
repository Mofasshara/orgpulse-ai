from orgpulse.pipeline.llm import OfflineLLM
from orgpulse.pipeline.nodes import analyst_node, recommendation_node
from orgpulse.pipeline.state import KpiFinding, HUMAN_OVERSIGHT_DISCLAIMER


def _state():
    return {
        "kpi_findings": [
            KpiFinding(kpi="daily_enrollment", severity="CRITICAL", detail="high CUSUM", latest_value=73.0),
            KpiFinding(kpi="pass_rate", severity="SAFE", detail="stable", latest_value=0.98),
        ],
        "overall_severity": "CRITICAL",
        "forecast": {"daily_enrollment": {"h30": {"yhat": 60.0, "lower": 40.0, "upper": 80.0}}},
    }


def test_analyst_node_returns_text():
    out = analyst_node(_state(), OfflineLLM())
    assert isinstance(out["analysis"], str) and len(out["analysis"]) > 0


def test_recommendation_node_flags_and_gates():
    out = recommendation_node(_state(), OfflineLLM())
    assert out["alert"] is True                       # overall CRITICAL
    recs = out["recommendations"]
    assert recs and all(r.review_status == "PENDING_HUMAN_REVIEW" for r in recs)
    assert all(r.disclaimer == HUMAN_OVERSIGHT_DISCLAIMER for r in recs)
    # only the flagged (non-SAFE) KPI gets a recommendation
    assert {r.kpi for r in recs} == {"daily_enrollment"}


def test_data_quality_finding_gets_data_quality_rec_and_alerts():
    state = {
        "kpi_findings": [KpiFinding(kpi="pass_rate", severity="DATA_QUALITY", detail="non-finite", latest_value=None)],
        "overall_severity": "DATA_QUALITY",
        "forecast": {},
    }
    out = recommendation_node(state, OfflineLLM())
    assert out["alert"] is True
    recs = out["recommendations"]
    assert len(recs) == 1 and recs[0].capa_frame == "DATA-QUALITY"
    assert recs[0].review_status == "PENDING_HUMAN_REVIEW"
    assert "do NOT treat" in recs[0].text
