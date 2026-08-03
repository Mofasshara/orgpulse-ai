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
