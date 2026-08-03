import json

from orgpulse.pipeline.state import (
    KpiFinding, Recommendation, Findings, HUMAN_OVERSIGHT_DISCLAIMER,
)


def test_recommendation_defaults_to_human_review():
    r = Recommendation(kpi="enroll", text="t", capa_frame="CAPA-000", disclaimer=HUMAN_OVERSIGHT_DISCLAIMER)
    assert r.review_status == "PENDING_HUMAN_REVIEW"
    assert "human" in r.disclaimer.lower()


def test_findings_json_serializable_no_nan():
    f = Findings(
        run_id="r1", generated_at="2026-08-04T07:00:00", overall_severity="WARNING", alert=True,
        kpi_findings=[KpiFinding(kpi="enroll", severity="WARNING", detail="d", latest_value=None)],
        forecast={"enroll": {"h30": {"yhat": 12.3, "lower": 8.0, "upper": 17.0}}},
        analysis="a",
        recommendations=[Recommendation(kpi="enroll", text="t", capa_frame="CAPA-001",
                                        disclaimer=HUMAN_OVERSIGHT_DISCLAIMER)],
    )
    d = f.to_json_dict()
    json.dumps(d, allow_nan=False)                       # must not raise
    assert d["recommendations"][0]["review_status"] == "PENDING_HUMAN_REVIEW"
    assert d["alert"] is True and d["kpi_findings"][0]["latest_value"] is None
