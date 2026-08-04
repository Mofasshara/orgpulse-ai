import json
from datetime import date

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical
from orgpulse.generators.qc import generate_qc
from orgpulse.generators.supply import generate_supply
from orgpulse.pipeline.graph import run_pipeline, render_report
from orgpulse.pipeline.state import Findings, HUMAN_OVERSIGHT_DISCLAIMER


def _streams(days=260):
    return (
        generate_clinical(date(2026, 1, 1), days, make_rng(42)),
        generate_qc(date(2026, 1, 1), days, make_rng(43)),
        generate_supply(date(2026, 1, 1), days, make_rng(44)),
    )


def test_run_pipeline_offline_end_to_end():
    c, q, s = _streams()
    f = run_pipeline(c, q, s, offline=True, run_id="run-1", now="2026-08-04T07:00:00")
    assert isinstance(f, Findings)
    assert f.run_id == "run-1" and f.overall_severity in {"SAFE", "WARNING", "CRITICAL", "DATA_QUALITY"}
    assert f.kpi_findings and f.recommendations and f.analysis
    # JSON-safe, human-gated:
    json.dumps(f.to_json_dict(), allow_nan=False)
    assert all(r.review_status == "PENDING_HUMAN_REVIEW" for r in f.recommendations)


def test_run_pipeline_is_deterministic_offline():
    c, q, s = _streams()
    a = run_pipeline(c, q, s, offline=True, run_id="r", now="2026-08-04T07:00:00").to_json_dict()
    b = run_pipeline(c, q, s, offline=True, run_id="r", now="2026-08-04T07:00:00").to_json_dict()
    assert a == b


def test_render_report_contains_disclaimer_and_alert():
    c, q, s = _streams()
    f = run_pipeline(c, q, s, offline=True, run_id="r", now="2026-08-04T07:00:00")
    md = render_report(f)
    assert "OrgPulse" in md and "PENDING_HUMAN_REVIEW" in md
    assert HUMAN_OVERSIGHT_DISCLAIMER.split(" — ")[0] in md
