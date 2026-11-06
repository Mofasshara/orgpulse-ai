import json

from orgpulse.agent.baseline_memory import BaselineMemory
from orgpulse.agent.tick import run_tick
from orgpulse.domains.pharma import PHARMA, make_pharma_domain


def _input_hash(runs_root, run_id_prefix):
    entries = [json.loads(l) for l in (runs_root / "latest" / "audit.jsonl").read_text().splitlines()]
    return next(e["payload"]["input_data_hash"] for e in entries if e["kind"] == "run")


def test_source_tier_changes_the_recorded_input_data_hash(tmp_path):
    clean_runs, degraded_runs = tmp_path / "clean", tmp_path / "degraded"
    run_tick(BaselineMemory(tmp_path / "m1.json"), 0, out_root=clean_runs,
              now="2026-08-04T07:00:00", offline=True, domain=PHARMA)
    degraded = make_pharma_domain(paper_site_ids=frozenset({0, 1, 2}))
    run_tick(BaselineMemory(tmp_path / "m2.json"), 0, out_root=degraded_runs,
              now="2026-08-04T07:00:00", offline=True, domain=degraded)
    assert _input_hash(clean_runs, "tick-0") != _input_hash(degraded_runs, "tick-0")


def test_degraded_lineage_tick_produces_a_low_trust_alerting_run(tmp_path):
    degraded = make_pharma_domain(paper_site_ids=frozenset({0, 1, 2}))
    findings = run_tick(BaselineMemory(tmp_path / "m.json"), 0, out_root=tmp_path / "runs",
                          now="2026-08-04T07:00:00", offline=True, domain=degraded)
    assert findings.alert is True
    assert any(r.capa_frame == "LINEAGE-RISK" for r in findings.recommendations)
    assert findings.forecast["daily_enrollment"]["lineage"]["low_trust"] is True
