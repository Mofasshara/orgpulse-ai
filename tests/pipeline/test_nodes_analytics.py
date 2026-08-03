from datetime import date

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical
from orgpulse.generators.qc import generate_qc
from orgpulse.generators.supply import generate_supply
from orgpulse.pipeline.nodes import detect_node, forecast_node


def _state(days=260):
    return {
        "clinical": generate_clinical(date(2026, 1, 1), days, make_rng(42)),
        "qc": generate_qc(date(2026, 1, 1), days, make_rng(43)),
        "supply": generate_supply(date(2026, 1, 1), days, make_rng(44)),
    }


def test_detect_node_produces_findings_and_overall_severity():
    out = detect_node(_state())
    assert out["kpi_findings"] and out["overall_severity"] in {"SAFE", "WARNING", "CRITICAL", "DATA_QUALITY"}
    kpis = {f.kpi for f in out["kpi_findings"]}
    assert "daily_enrollment" in kpis            # hero present
    for f in out["kpi_findings"]:
        assert f.severity in {"SAFE", "WARNING", "CRITICAL", "DATA_QUALITY"}


def test_forecast_node_hero_horizons():
    out = forecast_node(_state())
    fc = out["forecast"]["daily_enrollment"]
    for h in ("h30", "h60", "h90"):
        assert fc[h]["lower"] <= fc[h]["yhat"] <= fc[h]["upper"]


def test_nodes_are_deterministic():
    a = detect_node(_state())["overall_severity"]
    b = detect_node(_state())["overall_severity"]
    assert a == b
