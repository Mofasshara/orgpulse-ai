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


def test_forecast_node_applies_calibration_to_published_bounds():
    """The published interval must be the CALIBRATED (widened) one, not raw Prophet.
    Regression for the 'calibrated 90%' overclaim: calibration was diagnosed but the
    node copied raw yhat_lower/yhat_upper."""
    from orgpulse.forecast.prophet_forecaster import forecast_kpi, series_from_generator

    state = _state()
    out = forecast_node(state)
    fc = out["forecast"]["daily_enrollment"]

    cal = fc["calibration"]
    assert cal["applied"] is True and cal["in_sample"] is True
    assert cal["scale_factor"] > 1.0                      # raw intervals were overconfident
    assert cal["raw_coverage"] is not None and cal["raw_coverage"] < 0.9

    # Published half-widths must exceed the raw Prophet half-widths by ~scale.
    raw = forecast_kpi(series_from_generator(state["clinical"], "date", "daily_enrollment"),
                       "daily_enrollment", horizons=(30, 60, 90))
    for h in (30, 60, 90):
        r = raw.at_horizon(h)
        published = fc[f"h{h}"]
        raw_half = float(r["yhat_upper"]) - float(r["yhat"])
        pub_half = published["upper"] - published["yhat"]
        assert pub_half > raw_half                         # genuinely widened


def test_nodes_are_deterministic():
    a = detect_node(_state())["overall_severity"]
    b = detect_node(_state())["overall_severity"]
    assert a == b


def test_detect_node_data_quality_dominates_and_alerts():
    # Regression: a partial data failure must make the whole run DATA_QUALITY (it
    # dominates even a CRITICAL clinical stream) and trigger the alert. Previously a
    # bare numeric max over Severity (DATA_QUALITY=-1) collapsed it to a lower severity.
    import numpy as np
    from orgpulse.pipeline.nodes import recommendation_node
    from orgpulse.pipeline.llm import OfflineLLM

    st = _state(260)
    qc = st["qc"].copy()
    qc.loc[qc.index[-1], "pass_rate"] = np.nan     # non-finite in the recent window
    st["qc"] = qc

    out = detect_node(st)
    assert out["overall_severity"] == "DATA_QUALITY"
    rec = recommendation_node(
        {"kpi_findings": out["kpi_findings"], "overall_severity": out["overall_severity"], "forecast": {}},
        OfflineLLM(),
    )
    assert rec["alert"] is True
