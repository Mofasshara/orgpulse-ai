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


def test_all_nan_clinical_is_data_quality_not_safe_zero():
    """Fail-open regression: an all-NaN enrollment stream previously aggregated to 0.0 and
    passed as SAFE. It must now surface DATA_QUALITY (never SAFE) with no numeric forecast."""
    import numpy as np
    import pandas as pd

    clinical = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=200),
        "daily_enrollment": np.nan,
    })
    state = {"clinical": clinical, "qc": _state(200)["qc"], "supply": _state(200)["supply"]}

    det = detect_node(state)
    hero = next(f for f in det["kpi_findings"] if f.kpi == "daily_enrollment")
    assert hero.severity == "DATA_QUALITY" and hero.latest_value is None
    assert det["overall_severity"] == "DATA_QUALITY"
    # no numeric forecast published for an unusable hero series
    assert forecast_node(state)["forecast"].get("daily_enrollment") is None


def test_partial_nan_clinical_is_data_quality_not_masked_sum():
    """Fail-open regression (#1): one missing SITE on one date previously left a valid-looking
    partial sum that passed SAFE. Any NaN in a date's rows must make that day's total NaN."""
    import numpy as np
    from datetime import date
    from orgpulse.generators.clinical import generate_clinical

    clinical = generate_clinical(date(2026, 1, 1), 200, make_rng(42))
    d0 = clinical["date"].iloc[0]
    clinical.loc[(clinical.date == d0) & (clinical.site_id == 0), "daily_enrollment"] = np.nan
    state = {"clinical": clinical, "qc": _state(200)["qc"], "supply": _state(200)["supply"]}

    hero = next(f for f in detect_node(state)["kpi_findings"] if f.kpi == "daily_enrollment")
    assert hero.severity == "DATA_QUALITY"
    assert forecast_node(state)["forecast"].get("daily_enrollment") is None


def test_insufficient_kpi_is_data_quality_not_silently_omitted():
    """Fail-open regression (#2): a KPI with <120 points was returned as None and dropped, so
    an insufficient stream left overall SAFE. It must now be a DATA_QUALITY finding that dominates."""
    good = _state(200)
    state = {"clinical": good["clinical"], "qc": good["qc"], "supply": good["supply"].iloc[:119]}
    det = detect_node(state)
    sev = {f.kpi: f.severity for f in det["kpi_findings"]}
    assert sev.get("lead_time_days") == "DATA_QUALITY"     # present, not omitted
    assert set(sev) == {"daily_enrollment", "pass_rate", "calibration_offset", "lead_time_days"}
    assert det["overall_severity"] == "DATA_QUALITY"


def test_impossible_domain_values_are_data_quality_not_safe():
    """Semantic gap (#1): a consistently IMPOSSIBLE value (pass_rate>1, negative enrollment,
    negative lead time) is statistically stable and previously read SAFE. It must be DATA_QUALITY."""
    good = _state(200)

    qc_bad = good["qc"].copy(); qc_bad["pass_rate"] = 1.5
    sev = {f.kpi: f.severity for f in detect_node({"qc": qc_bad, "supply": good["supply"]})["kpi_findings"]}
    assert sev["pass_rate"] == "DATA_QUALITY"

    sup_bad = good["supply"].copy(); sup_bad["lead_time_days"] = -5.0
    sev = {f.kpi: f.severity for f in detect_node({"qc": good["qc"], "supply": sup_bad})["kpi_findings"]}
    assert sev["lead_time_days"] == "DATA_QUALITY"


def test_duplicate_clinical_records_are_data_quality():
    """#1: duplicate (date, site_id) rows double-count the daily total — reject as unusable."""
    import pandas as pd
    good = _state(200)
    dup = pd.concat([good["clinical"], good["clinical"].iloc[[0]]], ignore_index=True)
    det = detect_node({"clinical": dup, "qc": good["qc"], "supply": good["supply"]})
    hero = next(f for f in det["kpi_findings"] if f.kpi == "daily_enrollment")
    assert hero.severity == "DATA_QUALITY"


def test_recommendation_node_survives_malformed_llm_output():
    """Self-review: the online LLM path isn't runtime-schema-guaranteed. A malformed recommend()
    response must not crash the governed pipeline — malformed items are skipped, the alert still
    fires on overall_severity, and valid items still come through."""
    from orgpulse.pipeline.nodes import recommendation_node

    class BadLLM:
        def analyze(self, system, prompt): return "x"
        def recommend(self, system, prompt, kpis):
            return [{"kpi": "daily_enrollment"},              # missing text/capa_frame
                    "not-a-dict",
                    {"kpi": "pass_rate", "text": "t", "capa_frame": "CAPA-001"}]  # valid

    state = {"kpi_findings": [], "overall_severity": "CRITICAL", "forecast": {}}
    out = recommendation_node(state, BadLLM())              # must not raise
    assert out["alert"] is True
    assert [r.kpi for r in out["recommendations"]] == ["pass_rate"]  # only the valid one


def test_forecast_uses_the_same_gate_as_detection():
    """#1 (r7): forecasting must NOT publish a numeric forecast for a hero series detection rules
    DATA_QUALITY — per-site negative (masked by the sum), NaT date, non-integer count, duplicate."""
    import numpy as np
    import pandas as pd
    from datetime import date
    from orgpulse.generators.clinical import generate_clinical

    base = generate_clinical(date(2026, 1, 1), 200, make_rng(42))
    cases = {}
    c = base.copy(); c["daily_enrollment"] = 1; c.loc[c.site_id == 0, "daily_enrollment"] = -1
    cases["per_site_negative"] = c                                   # aggregate stays positive
    c = base.copy(); c.loc[c.index[0], "date"] = pd.NaT
    cases["nat_date"] = c
    c = base.copy(); c["daily_enrollment"] = 0.5
    cases["fractional_count"] = c
    cases["duplicate"] = pd.concat([base, base.iloc[[0]]], ignore_index=True)

    for name, clin in cases.items():
        hero = next(f for f in detect_node({"clinical": clin})["kpi_findings"] if f.kpi == "daily_enrollment")
        assert hero.severity == "DATA_QUALITY", name
        assert forecast_node({"clinical": clin})["forecast"].get("daily_enrollment") is None, name


def test_detect_node_survives_malformed_inputs_as_data_quality():
    """Schema-corrupt / 2-D KPI inputs must become DATA_QUALITY findings, not crash."""
    import numpy as np
    import pandas as pd

    good = _state(200)
    # QC column of 2-element arrays (ragged/2-D) + a supply frame missing its column
    bad_qc = pd.DataFrame({
        "pass_rate": list(np.ones((200, 2))),
        "calibration_offset": np.ones(200),
    })
    state = {"clinical": good["clinical"], "qc": bad_qc, "supply": pd.DataFrame({"wrong_col": np.ones(200)})}

    det = detect_node(state)                       # must not raise
    sev = {f.kpi: f.severity for f in det["kpi_findings"]}
    assert sev["pass_rate"] == "DATA_QUALITY"      # 2-D column
    assert sev["lead_time_days"] == "DATA_QUALITY" # missing column
    assert det["overall_severity"] == "DATA_QUALITY"
