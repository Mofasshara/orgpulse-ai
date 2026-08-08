import pytest

from orgpulse.core.domain import DetectionMode, Domain, KpiSpec


def _domain(specs):
    return Domain(
        name="t", generate=lambda **k: {}, extract=lambda s, n: None,
        kpi_specs=tuple(specs), analyst_system="a", recommend_system="r",
    )


def test_hero_bounds_and_spec():
    d = _domain([KpiSpec("x", (0.0, 1.0), is_hero=True), KpiSpec("y", (0.0, None))])
    assert d.hero() == "x"
    assert d.bounds() == {"x": (0.0, 1.0), "y": (0.0, None)}
    assert d.spec("y").detection_mode is DetectionMode.TIMESERIES


def test_requires_exactly_one_hero():
    with pytest.raises(ValueError):
        _domain([KpiSpec("x", (0.0, 1.0)), KpiSpec("y", (0.0, 1.0))])          # zero heroes
    with pytest.raises(ValueError):
        _domain([KpiSpec("x", is_hero=True), KpiSpec("y", is_hero=True)])      # two heroes


def test_rejects_duplicate_kpi_names():
    with pytest.raises(ValueError):
        _domain([KpiSpec("x", is_hero=True), KpiSpec("x")])


@pytest.mark.parametrize("bad_bounds", [(float("nan"), 1.0), (0.0, float("inf")), (1.0, 0.0), (True, 1.0)])
def test_rejects_unsafe_bounds_at_construction(bad_bounds):
    """A self-consistent Domain must reject NaN/inf/inverted/bool bounds when built, not fail open
    at run time. Mutation-constrains the __post_init__ bounds validation."""
    with pytest.raises(ValueError):
        _domain([KpiSpec("x", bad_bounds, is_hero=True)])


def test_rejects_non_enum_detection_mode():
    with pytest.raises(ValueError):
        _domain([KpiSpec("x", detection_mode="timeseries", is_hero=True)])  # str, not DetectionMode


def test_threshold_mode_requires_valid_thresholds():
    with pytest.raises(ValueError):   # THRESHOLD KPI without thresholds
        _domain([KpiSpec("h", (0.0, 1.0), is_hero=True),
                 KpiSpec("x", (0.0, None), DetectionMode.THRESHOLD)])
    with pytest.raises(ValueError):   # missing 'critical'
        _domain([KpiSpec("h", (0.0, 1.0), is_hero=True),
                 KpiSpec("x", (0.0, None), DetectionMode.THRESHOLD, thresholds={"warn": 1.0})])
    with pytest.raises(ValueError):   # bad direction
        _domain([KpiSpec("h", (0.0, 1.0), is_hero=True),
                 KpiSpec("x", (0.0, None), DetectionMode.THRESHOLD,
                         thresholds={"warn": 1.0, "critical": 0.5, "direction": "sideways"})])


@pytest.mark.parametrize("bad_th", [
    {"warn": float("nan"), "critical": 10.5, "direction": "below"},     # non-finite warn
    {"warn": 11.5, "critical": float("inf"), "direction": "below"},     # non-finite critical
    {"warn": 10.0, "critical": 12.0, "direction": "below"},             # inverted: critical > warn (below)
    {"warn": 95.0, "critical": 80.0, "direction": "above"},             # inverted: critical < warn (above)
    {"warn": True, "critical": 10.5, "direction": "below"},             # bool is not a number
])
def test_rejects_unsafe_threshold_config_at_construction(bad_th):
    """A THRESHOLD KPI with a non-finite or directionally-inverted band must fail at CONSTRUCTION,
    not crash a run or silently return a wrong/unreachable severity. Mutation-constrains the
    __post_init__ THRESHOLD finiteness + ordering checks."""
    with pytest.raises(ValueError):
        _domain([KpiSpec("h", (0.0, 1.0), is_hero=True),
                 KpiSpec("x", (0.0, None), DetectionMode.THRESHOLD, thresholds=bad_th)])


@pytest.mark.parametrize("bad_th", [
    {"warn_days": "bad", "critical_days": 3},        # non-numeric -> would crash float() at run time
    {"warn_days": float("nan"), "critical_days": 3}, # non-finite
    {"warn_days": 3, "critical_days": 14},           # inverted: critical_days > warn_days
    {"warn_days": -1, "critical_days": 3},           # negative days-remaining threshold
])
def test_rejects_unsafe_datemath_config_at_construction(bad_th):
    """A DATEMATH KPI with a non-numeric/non-finite/negative/inverted threshold must fail at
    CONSTRUCTION. The string case is the exact one Codex reproduced crashing run_pipeline/run_tick
    with `could not convert string to float`. Mutation-constrains the __post_init__ DATEMATH check."""
    with pytest.raises(ValueError):
        _domain([KpiSpec("h", (0.0, 1.0), is_hero=True),
                 KpiSpec("d", (None, None), DetectionMode.DATEMATH, thresholds=bad_th)])


def test_accepts_valid_threshold_and_datemath_configs():
    """The tightened validation must not reject well-formed configs (guards against over-rejection)."""
    _domain([
        KpiSpec("h", (0.0, 1.0), is_hero=True),
        KpiSpec("floor", (0.0, None), DetectionMode.THRESHOLD,
                thresholds={"warn": 11.5, "critical": 10.5, "direction": "below"}),
        KpiSpec("ceiling", (0.0, None), DetectionMode.THRESHOLD,
                thresholds={"warn": 80.0, "critical": 95.0, "direction": "above"}),
        KpiSpec("deadline", (None, None), DetectionMode.DATEMATH,
                thresholds={"warn_days": 14, "critical_days": 3}),
        KpiSpec("deadline_default", (None, None), DetectionMode.DATEMATH),  # thresholds=None -> defaults
    ])


def test_rejects_non_callable_offline_recommend():
    with pytest.raises(ValueError):
        Domain(name="t", generate=lambda **k: {}, extract=lambda s, n: None,
               kpi_specs=(KpiSpec("x", is_hero=True),), analyst_system="a", recommend_system="r",
               offline_recommend=None)
