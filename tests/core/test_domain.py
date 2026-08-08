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


def test_rejects_non_callable_offline_recommend():
    with pytest.raises(ValueError):
        Domain(name="t", generate=lambda **k: {}, extract=lambda s, n: None,
               kpi_specs=(KpiSpec("x", is_hero=True),), analyst_system="a", recommend_system="r",
               offline_recommend=None)
