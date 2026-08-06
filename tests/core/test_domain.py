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
