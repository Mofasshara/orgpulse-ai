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


@pytest.mark.parametrize("bad_name", [None, 123, "", "   "])
def test_rejects_invalid_kpi_name(bad_name):
    """A KPI name is a dict key, a report label (name.replace(...)) and a rec identifier; a
    non-string or empty name would crash rendering or write an unusable governed rec, so it must
    fail at CONSTRUCTION. Mutation: drop the non-empty-string name check."""
    with pytest.raises(ValueError):
        _domain([KpiSpec(bad_name, (0.0, None), is_hero=True)])


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
    {"warn": 10.0, "critical": 10.0, "direction": "below"},             # equal -> WARNING unreachable (below)
    {"warn": 80.0, "critical": 80.0, "direction": "above"},             # equal -> WARNING unreachable (above)
    {"warn": 11.5, "critical": 10.5, "junk": 99},                       # unknown key (typo/garbage)
    None,                                                                # THRESHOLD requires a dict
    [],                                                                  # non-dict (falsy)
])
def test_rejects_unsafe_threshold_config_at_construction(bad_th):
    """A THRESHOLD KPI with a non-finite, directionally-inverted, EQUAL-cutoff (unreachable
    WARNING), unknown-key, or non-dict config must fail at CONSTRUCTION — not crash a run, silently
    return a wrong/unreachable severity, or mask a typo. Mutation-constrains the __post_init__
    THRESHOLD finiteness + strict-ordering + unknown-key checks."""
    with pytest.raises(ValueError):
        _domain([KpiSpec("h", (0.0, 1.0), is_hero=True),
                 KpiSpec("x", (0.0, None), DetectionMode.THRESHOLD, thresholds=bad_th)])


@pytest.mark.parametrize("bad_th", [
    {"warn_days": "bad", "critical_days": 3},        # non-numeric -> would crash float() at run time
    {"warn_days": float("nan"), "critical_days": 3}, # non-finite
    {"warn_days": 3, "critical_days": 14},           # inverted: critical_days > warn_days
    {"warn_days": -1, "critical_days": 3},           # negative days-remaining threshold
    {"warn_days": 3, "critical_days": 3},            # equal -> WARNING unreachable
    {"warn_day": 1, "critical_days": 0},             # unknown key (typo 'warn_day') silently masks intent
    {"warn_days": 14, "critical_days": 3, "junk": 99},  # arbitrary extra key
    [],                                              # falsy non-dict -> must NOT become defaults
    "",                                              # falsy non-dict
    False,                                           # falsy non-dict
])
def test_rejects_unsafe_datemath_config_at_construction(bad_th):
    """A DATEMATH KPI with a non-numeric/non-finite/negative/inverted/equal threshold, an unknown
    key, or a falsy non-dict config must fail at CONSTRUCTION. The string `warn_days` is the exact
    case Codex reproduced crashing run_pipeline/run_tick; the falsy non-dicts must NOT silently
    fall back to the 14/3 defaults. Mutation-constrains the __post_init__ DATEMATH check."""
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


@pytest.mark.parametrize("field", ["analyst_system", "recommend_system", "offline_framing"])
@pytest.mark.parametrize("bad", [None, 123, ["x"]])
def test_rejects_non_string_framing_fields(field, bad):
    """The framing fields are consumed as strings (online LLM system prompts / the offline stub's
    sentence); a non-string would crash the online call. Reject at construction."""
    kwargs = dict(name="t", generate=lambda **k: {}, extract=lambda s, n: None,
                  kpi_specs=(KpiSpec("x", is_hero=True),), analyst_system="a", recommend_system="r")
    kwargs[field] = bad
    with pytest.raises(ValueError):
        Domain(**kwargs)
