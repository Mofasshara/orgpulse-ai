from datetime import date

import numpy as np

from orgpulse.config import OrgPulseConfig, make_rng


def test_default_config_values():
    cfg = OrgPulseConfig()
    assert cfg.seed == 42
    assert cfg.start_date == date(2026, 1, 1)
    assert cfg.n_sites == 8


def test_config_is_frozen():
    cfg = OrgPulseConfig()
    try:
        cfg.seed = 7  # type: ignore[misc]
    except Exception as exc:  # frozen dataclass raises FrozenInstanceError
        assert "cannot assign" in str(exc).lower() or exc.__class__.__name__ == "FrozenInstanceError"
    else:
        raise AssertionError("expected OrgPulseConfig to be frozen")


def test_make_rng_is_deterministic():
    a = make_rng(42).random(5)
    b = make_rng(42).random(5)
    assert np.array_equal(a, b)
    assert isinstance(make_rng(42), np.random.Generator)
