from datetime import date

import numpy as np
import pandas as pd

from orgpulse.config import make_rng
from orgpulse.generators.supply import generate_supply, REORDER_POINT


def _df(days=365, seed=42):
    return generate_supply(date(2026, 1, 1), days, make_rng(seed))


def test_schema_and_types():
    df = _df()
    assert list(df.columns) == [
        "date", "inventory_units", "lead_time_days", "supplier_reliability",
    ]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_inventory_nonnegative_and_replenishes():
    df = _df()
    assert (df["inventory_units"] >= 0).all()
    # Sawtooth: inventory both draws down and jumps back up.
    diffs = np.diff(df["inventory_units"].to_numpy())
    assert (diffs < 0).any() and (diffs > 0).any()


def test_lead_time_has_floor_and_spikes():
    df = _df()
    assert (df["lead_time_days"] >= 3.0).all()
    assert df["lead_time_days"].max() > df["lead_time_days"].median() * 1.5


def test_reliability_bounded_and_seasonal():
    df = _df()
    assert df["supplier_reliability"].between(0, 1).all()
    assert df["supplier_reliability"].max() - df["supplier_reliability"].min() > 0.05


def test_deterministic_given_seed():
    pd.testing.assert_frame_equal(_df(seed=7), _df(seed=7))
