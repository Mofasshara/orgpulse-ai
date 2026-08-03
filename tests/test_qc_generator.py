from datetime import date

import numpy as np
import pandas as pd

from orgpulse.config import make_rng
from orgpulse.generators.qc import generate_qc


def _df(days=200, seed=42):
    return generate_qc(date(2026, 1, 1), days, make_rng(seed))


def test_schema_and_types():
    df = _df()
    assert list(df.columns) == [
        "date", "batches_tested", "batches_passed", "pass_rate",
        "oos_events", "calibration_offset",
    ]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert len(df) == 200


def test_pass_rate_bounded_and_consistent():
    df = _df()
    assert df["pass_rate"].between(0, 1).all()
    assert (df["batches_passed"] <= df["batches_tested"]).all()


def test_calibration_offset_drifts_upward():
    df = _df(days=300)
    early = df["calibration_offset"].head(50).mean()
    late = df["calibration_offset"].tail(50).mean()
    assert late > early                          # signed upward drift, not just magnitude
    assert df["calibration_offset"].iloc[-1] > 0


def test_oos_events_present_and_nonnegative():
    df = _df(days=365)
    assert (df["oos_events"] >= 0).all()
    assert df["oos_events"].sum() > 0


def test_deterministic_given_seed():
    pd.testing.assert_frame_equal(_df(seed=7), _df(seed=7))
