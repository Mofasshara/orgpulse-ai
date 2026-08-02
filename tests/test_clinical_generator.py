from datetime import date

import numpy as np
import pandas as pd

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical, PROTOCOL_AMENDMENT_DAY


def _df(days=200, n_sites=8, seed=42):
    return generate_clinical(date(2026, 1, 1), days, make_rng(seed), n_sites=n_sites)


def test_schema_and_types():
    df = _df()
    assert list(df.columns) == [
        "date", "site_id", "daily_enrollment", "cumulative_enrollment",
        "screen_fail_rate", "dropout_rate",
    ]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["site_id"].nunique() == 8


def test_cumulative_enrollment_non_decreasing_per_site():
    df = _df()
    for _, g in df.groupby("site_id"):
        g = g.sort_values("date")
        assert np.all(np.diff(g["cumulative_enrollment"].to_numpy()) >= 0)


def test_rates_are_bounded():
    df = _df()
    assert df["screen_fail_rate"].between(0, 1).all()
    assert df["dropout_rate"].between(0, 1).all()
    assert (df["daily_enrollment"] >= 0).all()


def test_site_zero_is_the_laggard():
    df = _df()
    totals = df.groupby("site_id")["daily_enrollment"].sum()
    assert totals[0] < totals.median()


def test_protocol_amendment_raises_enrollment():
    # Compare a 30-day window before vs after the amendment, summed across sites.
    df = _df(days=200)
    per_day = df.groupby("date")["daily_enrollment"].sum().sort_index()
    amend_date = per_day.index[PROTOCOL_AMENDMENT_DAY]
    before = per_day[per_day.index < amend_date].tail(30).mean()
    after = per_day[per_day.index >= amend_date].head(30).mean()
    assert after > before


def test_deterministic_given_seed():
    a = _df(seed=7)
    b = _df(seed=7)
    pd.testing.assert_frame_equal(a, b)
