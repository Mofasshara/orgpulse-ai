from datetime import date

import numpy as np
import pandas as pd
import pytest

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical, PROTOCOL_AMENDMENT_DAY


def _df(days=200, n_sites=8, seed=42):
    return generate_clinical(date(2026, 1, 1), days, make_rng(seed), n_sites=n_sites)


def test_schema_and_types():
    df = _df()
    assert list(df.columns) == [
        "date", "site_id", "daily_enrollment", "cumulative_enrollment",
        "screen_fail_rate", "dropout_rate", "source_tier",
    ]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["site_id"].nunique() == 8


def test_source_tier_defaults_to_system_of_record_for_all_sites():
    df = _df()
    assert (df["source_tier"] == "system_of_record").all()


def test_paper_and_missing_site_ids_tag_correctly():
    df = generate_clinical(
        date(2026, 1, 1), 60, make_rng(42), n_sites=8,
        paper_site_ids=frozenset({0}), missing_source_site_ids=frozenset({1}),
    )
    assert (df.loc[df.site_id == 0, "source_tier"] == "manual_scanned").all()
    assert (df.loc[df.site_id == 1, "source_tier"] == "missing").all()
    assert (df.loc[~df.site_id.isin([0, 1]), "source_tier"] == "system_of_record").all()


@pytest.mark.parametrize("paper,missing", [
    (frozenset({8}), frozenset()),          # out of range (n_sites=8, valid ids 0..7)
    (frozenset({-1}), frozenset()),         # negative
    (frozenset({0}), frozenset({0})),       # overlap
])
def test_rejects_bad_site_tier_config(paper, missing):
    with pytest.raises(ValueError):
        generate_clinical(date(2026, 1, 1), 30, make_rng(0), n_sites=8,
                           paper_site_ids=paper, missing_source_site_ids=missing)


def test_source_tier_config_does_not_change_any_numeric_column():
    plain = _df(seed=7)
    tagged = generate_clinical(
        date(2026, 1, 1), 200, make_rng(7), n_sites=8,
        paper_site_ids=frozenset({2}), missing_source_site_ids=frozenset({5}),
    )
    numeric_cols = ["date", "site_id", "daily_enrollment", "cumulative_enrollment",
                    "screen_fail_rate", "dropout_rate"]
    pd.testing.assert_frame_equal(plain[numeric_cols], tagged[numeric_cols])


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


def test_protocol_amendment_is_a_sustained_surge():
    df = _df(days=200)
    per = df.groupby("date")["daily_enrollment"].sum().sort_index().reset_index(drop=True)
    before = per.iloc[PROTOCOL_AMENDMENT_DAY - 30 : PROTOCOL_AMENDMENT_DAY].mean()
    after_window = per.iloc[PROTOCOL_AMENDMENT_DAY : PROTOCOL_AMENDMENT_DAY + 40]
    after = after_window.mean()
    assert after > before                          # raises enrollment (direction)
    assert after_window.max() < 4 * after          # sustained, not a 1-day cliff


def test_deterministic_given_seed():
    a = _df(seed=7)
    b = _df(seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_amendment_effect_is_real_across_seeds():
    # Mutation-resistant: the surge must show over MANY seeds, not one.
    wins = 0
    for seed in range(20):
        df = generate_clinical(date(2026, 1, 1), 200, make_rng(seed))
        per = df.groupby("date")["daily_enrollment"].sum().sort_index().reset_index(drop=True)
        before = per.iloc[90:120].mean()
        after = per.iloc[120:160].mean()
        wins += after > before
    assert wins >= 18                              # ≥90% of seeds show the surge


def test_holiday_dip_actually_reduces_enrollment():
    # Compare late-December enrollment with the rest of December, across seeds.
    # A working holiday dip makes 22–31 Dec markedly lower than 1–21 Dec.
    ratios = []
    for seed in range(20):
        df = generate_clinical(date(2026, 1, 1), 365, make_rng(seed))
        s = df.groupby("date")["daily_enrollment"].sum()
        early_dec = s[[d for d in s.index if d.month == 12 and d.day <= 21]].mean()
        late_dec = s[[d for d in s.index if d.month == 12 and d.day >= 22]].mean()
        ratios.append(late_dec / early_dec)
    assert np.mean(ratios) < 0.7                    # holiday window clearly lower


def test_generate_clinical_rejects_bad_args():
    with pytest.raises(ValueError):
        generate_clinical(date(2026, 1, 1), 0, make_rng(0))
    with pytest.raises(ValueError):
        generate_clinical(date(2026, 1, 1), 30, make_rng(0), n_sites=0)
