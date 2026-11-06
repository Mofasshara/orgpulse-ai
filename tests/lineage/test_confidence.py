import pandas as pd
import pytest

from orgpulse.lineage import lineage_confidence, LOW_TRUST_THRESHOLD, UNSAFE_LINEAGE


def test_all_system_of_record_gives_full_confidence_no_widening():
    dates = pd.date_range("2026-01-01", periods=10)
    tiers = ["system_of_record"] * 10
    r = lineage_confidence(dates, tiers)
    assert r.confidence == 1.0
    assert r.low_trust is False
    assert r.widen_factor == 1.0


def test_per_day_minimum_then_mean_across_days():
    # 5 days with one manual_scanned record alongside a system_of_record record (day weight 0.4),
    # 5 pure system_of_record days (day weight 1.0) -> mean == 0.7
    degraded_days = pd.date_range("2026-01-01", periods=5)
    clean_days = pd.date_range("2026-01-06", periods=5)
    dates = list(degraded_days) * 2 + list(clean_days)
    tiers = (["manual_scanned"] * 5 + ["system_of_record"] * 5) + ["system_of_record"] * 5
    r = lineage_confidence(dates, tiers)
    assert r.confidence == pytest.approx(0.7)
    assert r.low_trust is True          # 0.7 < LOW_TRUST_THRESHOLD (0.8)
    assert r.widen_factor == pytest.approx(1.3)   # 1.0 + 1.0*(1-0.7)


@pytest.mark.parametrize("dates,tiers", [
    (None, None),
    ([], []),
    (pd.date_range("2026-01-01", periods=3), ["system_of_record"]),       # length mismatch
    (pd.date_range("2026-01-01", periods=2), ["system_of_record", "typo_tier"]),  # unknown tier
    ([pd.NaT, "2026-01-02"], ["system_of_record", "system_of_record"]),   # unparseable date
])
def test_missing_or_corrupt_lineage_data_is_unsafe(dates, tiers):
    r = lineage_confidence(dates, tiers)
    assert r == UNSAFE_LINEAGE


def test_low_trust_threshold_is_honest_about_its_boundary():
    # confidence exactly at the threshold is NOT low_trust (< is strict)
    dates = pd.date_range("2026-01-01", periods=1)
    tiers = ["system_of_record"]
    assert lineage_confidence(dates, tiers).confidence >= LOW_TRUST_THRESHOLD
