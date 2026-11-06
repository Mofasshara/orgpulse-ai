"""Data-lineage confidence: the single chokepoint turning per-record provenance tags into a
confidence signal consumed by forecast_node (interval widening) and recommendation_node
(governance downgrade). Missing, corrupt, or unparseable input is NOT a free pass to high
confidence: it is the unsafe default (confidence=0.0, low_trust=True), earned high confidence
only when every contributing record is affirmatively tagged and parseable.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class SourceTier(str, Enum):
    SYSTEM_OF_RECORD = "system_of_record"
    MANUAL_SCANNED = "manual_scanned"
    MISSING = "missing"


TIER_WEIGHT: dict[str, float] = {
    SourceTier.SYSTEM_OF_RECORD.value: 1.0,
    SourceTier.MANUAL_SCANNED.value: 0.4,
    SourceTier.MISSING.value: 0.0,
}

# A confidence below this reads as "low trust". Illustrative, not empirically calibrated
# against real clinical operations (see docs/CASE_STUDY.md).
LOW_TRUST_THRESHOLD: float = 0.8
# widen_factor ranges [1.0, 1.0 + MAX_EXTRA_WIDEN]; at confidence=0.0 the published interval's
# half-width is doubled on top of the existing statistical calibration.
MAX_EXTRA_WIDEN: float = 1.0


@dataclass(frozen=True)
class LineageResult:
    confidence: float
    tier_counts: dict[str, int]
    reason: str
    low_trust: bool
    widen_factor: float


UNSAFE_LINEAGE = LineageResult(
    confidence=0.0,
    tier_counts={},
    reason="lineage data unavailable or corrupt",
    low_trust=True,
    widen_factor=1.0 + MAX_EXTRA_WIDEN,
)


def lineage_confidence(dates, tiers) -> LineageResult:
    """Turn per-record (date, source_tier) pairs into one LineageResult.

    Missing, empty, mismatched-length, unparseable-date, or unrecognized-tier input is corrupt
    or absent lineage data and returns UNSAFE_LINEAGE verbatim, never a silently high score.
    Otherwise: each day's weight is the MINIMUM tier weight among that day's contributing
    records (one untrustworthy record taints the whole day, mirroring validated_clinical_daily's
    any-NaN-in-a-group rule in pipeline/nodes.py); overall confidence is the mean of all days'
    minimums.
    """
    if dates is None or tiers is None:
        return UNSAFE_LINEAGE
    try:
        d = pd.to_datetime(pd.Series(list(dates)))
        t = pd.Series(list(tiers))
    except (ValueError, TypeError):
        return UNSAFE_LINEAGE
    if len(d) != len(t) or len(d) == 0:
        return UNSAFE_LINEAGE
    if d.isna().any():
        return UNSAFE_LINEAGE
    unknown = set(t) - set(TIER_WEIGHT)
    if unknown:
        return UNSAFE_LINEAGE

    weights = t.map(TIER_WEIGHT)
    day_weight = weights.groupby(d.dt.normalize().to_numpy()).min()
    confidence = float(day_weight.mean())
    tier_counts = {tier: int((t == tier).sum()) for tier in TIER_WEIGHT}
    low_trust = confidence < LOW_TRUST_THRESHOLD
    widen_factor = 1.0 + MAX_EXTRA_WIDEN * (1.0 - confidence)
    pct_degraded = 100.0 * (1.0 - float((t == SourceTier.SYSTEM_OF_RECORD.value).mean()))
    reason = (
        f"{pct_degraded:.0f}% of contributing records are not from a verified system of record "
        f"(confidence={confidence:.2f})"
    )
    return LineageResult(confidence, tier_counts, reason, low_trust, widen_factor)
