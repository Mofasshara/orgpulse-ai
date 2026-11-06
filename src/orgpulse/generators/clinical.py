from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from orgpulse.generators.base import date_series, is_holiday_dip
from orgpulse.lineage import SourceTier

PROTOCOL_AMENDMENT_DAY: int = 120
_AMENDMENT_BOOST: float = 1.6
_AMENDMENT_RAMP_OFFSET: int = 20
_AMENDMENT_RAMP_STEEPNESS: float = 0.15


def generate_clinical(
    start_date: date, days: int, rng: np.random.Generator, n_sites: int = 8,
    paper_site_ids: frozenset[int] = frozenset(),
    missing_source_site_ids: frozenset[int] = frozenset(),
) -> pd.DataFrame:
    if days < 1:
        raise ValueError("days must be >= 1")
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")
    overlap = paper_site_ids & missing_source_site_ids
    if overlap:
        raise ValueError(f"site id(s) {sorted(overlap)} cannot be both paper and missing-source")
    for label, ids in (("paper_site_ids", paper_site_ids), ("missing_source_site_ids", missing_source_site_ids)):
        bad = {i for i in ids if i < 0 or i >= n_sites}
        if bad:
            raise ValueError(f"{label} contains out-of-range site id(s) {sorted(bad)} for n_sites={n_sites}")
    dates = date_series(start_date, days)
    t = np.arange(days)
    rows = []
    for site_id in range(n_sites):
        if site_id in missing_source_site_ids:
            source_tier = SourceTier.MISSING.value
        elif site_id in paper_site_ids:
            source_tier = SourceTier.MANUAL_SCANNED.value
        else:
            source_tier = SourceTier.SYSTEM_OF_RECORD.value
        laggard = site_id == 0
        base_rate = rng.uniform(3.0, 5.0) * (0.35 if laggard else 1.0)
        activation = int(rng.integers(0, 30))
        ramp = 1.0 / (1.0 + np.exp(-0.2 * (t - (activation + 15))))
        amend = 1.0 + (_AMENDMENT_BOOST - 1.0) / (
            1.0 + np.exp(-_AMENDMENT_RAMP_STEEPNESS * (t - (PROTOCOL_AMENDMENT_DAY + _AMENDMENT_RAMP_OFFSET)))
        )
        amend = np.where(t >= PROTOCOL_AMENDMENT_DAY, amend, 1.0)
        cum = 0
        for i, d in enumerate(dates):
            rate = base_rate * ramp[i] * amend[i] * is_holiday_dip(d)
            daily_int = int(rng.poisson(max(rate, 0.0)))
            cum += daily_int
            rows.append(
                {
                    "date": d,
                    "site_id": site_id,
                    "daily_enrollment": daily_int,
                    "cumulative_enrollment": cum,
                    "screen_fail_rate": float(np.clip(rng.normal(0.18, 0.03), 0, 1)),
                    "dropout_rate": float(np.clip(rng.normal(0.08, 0.02), 0, 1)),
                    "source_tier": source_tier,
                }
            )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["site_id", "date"]).reset_index(drop=True)
