from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from orgpulse.generators.base import date_series, is_holiday_dip, logistic_curve

PROTOCOL_AMENDMENT_DAY: int = 120
_AMENDMENT_BOOST: float = 1.6
_AMENDMENT_RAMP_OFFSET: int = 20        # extra capacity centres ~20 days post-amendment
_AMENDMENT_RAMP_STEEPNESS: float = 0.15


def generate_clinical(
    start_date: date, days: int, rng: np.random.Generator, n_sites: int = 8
) -> pd.DataFrame:
    dates = date_series(start_date, days)
    rows = []
    for site_id in range(n_sites):
        laggard = site_id == 0
        ceiling = rng.uniform(400, 600) * (0.35 if laggard else 1.0)
        midpoint = rng.uniform(40, 80)
        steepness = rng.uniform(0.05, 0.12)
        activation = int(rng.integers(0, 30))

        base_curve = logistic_curve(days, ceiling, midpoint + activation, steepness)

        # Protocol amendment adds NEW addressable capacity that ramps in over
        # weeks AFTER day 120 (its own logistic, centred `_AMENDMENT_RAMP_OFFSET`
        # days after the amendment and normalised to 0 at day 120). This yields a
        # sustained multi-week surge, NOT a one-day cliff — a naive "switch to a
        # higher-ceiling curve" dumps the whole capacity gap into a single day
        # because the base curve is already saturated by day 120.
        extra_ceiling = ceiling * (_AMENDMENT_BOOST - 1.0)
        t = np.arange(days)
        ramp_mid = PROTOCOL_AMENDMENT_DAY + _AMENDMENT_RAMP_OFFSET
        s = 1.0 / (1.0 + np.exp(-_AMENDMENT_RAMP_STEEPNESS * (t - ramp_mid)))
        s0 = 1.0 / (1.0 + np.exp(-_AMENDMENT_RAMP_STEEPNESS * (PROTOCOL_AMENDMENT_DAY - ramp_mid)))
        extra_curve = np.where(
            t >= PROTOCOL_AMENDMENT_DAY,
            extra_ceiling * (s - s0) / (1.0 - s0),
            0.0,
        )
        cumulative_ideal = base_curve + extra_curve

        prev_cum = 0.0
        cum = 0
        for day_idx, d in enumerate(dates):
            if day_idx < activation:
                daily = 0.0
            else:
                inc = max(0.0, cumulative_ideal[day_idx] - prev_cum)
                prev_cum = cumulative_ideal[day_idx]
                daily = inc * is_holiday_dip(d)
                daily += rng.normal(0, 0.5)  # measurement noise
            daily_int = int(max(0, round(daily)))
            cum += daily_int
            rows.append(
                {
                    "date": d,
                    "site_id": site_id,
                    "daily_enrollment": daily_int,
                    "cumulative_enrollment": cum,
                    "screen_fail_rate": float(np.clip(rng.normal(0.18, 0.03), 0, 1)),
                    "dropout_rate": float(np.clip(rng.normal(0.08, 0.02), 0, 1)),
                }
            )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["site_id", "date"]).reset_index(drop=True)
