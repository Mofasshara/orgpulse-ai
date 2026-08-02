from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from orgpulse.generators.base import date_series, is_holiday_dip, logistic_curve

PROTOCOL_AMENDMENT_DAY: int = 120
_AMENDMENT_BOOST: float = 1.6


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

        # Amendment increases the ceiling (capacity), not just daily multiplier
        cumulative_ideal_before = logistic_curve(days, ceiling, midpoint + activation, steepness)
        boosted_ceiling = ceiling * _AMENDMENT_BOOST
        cumulative_ideal_after = logistic_curve(days, boosted_ceiling, midpoint + activation, steepness)

        prev_cum = 0.0
        cum = 0
        for day_idx, d in enumerate(dates):
            if day_idx < activation:
                daily = 0.0
            else:
                # Use appropriate curve based on amendment date
                if day_idx < PROTOCOL_AMENDMENT_DAY:
                    target_cum = cumulative_ideal_before[day_idx]
                else:
                    target_cum = cumulative_ideal_after[day_idx]

                inc = max(0.0, target_cum - prev_cum)
                prev_cum = target_cum
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
