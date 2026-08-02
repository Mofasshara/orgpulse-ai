from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from orgpulse.generators.base import date_series

REORDER_POINT: int = 200
_REORDER_QTY: int = 600


def generate_supply(start_date: date, days: int, rng: np.random.Generator) -> pd.DataFrame:
    dates = date_series(start_date, days)
    inventory = 500
    rows = []
    for i, d in enumerate(dates):
        # Daily consumption draws inventory down; reorder restocks when low.
        consumption = int(rng.integers(15, 45))
        inventory = max(0, inventory - consumption)
        if inventory < REORDER_POINT:
            inventory += _REORDER_QTY

        # Lead time: base ~5 days, occasional spike.
        lead = 5.0 + rng.normal(0, 0.6)
        if rng.random() < 0.04:
            lead += rng.uniform(4, 10)  # disruption spike
        lead = max(3.0, lead)

        # Reliability: annual seasonality + noise.
        seasonal = 0.9 + 0.06 * np.sin(2 * np.pi * i / 365.0)
        reliability = float(np.clip(seasonal + rng.normal(0, 0.01), 0, 1))

        rows.append(
            {
                "date": d,
                "inventory_units": int(inventory),
                "lead_time_days": float(lead),
                "supplier_reliability": reliability,
            }
        )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
