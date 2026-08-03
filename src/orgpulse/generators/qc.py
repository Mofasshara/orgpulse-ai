from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from orgpulse.generators.base import date_series


def generate_qc(start_date: date, days: int, rng: np.random.Generator) -> pd.DataFrame:
    if days < 1:
        raise ValueError("days must be >= 1")
    dates = date_series(start_date, days)
    # Calibration offset: random walk with small positive drift (slow de-calibration).
    drift = 0.004
    steps = rng.normal(drift, 0.01, size=days)
    calibration_offset = np.cumsum(steps)

    rows = []
    for i, d in enumerate(dates):
        batches_tested = int(rng.integers(8, 20))
        # Pass rate degrades slightly as calibration drifts.
        base_pass = 0.985 - min(0.15, abs(calibration_offset[i]) * 0.05)
        batches_passed = int(rng.binomial(batches_tested, np.clip(base_pass, 0, 1)))
        pass_rate = batches_passed / batches_tested
        # drift-dependent OOS incidence: Poisson rate rises with |calibration_offset| (not an explicit burst/cluster process)
        oos_lambda = 0.05 + abs(calibration_offset[i]) * 0.1
        oos_events = int(rng.poisson(oos_lambda))
        rows.append(
            {
                "date": d,
                "batches_tested": batches_tested,
                "batches_passed": batches_passed,
                "pass_rate": float(pass_rate),
                "oos_events": oos_events,
                "calibration_offset": float(calibration_offset[i]),
            }
        )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
