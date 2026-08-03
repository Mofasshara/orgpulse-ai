from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Forecast:
    kpi: str
    frame: pd.DataFrame
    horizons: tuple[int, ...]
    interval_width: float

    def at_horizon(self, h: int) -> dict:
        row = self.frame.iloc[h - 1]
        return {
            "ds": row["ds"],
            "yhat": float(row["yhat"]),
            "yhat_lower": float(row["yhat_lower"]),
            "yhat_upper": float(row["yhat_upper"]),
        }
