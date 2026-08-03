from __future__ import annotations

import numpy as np
import pandas as pd

from orgpulse.forecast.prophet_forecaster import forecast_kpi
from orgpulse.forecast.types import CalibrationResult


def interval_calibration(
    series: pd.DataFrame, horizon: int = 30, n_folds: int = 4, nominal: float = 0.9, min_train: int = 150
) -> CalibrationResult:
    n = len(series)
    last_cutoff = n - horizon
    cutoffs = np.linspace(min_train, last_cutoff, n_folds).astype(int)
    actual_all, yhat_all, lower_all, upper_all = [], [], [], []
    for c in cutoffs:
        fc = forecast_kpi(series.iloc[:c], kpi="cal", horizons=(horizon,), interval_width=nominal)
        f = fc.frame
        actual_all.append(series["y"].to_numpy()[c : c + horizon])
        yhat_all.append(f["yhat"].to_numpy()[:horizon])
        lower_all.append(f["yhat_lower"].to_numpy()[:horizon])
        upper_all.append(f["yhat_upper"].to_numpy()[:horizon])
    actual = np.concatenate(actual_all)
    yhat = np.concatenate(yhat_all)
    lower = np.concatenate(lower_all)
    upper = np.concatenate(upper_all)

    raw_cov = float(np.mean((actual >= lower) & (actual <= upper)))
    half = np.maximum((upper - lower) / 2.0, 1e-9)
    ratio = np.abs(actual - yhat) / half
    scale = float(max(1.0, np.quantile(ratio, nominal)))
    adj_lower = yhat - scale * (yhat - lower)
    adj_upper = yhat + scale * (upper - yhat)
    adj_cov = float(np.mean((actual >= adj_lower) & (actual <= adj_upper)))
    return CalibrationResult(
        nominal=nominal, raw_coverage=raw_cov, scale_factor=scale,
        adjusted_coverage=adj_cov, n_points=int(actual.size),
    )
