from __future__ import annotations

import numpy as np
import pandas as pd

from orgpulse.forecast.prophet_forecaster import forecast_kpi
from orgpulse.forecast.types import BacktestResult


def rolling_origin_backtest(
    series: pd.DataFrame, horizon: int = 30, n_folds: int = 4, min_train: int = 150, interval_width: float = 0.9
) -> BacktestResult:
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if min_train < 1:
        raise ValueError(f"min_train must be >= 1, got {min_train}")
    if not 0.0 < interval_width < 1.0:
        raise ValueError(f"interval_width must be in (0, 1), got {interval_width}")
    n = len(series)
    last_cutoff = n - horizon
    if last_cutoff <= min_train:
        raise ValueError("series too short for the requested horizon/min_train")
    cutoffs = np.linspace(min_train, last_cutoff, n_folds).astype(int)
    mapes, mases = [], []
    for c in cutoffs:
        train = series.iloc[:c]
        fc = forecast_kpi(train, kpi="backtest", horizons=(horizon,), interval_width=interval_width)
        pred = fc.frame["yhat"].to_numpy()[:horizon]
        actual = series["y"].to_numpy()[c : c + horizon]
        scale = float(max(np.mean(np.abs(np.diff(train["y"].to_numpy()))), 1e-9))
        mapes.append(float(np.mean(np.abs((actual - pred) / np.maximum(actual, 1.0))) * 100))
        mases.append(float(np.mean(np.abs(actual - pred)) / scale))
    return BacktestResult(
        horizon=horizon, n_folds=n_folds,
        mape=float(np.mean(mapes)), mase=float(np.mean(mases)),
        per_fold_mape=tuple(mapes), per_fold_mase=tuple(mases),
    )
