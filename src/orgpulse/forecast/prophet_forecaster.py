from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from orgpulse.forecast.types import Forecast


def _silence_prophet_logs() -> None:
    # cmdstanpy attaches its own stderr handler on import and ignores setLevel;
    # clearing the handlers + NullHandler is what actually suppresses the
    # "Chain [1] start processing" INFO spam (verified). Must run AFTER Prophet
    # is imported (it is imported lazily inside forecast_kpi).
    lg = logging.getLogger("cmdstanpy")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    lg.setLevel(logging.WARNING)


def series_from_generator(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    s = df.groupby(date_col)[value_col].sum().sort_index()
    return pd.DataFrame({"ds": pd.to_datetime(s.index), "y": s.to_numpy(dtype=float)}).reset_index(drop=True)


def forecast_kpi(history: pd.DataFrame, kpi: str, horizons=(30, 60, 90), interval_width: float = 0.9) -> Forecast:
    if history is None or len(history) < 2:
        raise ValueError("history needs at least 2 rows")
    y = np.asarray(history["y"], dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("history y contains non-finite values")
    from prophet import Prophet
    _silence_prophet_logs()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(
            interval_width=interval_width,
            weekly_seasonality=True,
            daily_seasonality=False,
            yearly_seasonality=False,
        )
        m.fit(history[["ds", "y"]])
        future = m.make_future_dataframe(periods=max(horizons))
        fc = m.predict(future).tail(max(horizons))
    frame = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)
    return Forecast(kpi=kpi, frame=frame, horizons=tuple(horizons), interval_width=interval_width)
