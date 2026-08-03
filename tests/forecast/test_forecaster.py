from datetime import date

import numpy as np
import pandas as pd
import pytest

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical
from orgpulse.forecast.prophet_forecaster import forecast_kpi, series_from_generator
from orgpulse.forecast.types import Forecast


def _hist(days=200):
    df = generate_clinical(date(2026, 1, 1), days, make_rng(42))
    return series_from_generator(df, "date", "daily_enrollment")


def test_series_from_generator_shape():
    s = _hist()
    assert list(s.columns) == ["ds", "y"]
    assert len(s) == 200
    assert pd.api.types.is_datetime64_any_dtype(s["ds"])


def test_forecast_has_horizons_and_ordered_intervals():
    fc = forecast_kpi(_hist(), kpi="daily_enrollment", horizons=(30, 60, 90))
    assert isinstance(fc, Forecast)
    assert set(["ds", "yhat", "yhat_lower", "yhat_upper"]).issubset(fc.frame.columns)
    assert len(fc.frame) == 90                       # max horizon rows
    assert (fc.frame["yhat_lower"] <= fc.frame["yhat"]).all()
    assert (fc.frame["yhat"] <= fc.frame["yhat_upper"]).all()
    for h in (30, 60, 90):
        row = fc.at_horizon(h)
        assert row["yhat_lower"] <= row["yhat"] <= row["yhat_upper"]


def test_forecast_is_deterministic():
    a = forecast_kpi(_hist(), kpi="k").frame["yhat"].to_numpy()
    b = forecast_kpi(_hist(), kpi="k").frame["yhat"].to_numpy()
    assert np.max(np.abs(a - b)) < 1e-6


def test_forecast_rejects_bad_history():
    with pytest.raises(ValueError):
        forecast_kpi(pd.DataFrame({"ds": [], "y": []}), kpi="k")
    with pytest.raises(ValueError):
        forecast_kpi(pd.DataFrame({"ds": pd.to_datetime(["2026-01-01", "2026-01-02"]), "y": [1.0, np.nan]}), kpi="k")
