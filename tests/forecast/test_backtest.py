from datetime import date

import numpy as np

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical
from orgpulse.forecast.prophet_forecaster import series_from_generator
from orgpulse.forecast.backtest import rolling_origin_backtest
from orgpulse.forecast.types import BacktestResult


def _series(days=260):
    df = generate_clinical(date(2026, 1, 1), days, make_rng(42))
    return series_from_generator(df, "date", "daily_enrollment")


def test_backtest_returns_finite_plausible_metrics():
    r = rolling_origin_backtest(_series(), horizon=30, n_folds=4)
    assert isinstance(r, BacktestResult)
    assert r.n_folds == 4 and len(r.per_fold_mape) == 4
    assert np.isfinite(r.mape) and 0 < r.mape < 60           # honest: ~17%
    assert np.isfinite(r.mase) and 0 < r.mase < 5            # honest: ~1.4 (does NOT beat naive)


def test_backtest_is_deterministic():
    # Determinism / regression check: same series -> same metrics.
    a = rolling_origin_backtest(_series(), horizon=30, n_folds=3)
    b = rolling_origin_backtest(_series(), horizon=30, n_folds=3)
    assert abs(a.mape - b.mape) < 1e-6 and abs(a.mase - b.mase) < 1e-6
