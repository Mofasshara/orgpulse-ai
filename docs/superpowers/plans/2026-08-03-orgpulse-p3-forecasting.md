# OrgPulse P3 — Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Forecast the hero KPI (clinical daily enrollment) 30/60/90 days ahead with Prophet, and — the real rigor — **honestly measure** forecast quality with a rolling-origin backtest (MAPE/MASE) and an interval-calibration check that detects Prophet's overconfidence and corrects it to nominal coverage. Report the truth, including where the model is beaten by naive.

**Architecture:** A new `orgpulse.forecast` subpackage. `forecast_kpi` wraps Prophet (deterministic MAP fit) and returns a typed `Forecast` with point + 90% interval at each horizon. `rolling_origin_backtest` refits at successive cutoffs and reports averaged MAPE and MASE (scaled by the in-sample one-step naive error). `interval_calibration` measures empirical coverage of the nominal intervals over backtest folds and computes an empirical widening factor that brings coverage back to nominal. All three consume the Plan-1 generators as data. No LLM/Hermes/Streamlit here.

**Tech Stack:** Python 3.11, uv, Prophet 1.3, pandas, numpy, pytest. (Prophet install de-risked: installs in ~2s, deterministic MAP fit ~1-2s.)

## Global Constraints

- Python **3.11**; deps via **uv**; project venv (use `uv run`). Ignore `VIRTUAL_ENV ... does not match`.
- Prophet is a new runtime dep (already added via `uv add "prophet>=1.1"`). It is a **compute-plane** dependency; the Streamlit app (Plan 6) will NOT import it (split-plane).
- Prophet noise: silence it in code — `logging.getLogger("cmdstanpy").setLevel(logging.ERROR)` and `warnings` filter inside the module, so the suite output stays clean.
- Forecasts must be **deterministic** (Prophet MAP/LBFGS is; verified 0.0 diff between identical fits). Tests compare with tolerances, never exact floats across environments.
- **Honesty over flattery:** tests assert what is TRUE and robust (finite metrics, plausible ranges, calibration *improves* coverage), NOT "MASE < 1" (it is ~1.4) or a fixed coverage number. Observed on the hero series (5 folds, h=30): MAPE ≈17%, MASE ≈1.4, raw 90%-coverage ≈0.70.
- Prophet fits are ~1-2s; keep backtest folds modest in tests (3-4) so the suite stays reasonable.
- Commit style: `git -c user.name="naloo" -c user.email="mofasshara@hotmail.com" commit -m "<msg>"`; no Co-Authored-By.
- TDD throughout. Never weaken a test to pass; report BLOCKED with evidence if a bound genuinely fails.

---

### Task 1: Prophet forecaster + `Forecast` type

**Files:**
- Create: `src/orgpulse/forecast/__init__.py`
- Create: `src/orgpulse/forecast/types.py`
- Create: `src/orgpulse/forecast/prophet_forecaster.py`
- Test: `tests/forecast/__init__.py` (empty), `tests/forecast/test_forecaster.py`

**Interfaces:**
- `orgpulse.forecast.types.Forecast` — frozen dataclass: `kpi: str`, `frame: pandas.DataFrame` (columns `ds, yhat, yhat_lower, yhat_upper`, one row per forecasted day), `horizons: tuple[int, ...]`, `interval_width: float`. Method `at_horizon(h: int) -> dict` returning `{"ds","yhat","yhat_lower","yhat_upper"}` for the row `h` days past the last training date (h in horizons).
- `orgpulse.forecast.prophet_forecaster.forecast_kpi(history: pandas.DataFrame, kpi: str, horizons=(30,60,90), interval_width=0.9) -> Forecast` — `history` has columns `ds` (datetime) and `y` (float); validates non-empty, ≥2 rows, finite `y` (raises `ValueError` otherwise); fits `Prophet(interval_width=interval_width, weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)`; forecasts `max(horizons)` days; returns the horizon rows.
- `orgpulse.forecast.prophet_forecaster.series_from_generator(df, date_col, value_col) -> pandas.DataFrame` — helper: aggregates a generator DataFrame to a daily `ds,y` frame (sum for counts).

- [ ] **Step 1: Write the failing test**

`tests/forecast/test_forecaster.py`:
```python
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
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/forecast/test_forecaster.py -v` (ModuleNotFound).

- [ ] **Step 3: Implement**

`src/orgpulse/forecast/__init__.py`:
```python
"""Forecasting: Prophet point+interval forecasts with honest backtest & calibration."""
```

`src/orgpulse/forecast/types.py`:
```python
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
```

`src/orgpulse/forecast/prophet_forecaster.py`:
```python
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from orgpulse.forecast.types import Forecast

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


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
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/forecast/test_forecaster.py -v` then `uv run pytest -q`.

- [ ] **Step 5: Commit**
```bash
git add src/orgpulse/forecast/ tests/forecast/
git commit -m "feat: Prophet forecaster (30/60/90d + 90% intervals), deterministic"
```

---

### Task 2: Rolling-origin backtest (MAPE / MASE)

**Files:**
- Create: `src/orgpulse/forecast/backtest.py`
- Test: `tests/forecast/test_backtest.py`

**Interfaces:**
- `orgpulse.forecast.types.BacktestResult` — frozen dataclass: `horizon: int`, `n_folds: int`, `mape: float`, `mase: float`, `per_fold_mape: tuple[float, ...]`, `per_fold_mase: tuple[float, ...]`. (Add to `types.py`.)
- `orgpulse.forecast.backtest.rolling_origin_backtest(series: pandas.DataFrame, horizon=30, n_folds=4, min_train=150, interval_width=0.9) -> BacktestResult` — cutoffs evenly spaced from `min_train` to `len(series)-horizon`; at each, fit on `series[:cutoff]`, forecast `horizon`, compare to actuals. **MAPE** = mean |(actual-yhat)/max(actual,1)|·100. **MASE** = mean|actual-yhat| / (in-sample one-step naive MAE on the training y, i.e. `mean(|diff(train_y)|)`). Returns averaged + per-fold.

**Honesty note:** On the hero series this yields MAPE≈17%, MASE≈1.4 — the model does NOT beat the in-sample naive one-step scale, and the tests reflect that (assert plausible ranges, not superiority).

- [ ] **Step 1: Write the failing test**

`tests/forecast/test_backtest.py`:
```python
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


def test_backtest_uses_only_past_data_per_fold():
    # Determinism + no-leakage sanity: same series -> same metrics.
    a = rolling_origin_backtest(_series(), horizon=30, n_folds=3)
    b = rolling_origin_backtest(_series(), horizon=30, n_folds=3)
    assert abs(a.mape - b.mape) < 1e-6 and abs(a.mase - b.mase) < 1e-6
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** `src/orgpulse/forecast/backtest.py`
```python
from __future__ import annotations

import numpy as np
import pandas as pd

from orgpulse.forecast.prophet_forecaster import forecast_kpi
from orgpulse.forecast.types import BacktestResult


def rolling_origin_backtest(
    series: pd.DataFrame, horizon: int = 30, n_folds: int = 4, min_train: int = 150, interval_width: float = 0.9
) -> BacktestResult:
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
        scale = float(np.mean(np.abs(np.diff(train["y"].to_numpy()))))
        mapes.append(float(np.mean(np.abs((actual - pred) / np.maximum(actual, 1.0))) * 100))
        mases.append(float(np.mean(np.abs(actual - pred)) / scale))
    return BacktestResult(
        horizon=horizon, n_folds=n_folds,
        mape=float(np.mean(mapes)), mase=float(np.mean(mases)),
        per_fold_mape=tuple(mapes), per_fold_mase=tuple(mases),
    )
```

- [ ] **Step 4: Run to verify pass** (takes ~10-20s — Prophet fits per fold), then `uv run pytest -q`.

- [ ] **Step 5: Commit**
```bash
git add src/orgpulse/forecast/backtest.py tests/forecast/test_backtest.py
git commit -m "feat: rolling-origin backtest reporting honest MAPE/MASE"
```

---

### Task 3: Interval calibration (detect overconfidence, correct to nominal)

**Files:**
- Create: `src/orgpulse/forecast/calibration.py`
- Test: `tests/forecast/test_calibration.py`

**Interfaces:**
- `orgpulse.forecast.types.CalibrationResult` — frozen dataclass: `nominal: float`, `raw_coverage: float`, `scale_factor: float`, `adjusted_coverage: float`, `n_points: int`. (Add to `types.py`.)
- `orgpulse.forecast.calibration.interval_calibration(series, horizon=30, n_folds=4, nominal=0.9, min_train=150) -> CalibrationResult` — over rolling-origin folds, collect `(actual, yhat, yhat_lower, yhat_upper)` for all forecasted points. `raw_coverage` = fraction of actuals within `[yhat_lower, yhat_upper]`. Compute the empirical `scale_factor` s so that widening each interval about `yhat` by `s` (`yhat ± s·(yhat_upper-yhat)` and `yhat - s·(yhat-yhat_lower)`) makes `adjusted_coverage ≈ nominal`: set `s` = the `nominal`-quantile of the per-point ratio `|actual-yhat| / half_width` (half_width = (upper-lower)/2). `adjusted_coverage` = fraction covered by the widened intervals.

**Why this is the showcase:** Prophet's raw 90% intervals cover only ~70% here (overconfident). The empirical `scale_factor` (~1.5-2×) recalibrates them so the stated 90% band actually contains ~90% — honest uncertainty quantification.

- [ ] **Step 1: Write the failing test**

`tests/forecast/test_calibration.py`:
```python
from datetime import date

import numpy as np

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical
from orgpulse.forecast.prophet_forecaster import series_from_generator
from orgpulse.forecast.calibration import interval_calibration
from orgpulse.forecast.types import CalibrationResult


def _series(days=260):
    df = generate_clinical(date(2026, 1, 1), days, make_rng(42))
    return series_from_generator(df, "date", "daily_enrollment")


def test_calibration_detects_and_corrects_overconfidence():
    r = interval_calibration(_series(), horizon=30, n_folds=4, nominal=0.9)
    assert isinstance(r, CalibrationResult)
    assert r.nominal == 0.9 and r.n_points == 120
    # Prophet's raw intervals are overconfident here (measured ~0.70):
    assert r.raw_coverage < r.nominal
    # The empirical scale widens them (>= 1) and brings coverage up to ~nominal:
    assert r.scale_factor >= 1.0
    assert r.adjusted_coverage >= r.raw_coverage
    assert abs(r.adjusted_coverage - r.nominal) <= 0.08     # within 8pts of nominal
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** `src/orgpulse/forecast/calibration.py`
```python
from __future__ import annotations

import numpy as np
import pandas as pd

from orgpulse.forecast.prophet_forecaster import forecast_kpi
from orgpulse.forecast.types import CalibrationResult


def interval_calibration(
    series: pd.DataFrame, horizon: int = 30, n_folds: int = 4, nominal: float = 0.9, min_train: int = 150
) -> CalibrationResult:
    """Diagnose and correct Prophet's interval overconfidence to nominal coverage.

    Over rolling-origin folds, measures `raw_coverage` of the nominal intervals and
    computes a widening `scale_factor` (the nominal-quantile of the standardized
    residuals), then reports the resulting `adjusted_coverage`.

    HONESTY NOTE: this is an IN-SAMPLE calibration — the scale is both fit and
    evaluated on the same rolling-fold points, so `adjusted_coverage ≈ nominal`
    holds largely by construction and is NOT a held-out estimate of generalization.
    A production version would fit the scale on a separate calibration fold
    (split-conformal prediction). The value here is diagnostic: it quantifies how
    overconfident Prophet's raw intervals are (`raw_coverage` << nominal) and the
    correction factor needed to reach nominal.
    """
    n = len(series)
    last_cutoff = n - horizon
    if last_cutoff <= min_train:
        raise ValueError("series too short for the requested horizon/min_train")
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
```

- [ ] **Step 4: Run to verify pass** (~10-20s), then `uv run pytest -q`. If `raw_coverage < nominal` genuinely does not hold for some seed, report the observed coverage — do NOT flip the assertion; investigate.

- [ ] **Step 5: Commit**
```bash
git add src/orgpulse/forecast/calibration.py tests/forecast/test_calibration.py
git commit -m "feat: interval calibration — detect Prophet overconfidence, correct to nominal coverage"
```

---

## Self-Review

**Spec coverage (P3):**
- Prophet forecaster 30/60/90 + intervals → Task 1. ✓
- Rolling-origin backtest MAPE/MASE → Task 2. ✓
- Interval calibration / coverage → Task 3 (measures AND corrects). ✓
- Honesty constraint honored: tests assert true, robust facts (finite metrics, plausible ranges, calibration improves coverage), not fabricated superiority. Observed values (MAPE 17%, MASE 1.4, raw coverage 0.70) documented, not hidden.
- Deferred to later plans: wiring forecasts into the LangGraph DAG (Plan 4); forecasting the other two streams (lighter, Plan 4/later); the Streamlit forecast-band chart (Plan 6).

**Placeholder scan:** none. **Type consistency:** `Forecast`/`BacktestResult`/`CalibrationResult` all in `types.py`; `forecast_kpi`/`series_from_generator` signatures consistent across tasks; backtest and calibration both reuse `forecast_kpi`. Prophet noise suppressed. Determinism asserted with tolerances.

**Risk:** Prophet fits make Tasks 2–3 tests take ~10-40s total. Acceptable; if CI-time matters later, mark them `slow`. Coverage/MASE bounds are ranges validated by the controller before planning, not single-seed guesses.
