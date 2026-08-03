# OrgPulse P2 — Anomaly Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect anomalies in the three pharma-ops streams using statistical (Z-score, CUSUM) and ML (Isolation Forest) methods against a remembered baseline, and combine them into a Safe/Warning/Critical severity zoning — proven by integration tests that the detectors actually catch the structural breaks the generators inject (clinical amendment surge, QC calibration drift, supply lead-time spikes).

**Architecture:** A new `orgpulse.detect` subpackage. Pure functions/classes: a `Baseline` (mean/std fit from a historical window), three detectors each returning a typed `AnomalyResult`, and a `detect_anomalies` orchestrator that runs the right detectors per KPI and reduces their results to one severity per KPI. No LLM, no Hermes, no forecasting here — detection only. Consumes the Plan 1 generators as test fixtures.

**Tech Stack:** Python 3.11, uv, numpy, pandas, scikit-learn (Isolation Forest), pytest.

## Global Constraints

- Python **3.11**; deps via **uv**; the project venv already exists (`.venv`, 3.11.14) — use `uv run`.
- This plan adds ONE new runtime dependency: **scikit-learn** (`scikit-learn>=1.4`). numpy/pandas already present. Do NOT add Prophet/LangGraph/Streamlit here.
- Detectors MUST be **deterministic**: given the same inputs (and fixed `random_state` for Isolation Forest), identical outputs.
- All severities use the single `Severity` enum from Task 1 — never bare strings.
- Commit style: `git -c user.name="naloo" -c user.email="mofasshara@hotmail.com" commit -m "<msg>"`; **no** Co-Authored-By line.
- The generators write CSV where `date` reloads as a string; tests here consume the in-memory generator DataFrames directly (dtype datetime64), so no CSV round-trip is needed.
- TDD: failing test → run (see fail) → minimal impl → run (see pass) → commit. Never weaken a test to make it pass; never add randomness to force a pass.

---

### Task 1: Detection types & baseline

**Files:**
- Create: `src/orgpulse/detect/__init__.py`
- Create: `src/orgpulse/detect/types.py`
- Create: `src/orgpulse/detect/baseline.py`
- Test: `tests/detect/__init__.py` (empty), `tests/detect/test_baseline.py`

**Interfaces:**
- Consumes: nothing from earlier plans (pure).
- Produces:
  - `orgpulse.detect.types.Severity` — `enum.Enum` with members `SAFE`, `WARNING`, `CRITICAL` and an ordering so `CRITICAL > WARNING > SAFE` (implement via `IntEnum` with SAFE=0, WARNING=1, CRITICAL=2).
  - `orgpulse.detect.types.AnomalyResult` — frozen dataclass: `kpi: str`, `method: str`, `severity: Severity`, `score: float`, `flagged: bool`, `detail: str`.
  - `orgpulse.detect.baseline.Baseline` — frozen dataclass: `mean: float`, `std: float`, `n: int`.
  - `orgpulse.detect.baseline.fit_baseline(values: numpy.ndarray) -> Baseline` — mean/std (population std, `ddof=0`) over the window; if `std == 0` it is floored to a tiny epsilon `1e-9` to avoid divide-by-zero downstream. Raises `ValueError` if `values` is empty.

- [ ] **Step 1: Write the failing test**

`tests/detect/test_baseline.py`:
```python
import numpy as np
import pytest

from orgpulse.detect.types import Severity, AnomalyResult
from orgpulse.detect.baseline import Baseline, fit_baseline


def test_severity_is_ordered():
    assert Severity.CRITICAL > Severity.WARNING > Severity.SAFE
    assert max(Severity.SAFE, Severity.CRITICAL) is Severity.CRITICAL


def test_anomaly_result_fields():
    r = AnomalyResult(kpi="k", method="zscore", severity=Severity.WARNING,
                      score=3.1, flagged=True, detail="d")
    assert r.kpi == "k" and r.flagged is True and r.severity is Severity.WARNING


def test_fit_baseline_mean_std():
    b = fit_baseline(np.array([10.0, 12.0, 14.0]))
    assert b.mean == pytest.approx(12.0)
    assert b.std == pytest.approx(np.std([10, 12, 14]))  # population std
    assert b.n == 3


def test_fit_baseline_zero_std_floored():
    b = fit_baseline(np.array([5.0, 5.0, 5.0]))
    assert b.std >= 1e-9 and b.std < 1e-6


def test_fit_baseline_empty_raises():
    with pytest.raises(ValueError):
        fit_baseline(np.array([]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/detect/test_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orgpulse.detect'`

- [ ] **Step 3: Write minimal implementation**

`src/orgpulse/detect/__init__.py`:
```python
"""Anomaly detection: statistical (Z-score, CUSUM) + ML (Isolation Forest)."""
```

`src/orgpulse/detect/types.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Severity(IntEnum):
    SAFE = 0
    WARNING = 1
    CRITICAL = 2


@dataclass(frozen=True)
class AnomalyResult:
    kpi: str
    method: str
    severity: Severity
    score: float
    flagged: bool
    detail: str
```

`src/orgpulse/detect/baseline.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-9


@dataclass(frozen=True)
class Baseline:
    mean: float
    std: float
    n: int


def fit_baseline(values: np.ndarray) -> Baseline:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("fit_baseline requires at least one value")
    std = float(np.std(arr, ddof=0))
    return Baseline(mean=float(np.mean(arr)), std=max(std, _EPS), n=int(arr.size))
```

Also create empty `tests/detect/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/detect/test_baseline.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orgpulse/detect/__init__.py src/orgpulse/detect/types.py src/orgpulse/detect/baseline.py tests/detect/
git commit -m "feat: detection types (Severity, AnomalyResult) and Baseline fit"
```

---

### Task 2: Z-score point-outlier detector

**Files:**
- Create: `src/orgpulse/detect/zscore.py`
- Test: `tests/detect/test_zscore.py`

**Interfaces:**
- Consumes: `Baseline`, `AnomalyResult`, `Severity`.
- Produces:
  - `orgpulse.detect.zscore.zscore_detect(value: float, baseline: Baseline, kpi: str, warn: float = 3.0, crit: float = 4.5) -> AnomalyResult`
  - `score` = signed z = `(value - mean) / std`. Severity by `abs(z)`: `>= crit` → CRITICAL, `>= warn` → WARNING, else SAFE. `flagged = severity > Severity.SAFE`.

- [ ] **Step 1: Write the failing test**

`tests/detect/test_zscore.py`:
```python
from orgpulse.detect.baseline import Baseline
from orgpulse.detect.types import Severity
from orgpulse.detect.zscore import zscore_detect


BASE = Baseline(mean=100.0, std=10.0, n=50)


def test_normal_value_is_safe():
    r = zscore_detect(105.0, BASE, kpi="enroll")
    assert r.severity is Severity.SAFE and r.flagged is False
    assert r.score == 0.5


def test_warning_at_three_sigma():
    r = zscore_detect(131.0, BASE, kpi="enroll")   # z = 3.1
    assert r.severity is Severity.WARNING and r.flagged is True


def test_critical_beyond_4_5_sigma():
    r = zscore_detect(146.0, BASE, kpi="enroll")   # z = 4.6
    assert r.severity is Severity.CRITICAL


def test_negative_direction_uses_absolute_z():
    r = zscore_detect(54.0, BASE, kpi="enroll")    # z = -4.6
    assert r.severity is Severity.CRITICAL and r.score < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/detect/test_zscore.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orgpulse.detect.zscore'`

- [ ] **Step 3: Write minimal implementation**

`src/orgpulse/detect/zscore.py`:
```python
from __future__ import annotations

from orgpulse.detect.baseline import Baseline
from orgpulse.detect.types import AnomalyResult, Severity


def zscore_detect(
    value: float, baseline: Baseline, kpi: str, warn: float = 3.0, crit: float = 4.5
) -> AnomalyResult:
    z = (value - baseline.mean) / baseline.std
    az = abs(z)
    if az >= crit:
        sev = Severity.CRITICAL
    elif az >= warn:
        sev = Severity.WARNING
    else:
        sev = Severity.SAFE
    return AnomalyResult(
        kpi=kpi,
        method="zscore",
        severity=sev,
        score=float(z),
        flagged=sev > Severity.SAFE,
        detail=f"z={z:.2f} (warn>={warn}, crit>={crit})",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/detect/test_zscore.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orgpulse/detect/zscore.py tests/detect/test_zscore.py
git commit -m "feat: Z-score point-outlier detector"
```

---

### Task 3: CUSUM drift/shift detector

**Files:**
- Create: `src/orgpulse/detect/cusum.py`
- Test: `tests/detect/test_cusum.py`

**Interfaces:**
- Consumes: `Baseline`, `AnomalyResult`, `Severity`.
- Produces:
  - `orgpulse.detect.cusum.cusum_detect(values: numpy.ndarray, baseline: Baseline, kpi: str, k: float = 0.5, h: float = 5.0) -> AnomalyResult`
  - Standardized two-sided tabular CUSUM. Standardize each `x_i` to `z_i = (x_i - mean)/std`. Recurrences (both start at 0):
    `S_hi[i] = max(0, S_hi[i-1] + z_i - k)`, `S_lo[i] = max(0, S_lo[i-1] - z_i - k)`.
  - `score` = the maximum of `max(S_hi)` and `max(S_lo)` over the series. Severity: `>= 2*h` → CRITICAL, `>= h` → WARNING, else SAFE. `detail` states which side (high/low) drove it and at which index it first crossed `h` (or `-1` if it never crossed).

**Why these defaults & thresholds:** `k=0.5` (slack, in σ units) tuned to detect a sustained ~1σ shift; `h=5.0` is the standard SPC decision interval. Severity uses two tiers above `h`: empirically (validated across 20 seeds) a stable in-control series can peak up to ~9 over a 60-sample window, while ANY sustained ≥1σ shift peaks well past 15 — so CRITICAL is set at `2*h`=10 (every stable series stays below it; every real shift clears it), and the `[h, 2*h)` band is the WARNING "gray zone". Setting CRITICAL at `h` alone would false-alarm on ~25% of stable windows.

- [ ] **Step 1: Write the failing test**

`tests/detect/test_cusum.py`:
```python
import numpy as np

from orgpulse.detect.baseline import Baseline, fit_baseline
from orgpulse.detect.types import Severity
from orgpulse.detect.cusum import cusum_detect


def test_stable_series_is_safe():
    rng = np.random.default_rng(0)
    base_window = rng.normal(50, 2, 100)
    baseline = fit_baseline(base_window)
    stable = rng.normal(50, 2, 60)          # same process
    r = cusum_detect(stable, baseline, kpi="pass_rate")
    assert r.severity is Severity.SAFE and r.flagged is False


def test_upward_shift_is_detected():
    rng = np.random.default_rng(1)
    baseline = fit_baseline(rng.normal(50, 2, 100))
    shifted = rng.normal(54, 2, 60)         # +2σ sustained shift
    r = cusum_detect(shifted, baseline, kpi="calibration")
    assert r.severity is Severity.CRITICAL and r.flagged is True
    assert "high" in r.detail


def test_downward_shift_is_detected():
    rng = np.random.default_rng(2)
    baseline = fit_baseline(rng.normal(50, 2, 100))
    dropped = rng.normal(46, 2, 60)         # -2σ sustained shift
    r = cusum_detect(dropped, baseline, kpi="reliability")
    assert r.severity is Severity.CRITICAL
    assert "low" in r.detail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/detect/test_cusum.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orgpulse.detect.cusum'`

- [ ] **Step 3: Write minimal implementation**

`src/orgpulse/detect/cusum.py`:
```python
from __future__ import annotations

import numpy as np

from orgpulse.detect.baseline import Baseline
from orgpulse.detect.types import AnomalyResult, Severity


def cusum_detect(
    values: np.ndarray, baseline: Baseline, kpi: str, k: float = 0.5, h: float = 5.0
) -> AnomalyResult:
    z = (np.asarray(values, dtype=float) - baseline.mean) / baseline.std
    s_hi = 0.0
    s_lo = 0.0
    max_hi = 0.0
    max_lo = 0.0
    cross_idx = -1
    for i, zi in enumerate(z):
        s_hi = max(0.0, s_hi + zi - k)
        s_lo = max(0.0, s_lo - zi - k)
        max_hi = max(max_hi, s_hi)
        max_lo = max(max_lo, s_lo)
        if cross_idx == -1 and (s_hi >= h or s_lo >= h):
            cross_idx = i
    score = max(max_hi, max_lo)
    side = "high" if max_hi >= max_lo else "low"
    if score >= 2.0 * h:
        sev = Severity.CRITICAL
    elif score >= h:
        sev = Severity.WARNING
    else:
        sev = Severity.SAFE
    return AnomalyResult(
        kpi=kpi,
        method="cusum",
        severity=sev,
        score=float(score),
        flagged=sev > Severity.SAFE,
        detail=f"{side} CUSUM peak={score:.2f} (h={h}); first cross at idx {cross_idx}",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/detect/test_cusum.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orgpulse/detect/cusum.py tests/detect/test_cusum.py
git commit -m "feat: standardized two-sided CUSUM drift/shift detector"
```

---

### Task 4: Isolation Forest multivariate detector

**Files:**
- Modify: `pyproject.toml` (add `scikit-learn>=1.4` to dependencies)
- Create: `src/orgpulse/detect/isolation.py`
- Test: `tests/detect/test_isolation.py`

**Interfaces:**
- Consumes: `AnomalyResult`, `Severity`.
- Produces:
  - `orgpulse.detect.isolation.isolation_detect(history: numpy.ndarray, current: numpy.ndarray, kpi: str, contamination: float = 0.05, random_state: int = 42) -> AnomalyResult`
  - `history` is a 2-D array (rows = past days, cols = features) used to fit the forest; `current` is a 1-D feature vector for the latest day. Fit `IsolationForest(contamination=contamination, random_state=random_state)` on `history`, then score `current` with `decision_function` (higher = more normal; negative = anomalous) and `predict` (-1 = anomaly).
  - `score` = the `decision_function` value for `current` (negated so higher = more anomalous, for consistency with the other detectors). Severity: predict == -1 AND score beyond a margin → escalate. Concretely: if `predict == -1` and `neg_score >= 0.10` → CRITICAL; if `predict == -1` → WARNING; else SAFE. (`neg_score = -decision_function`.)

- [ ] **Step 1: Add the dependency**

Edit `pyproject.toml` `dependencies` to add `"scikit-learn>=1.4"`. Then `uv sync` (or the next `uv run` resolves it).

- [ ] **Step 2: Write the failing test**

`tests/detect/test_isolation.py`:
```python
import numpy as np

from orgpulse.detect.types import Severity
from orgpulse.detect.isolation import isolation_detect


def test_normal_point_is_safe():
    rng = np.random.default_rng(0)
    history = rng.normal([100, 5], [8, 0.5], size=(200, 2))
    current = np.array([101.0, 5.1])          # squarely inside the cloud
    r = isolation_detect(history, current, kpi="supply")
    assert r.severity is Severity.SAFE and r.flagged is False


def test_outlier_point_is_flagged():
    rng = np.random.default_rng(0)
    history = rng.normal([100, 5], [8, 0.5], size=(200, 2))
    current = np.array([300.0, 25.0])         # far outside on both features
    r = isolation_detect(history, current, kpi="supply")
    assert r.flagged is True and r.severity in (Severity.WARNING, Severity.CRITICAL)


def test_deterministic_given_random_state():
    rng = np.random.default_rng(0)
    history = rng.normal([100, 5], [8, 0.5], size=(200, 2))
    current = np.array([300.0, 25.0])
    a = isolation_detect(history, current, kpi="supply")
    b = isolation_detect(history, current, kpi="supply")
    assert a.score == b.score and a.severity == b.severity
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/detect/test_isolation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orgpulse.detect.isolation'`

- [ ] **Step 4: Write minimal implementation**

`src/orgpulse/detect/isolation.py`:
```python
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from orgpulse.detect.types import AnomalyResult, Severity


def isolation_detect(
    history: np.ndarray,
    current: np.ndarray,
    kpi: str,
    contamination: float = 0.05,
    random_state: int = 42,
) -> AnomalyResult:
    hist = np.asarray(history, dtype=float)
    cur = np.asarray(current, dtype=float).reshape(1, -1)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(hist)
    pred = int(model.predict(cur)[0])                 # -1 anomaly, +1 normal
    neg_score = float(-model.decision_function(cur)[0])  # higher = more anomalous
    if pred == -1 and neg_score >= 0.10:
        sev = Severity.CRITICAL
    elif pred == -1:
        sev = Severity.WARNING
    else:
        sev = Severity.SAFE
    return AnomalyResult(
        kpi=kpi,
        method="isolation_forest",
        severity=sev,
        score=neg_score,
        flagged=sev > Severity.SAFE,
        detail=f"iforest neg_score={neg_score:.3f}, predict={pred}",
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/detect/test_isolation.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/orgpulse/detect/isolation.py tests/detect/test_isolation.py
git commit -m "feat: Isolation Forest multivariate anomaly detector"
```

---

### Task 5: `detect_anomalies` orchestrator + generator integration tests

**Files:**
- Create: `src/orgpulse/detect/orchestrator.py`
- Test: `tests/detect/test_orchestrator.py`
- Test: `tests/detect/test_integration_generators.py`

**Interfaces:**
- Consumes: all three detectors, `fit_baseline`, `Severity`, `AnomalyResult`, and (in the integration test only) the Plan 1 generators.
- Produces:
  - `orgpulse.detect.orchestrator.detect_kpi(history: numpy.ndarray, recent: numpy.ndarray, kpi: str) -> list[AnomalyResult]` — fits a baseline on `history`, runs Z-score on the latest value `recent[-1]` and CUSUM on the whole `recent` window; returns both results.
  - `orgpulse.detect.orchestrator.overall_severity(results: list[AnomalyResult]) -> Severity` — the max severity across results (SAFE if empty).

- [ ] **Step 1: Write the failing unit test**

`tests/detect/test_orchestrator.py`:
```python
import numpy as np

from orgpulse.detect.types import Severity
from orgpulse.detect.orchestrator import detect_kpi, overall_severity


def test_detect_kpi_returns_zscore_and_cusum():
    rng = np.random.default_rng(0)
    history = rng.normal(50, 2, 120)
    recent = rng.normal(50, 2, 30)
    results = detect_kpi(history, recent, kpi="pass_rate")
    methods = {r.method for r in results}
    assert methods == {"zscore", "cusum"}


def test_overall_severity_is_max():
    rng = np.random.default_rng(0)
    history = rng.normal(50, 2, 120)
    recent = rng.normal(50, 2, 30)
    calm = overall_severity(detect_kpi(history, recent, kpi="k"))
    assert calm is Severity.SAFE
    assert overall_severity([]) is Severity.SAFE


def test_shift_raises_overall_severity():
    rng = np.random.default_rng(1)
    history = rng.normal(50, 2, 120)
    recent = rng.normal(56, 2, 30)          # +3σ shift
    sev = overall_severity(detect_kpi(history, recent, kpi="k"))
    assert sev >= Severity.WARNING
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/detect/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orgpulse.detect.orchestrator'`

- [ ] **Step 3: Write minimal implementation**

`src/orgpulse/detect/orchestrator.py`:
```python
from __future__ import annotations

import numpy as np

from orgpulse.detect.baseline import fit_baseline
from orgpulse.detect.cusum import cusum_detect
from orgpulse.detect.types import AnomalyResult, Severity
from orgpulse.detect.zscore import zscore_detect


def detect_kpi(history: np.ndarray, recent: np.ndarray, kpi: str) -> list[AnomalyResult]:
    baseline = fit_baseline(np.asarray(history, dtype=float))
    recent = np.asarray(recent, dtype=float)
    return [
        zscore_detect(float(recent[-1]), baseline, kpi=kpi),
        cusum_detect(recent, baseline, kpi=kpi),
    ]


def overall_severity(results: list[AnomalyResult]) -> Severity:
    if not results:
        return Severity.SAFE
    return max(r.severity for r in results)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/detect/test_orchestrator.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Write the integration test (detectors catch injected breaks)**

`tests/detect/test_integration_generators.py`:
```python
from datetime import date

import numpy as np

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical, PROTOCOL_AMENDMENT_DAY
from orgpulse.generators.qc import generate_qc
from orgpulse.detect.baseline import fit_baseline
from orgpulse.detect.cusum import cusum_detect
from orgpulse.detect.orchestrator import detect_kpi, overall_severity
from orgpulse.detect.types import Severity


def test_detects_qc_calibration_drift():
    # QC calibration_offset drifts upward over time -> CUSUM should fire.
    df = generate_qc(date(2026, 1, 1), 300, make_rng(42))
    series = df["calibration_offset"].to_numpy()
    baseline = fit_baseline(series[:60])          # early, near-zero window
    recent = series[-60:]                          # late, drifted window
    r = cusum_detect(recent, baseline, kpi="calibration_offset")
    assert r.severity is Severity.CRITICAL and "high" in r.detail


def test_detects_clinical_amendment_surge():
    # Post-amendment daily enrollment surges vs the pre-amendment plateau.
    df = generate_clinical(date(2026, 1, 1), 200, make_rng(42))
    per_day = df.groupby("date")["daily_enrollment"].sum().sort_index().to_numpy()
    history = per_day[PROTOCOL_AMENDMENT_DAY - 40 : PROTOCOL_AMENDMENT_DAY - 10]
    recent = per_day[PROTOCOL_AMENDMENT_DAY : PROTOCOL_AMENDMENT_DAY + 30]
    sev = overall_severity(detect_kpi(history, recent, kpi="daily_enrollment"))
    assert sev >= Severity.WARNING


def test_calm_baseline_stays_safe():
    # A stable stretch of the QC pass_rate should not raise a critical alarm.
    df = generate_qc(date(2026, 1, 1), 200, make_rng(7))
    series = df["pass_rate"].to_numpy()
    history = series[:120]
    recent = series[120:150]
    sev = overall_severity(detect_kpi(history, recent, kpi="pass_rate"))
    assert sev <= Severity.WARNING     # no false CRITICAL on a calm stretch
```

- [ ] **Step 6: Run the integration test**

Run: `uv run pytest tests/detect/test_integration_generators.py -v`
Expected: PASS (3 tests). If `test_detects_clinical_amendment_surge` does not reach WARNING, the history/recent windows are the lever — the surge is real (validated in Plan 1); do NOT weaken the assertion without first printing the z/CUSUM scores to confirm the windows capture pre- vs post-amendment. If it genuinely won't fire, report the scores rather than lowering the bar.

- [ ] **Step 7: Run the full suite + commit**

Run: `uv run pytest -q`
Expected: all tests pass (Plan 1's 24 + this plan's new tests).

```bash
git add src/orgpulse/detect/orchestrator.py tests/detect/test_orchestrator.py tests/detect/test_integration_generators.py
git commit -m "feat: detect_anomalies orchestrator + generator integration tests"
```

---

## Self-Review

**Spec coverage (P2):**
- CUSUM detector → Task 3. ✓
- Z-score detector → Task 2. ✓
- Isolation Forest (ML) → Task 4. ✓
- Safe/Warning/Critical zoning → `Severity` (Task 1) + `overall_severity` (Task 5). ✓
- Compared against a remembered baseline → `Baseline`/`fit_baseline` (Task 1), consumed by every detector. ✓
- Detectors actually catch the generators' injected breaks → integration tests (Task 5). ✓
- Deferred correctly: forecasting (Plan 3), LangGraph/governance (Plan 3+), Hermes baseline *memory* across runs (Plan 4 — here the baseline is fit from a passed-in history window, which Hermes will later supply from memory).

**Placeholder scan:** No TBD/TODO; every code + test step has real content. ✓

**Type consistency:** `Severity` is one `IntEnum` used everywhere (ordering relied on by `overall_severity` and the zoning branches). Every detector returns `AnomalyResult` with identical field names/types. `fit_baseline(np.ndarray) -> Baseline` consumed uniformly. `detect_kpi` returns `list[AnomalyResult]`; `overall_severity(list[AnomalyResult]) -> Severity`. Isolation Forest's `random_state` is fixed (42) for determinism, matching the Global Constraints. ✓

**Dependency note:** scikit-learn is the only new runtime dep (Task 4); it ships binary wheels for macOS arm64, so no compile risk (unlike Prophet in Plan 3).
