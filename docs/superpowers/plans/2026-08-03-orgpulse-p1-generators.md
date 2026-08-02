# OrgPulse P1 — Scaffolding & Synthetic Data Generators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the project scaffolding and three synthetic pharma-ops data generators (clinical trials [hero], QC/lab, supply chain), exposed via an `orgpulse generate` CLI, with tests that prove each stream's deliberately-injected structural features are present.

**Architecture:** A `src/orgpulse` Python package. Generators are pure functions returning `pandas.DataFrame`s seeded by a shared `numpy.random.Generator` for reproducibility. Each generator owns one stream and injects realistic structure (staggered logistic enrollment + protocol-amendment step + holiday dip + a laggard site; QC calibration drift + OOS clusters; supply sawtooth + lead-time spikes + seasonality). A Typer CLI writes the streams to a dated output directory. No detection/forecasting/LLM here — this plan is the data foundation only.

**Tech Stack:** Python 3.11, uv, numpy, pandas, Typer, pytest.

## Global Constraints

- Python **3.11** exactly (Hermes venv target).
- Dependency management via **uv**.
- Generators MUST be **deterministic given a seed** — identical seed ⇒ identical DataFrame.
- No network calls, no real data — synthetic only.
- This plan adds ONLY these runtime deps: `numpy`, `pandas`, `typer`. (Prophet, scikit-learn, LangGraph, Streamlit belong to later plans — do not add them here.)
- Every generator returns a `pandas.DataFrame` with a `date` column of dtype `datetime64[ns]`.
- Follow TDD: failing test → run (see it fail) → minimal impl → run (see it pass) → commit.

---

### Task 1: Project scaffolding & config

**Files:**
- Create: `pyproject.toml`
- Create: `src/orgpulse/__init__.py`
- Create: `src/orgpulse/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `orgpulse.config.OrgPulseConfig` — a frozen dataclass with fields:
    `seed: int = 42`, `start_date: datetime.date = date(2026, 1, 1)`,
    `n_sites: int = 8`, `runs_dir: pathlib.Path = Path("runs")`.
  - `orgpulse.config.make_rng(seed: int) -> numpy.random.Generator`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "orgpulse"
version = "0.1.0"
description = "Autonomous pharma-ops organizational intelligence platform"
requires-python = "==3.11.*"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "typer>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
orgpulse = "orgpulse.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/orgpulse"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create empty package init**

`src/orgpulse/__init__.py`:
```python
"""OrgPulse AI — PharmaOps: autonomous organizational intelligence platform."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Write the failing test**

`tests/test_config.py`:
```python
from datetime import date

import numpy as np

from orgpulse.config import OrgPulseConfig, make_rng


def test_default_config_values():
    cfg = OrgPulseConfig()
    assert cfg.seed == 42
    assert cfg.start_date == date(2026, 1, 1)
    assert cfg.n_sites == 8


def test_config_is_frozen():
    cfg = OrgPulseConfig()
    try:
        cfg.seed = 7  # type: ignore[misc]
    except Exception as exc:  # frozen dataclass raises FrozenInstanceError
        assert "cannot assign" in str(exc).lower() or exc.__class__.__name__ == "FrozenInstanceError"
    else:
        raise AssertionError("expected OrgPulseConfig to be frozen")


def test_make_rng_is_deterministic():
    a = make_rng(42).random(5)
    b = make_rng(42).random(5)
    assert np.array_equal(a, b)
    assert isinstance(make_rng(42), np.random.Generator)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orgpulse.config'`

- [ ] **Step 5: Write minimal implementation**

`src/orgpulse/config.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OrgPulseConfig:
    seed: int = 42
    start_date: date = date(2026, 1, 1)
    n_sites: int = 8
    runs_dir: Path = field(default_factory=lambda: Path("runs"))


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/orgpulse/__init__.py src/orgpulse/config.py tests/test_config.py
git commit -m "feat: project scaffolding and OrgPulseConfig"
```

---

### Task 2: Shared generator primitives

**Files:**
- Create: `src/orgpulse/generators/__init__.py`
- Create: `src/orgpulse/generators/base.py`
- Test: `tests/test_base.py`

**Interfaces:**
- Consumes: `orgpulse.config.make_rng`.
- Produces:
  - `date_series(start: date, days: int) -> list[date]` — `days` consecutive dates from `start`.
  - `logistic_curve(days: int, ceiling: float, midpoint: float, steepness: float) -> numpy.ndarray` — length-`days` cumulative logistic values in `[0, ceiling]`.
  - `is_holiday_dip(d: date) -> float` — multiplier in `(0, 1]`; returns `0.4` for dates in the last 10 days of December, else `1.0`.

- [ ] **Step 1: Write the failing test**

`tests/test_base.py`:
```python
from datetime import date

import numpy as np

from orgpulse.generators.base import date_series, logistic_curve, is_holiday_dip


def test_date_series_length_and_spacing():
    ds = date_series(date(2026, 1, 1), 5)
    assert ds[0] == date(2026, 1, 1)
    assert ds[-1] == date(2026, 1, 5)
    assert len(ds) == 5
    assert (ds[1] - ds[0]).days == 1


def test_logistic_curve_is_monotonic_and_bounded():
    curve = logistic_curve(days=100, ceiling=500.0, midpoint=50.0, steepness=0.1)
    assert len(curve) == 100
    assert np.all(np.diff(curve) >= -1e-9)          # non-decreasing
    assert curve[-1] <= 500.0 + 1e-6                # bounded by ceiling
    assert curve[-1] > curve[0]                     # actually grows


def test_holiday_dip_reduces_late_december():
    assert is_holiday_dip(date(2026, 12, 24)) < 1.0
    assert is_holiday_dip(date(2026, 6, 15)) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orgpulse.generators'`

- [ ] **Step 3: Write minimal implementation**

`src/orgpulse/generators/__init__.py`:
```python
"""Synthetic data generators for the three pharma-ops streams."""
```

`src/orgpulse/generators/base.py`:
```python
from __future__ import annotations

from datetime import date, timedelta

import numpy as np


def date_series(start: date, days: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(days)]


def logistic_curve(days: int, ceiling: float, midpoint: float, steepness: float) -> np.ndarray:
    t = np.arange(days)
    return ceiling / (1.0 + np.exp(-steepness * (t - midpoint)))


def is_holiday_dip(d: date) -> float:
    # Enrollment slows in the winter-holiday window (last 10 days of December).
    if d.month == 12 and d.day >= 22:
        return 0.4
    return 1.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_base.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orgpulse/generators/__init__.py src/orgpulse/generators/base.py tests/test_base.py
git commit -m "feat: shared generator primitives (date series, logistic curve, holiday dip)"
```

---

### Task 3: Clinical-trial enrollment generator (HERO stream)

**Files:**
- Create: `src/orgpulse/generators/clinical.py`
- Test: `tests/test_clinical_generator.py`

**Interfaces:**
- Consumes: `make_rng`, `date_series`, `logistic_curve`, `is_holiday_dip`.
- Produces:
  - `generate_clinical(start_date: date, days: int, rng: numpy.random.Generator, n_sites: int = 8) -> pandas.DataFrame`
  - Returned DataFrame is **long format**, one row per (date, site), columns:
    `date` (datetime64), `site_id` (int 0..n_sites-1), `daily_enrollment` (int ≥ 0),
    `cumulative_enrollment` (int ≥ 0, non-decreasing per site),
    `screen_fail_rate` (float in [0,1]), `dropout_rate` (float in [0,1]).
  - Injected structure the forecaster is NOT told about:
    - **Site 0 is a laggard** — its enrollment ceiling is ~35% of the others.
    - **Staggered activation** — each site starts enrolling at a random day offset in `[0, 30]`.
    - **Protocol-amendment step** — from day `PROTOCOL_AMENDMENT_DAY = 120` onward, the enrollment **ceiling** is raised ×1.6 (an amendment loosened inclusion criteria, increasing the addressable population). NB: raising the ceiling — not multiplying the daily rate — is what actually produces a visible post-amendment surge, because by day 120 the logistic curve has plateaued and daily increments are near zero; multiplying ~0 stays ~0.
    - **Holiday dip** — via `is_holiday_dip`.
  - Module constant `PROTOCOL_AMENDMENT_DAY: int = 120` is exported for later tests/analysis.

- [ ] **Step 1: Write the failing test**

`tests/test_clinical_generator.py`:
```python
from datetime import date

import numpy as np
import pandas as pd

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical, PROTOCOL_AMENDMENT_DAY


def _df(days=200, n_sites=8, seed=42):
    return generate_clinical(date(2026, 1, 1), days, make_rng(seed), n_sites=n_sites)


def test_schema_and_types():
    df = _df()
    assert list(df.columns) == [
        "date", "site_id", "daily_enrollment", "cumulative_enrollment",
        "screen_fail_rate", "dropout_rate",
    ]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["site_id"].nunique() == 8


def test_cumulative_enrollment_non_decreasing_per_site():
    df = _df()
    for _, g in df.groupby("site_id"):
        g = g.sort_values("date")
        assert np.all(np.diff(g["cumulative_enrollment"].to_numpy()) >= 0)


def test_rates_are_bounded():
    df = _df()
    assert df["screen_fail_rate"].between(0, 1).all()
    assert df["dropout_rate"].between(0, 1).all()
    assert (df["daily_enrollment"] >= 0).all()


def test_site_zero_is_the_laggard():
    df = _df()
    totals = df.groupby("site_id")["daily_enrollment"].sum()
    assert totals[0] < totals.median()


def test_protocol_amendment_raises_enrollment():
    # Compare a 30-day window before vs after the amendment, summed across sites.
    df = _df(days=200)
    per_day = df.groupby("date")["daily_enrollment"].sum().sort_index()
    amend_date = per_day.index[PROTOCOL_AMENDMENT_DAY]
    before = per_day[per_day.index < amend_date].tail(30).mean()
    after = per_day[per_day.index >= amend_date].head(30).mean()
    assert after > before


def test_deterministic_given_seed():
    a = _df(seed=7)
    b = _df(seed=7)
    pd.testing.assert_frame_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_clinical_generator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orgpulse.generators.clinical'`

- [ ] **Step 3: Write minimal implementation**

`src/orgpulse/generators/clinical.py`:
```python
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from orgpulse.generators.base import date_series, is_holiday_dip, logistic_curve

PROTOCOL_AMENDMENT_DAY: int = 120
_AMENDMENT_BOOST: float = 1.6


def generate_clinical(
    start_date: date, days: int, rng: np.random.Generator, n_sites: int = 8
) -> pd.DataFrame:
    dates = date_series(start_date, days)
    rows = []
    for site_id in range(n_sites):
        laggard = site_id == 0
        ceiling = rng.uniform(400, 600) * (0.35 if laggard else 1.0)
        midpoint = rng.uniform(40, 80)
        steepness = rng.uniform(0.05, 0.12)
        activation = int(rng.integers(0, 30))

        # Amendment raises the enrollment CEILING (capacity), re-opening a real
        # surge; multiplying the plateaued daily rate would not (see note above).
        cumulative_ideal_before = logistic_curve(days, ceiling, midpoint + activation, steepness)
        boosted_ceiling = ceiling * _AMENDMENT_BOOST
        cumulative_ideal_after = logistic_curve(days, boosted_ceiling, midpoint + activation, steepness)

        prev_cum = 0.0
        cum = 0
        for day_idx, d in enumerate(dates):
            if day_idx < activation:
                daily = 0.0
            else:
                if day_idx < PROTOCOL_AMENDMENT_DAY:
                    target_cum = cumulative_ideal_before[day_idx]
                else:
                    target_cum = cumulative_ideal_after[day_idx]
                inc = max(0.0, target_cum - prev_cum)
                prev_cum = target_cum
                daily = inc * is_holiday_dip(d)
                daily += rng.normal(0, 0.5)  # measurement noise
            daily_int = int(max(0, round(daily)))
            cum += daily_int
            rows.append(
                {
                    "date": d,
                    "site_id": site_id,
                    "daily_enrollment": daily_int,
                    "cumulative_enrollment": cum,
                    "screen_fail_rate": float(np.clip(rng.normal(0.18, 0.03), 0, 1)),
                    "dropout_rate": float(np.clip(rng.normal(0.08, 0.02), 0, 1)),
                }
            )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["site_id", "date"]).reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_clinical_generator.py -v`
Expected: PASS (6 tests). The ceiling-raise mechanism yields ~2.2× higher enrollment in the 30-day post-amendment window vs. the 30-day pre-amendment window (before≈33/day, after≈74/day at seed 42), so `test_protocol_amendment_raises_enrollment` passes with wide margin. The seed is fixed — do not add randomness to "fix" any test.

- [ ] **Step 5: Commit**

```bash
git add src/orgpulse/generators/clinical.py tests/test_clinical_generator.py
git commit -m "feat: clinical-trial enrollment generator with injected structural breaks"
```

---

### Task 4: QC/lab generator

**Files:**
- Create: `src/orgpulse/generators/qc.py`
- Test: `tests/test_qc_generator.py`

**Interfaces:**
- Consumes: `make_rng`, `date_series`.
- Produces:
  - `generate_qc(start_date: date, days: int, rng: numpy.random.Generator) -> pandas.DataFrame`
  - **Wide format**, one row per date, columns:
    `date` (datetime64), `batches_tested` (int > 0), `batches_passed` (int ≥ 0),
    `pass_rate` (float in [0,1]), `oos_events` (int ≥ 0),
    `calibration_offset` (float, slowly drifting upward — the CUSUM target).
  - Injected structure: `calibration_offset` is a cumulative random-walk with positive drift
    (equipment slowly de-calibrating); OOS events are rare Poisson counts that cluster.

- [ ] **Step 1: Write the failing test**

`tests/test_qc_generator.py`:
```python
from datetime import date

import numpy as np
import pandas as pd

from orgpulse.config import make_rng
from orgpulse.generators.qc import generate_qc


def _df(days=200, seed=42):
    return generate_qc(date(2026, 1, 1), days, make_rng(seed))


def test_schema_and_types():
    df = _df()
    assert list(df.columns) == [
        "date", "batches_tested", "batches_passed", "pass_rate",
        "oos_events", "calibration_offset",
    ]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert len(df) == 200


def test_pass_rate_bounded_and_consistent():
    df = _df()
    assert df["pass_rate"].between(0, 1).all()
    assert (df["batches_passed"] <= df["batches_tested"]).all()


def test_calibration_offset_drifts_upward():
    df = _df(days=300)
    early = df["calibration_offset"].head(50).mean()
    late = df["calibration_offset"].tail(50).mean()
    assert late > early                          # signed upward drift, not just magnitude
    assert df["calibration_offset"].iloc[-1] > 0


def test_oos_events_present_and_nonnegative():
    df = _df(days=365)
    assert (df["oos_events"] >= 0).all()
    assert df["oos_events"].sum() > 0


def test_deterministic_given_seed():
    pd.testing.assert_frame_equal(_df(seed=7), _df(seed=7))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_qc_generator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orgpulse.generators.qc'`

- [ ] **Step 3: Write minimal implementation**

`src/orgpulse/generators/qc.py`:
```python
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from orgpulse.generators.base import date_series


def generate_qc(start_date: date, days: int, rng: np.random.Generator) -> pd.DataFrame:
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
        # OOS events: rare, and more likely when miscalibrated (clustering).
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_qc_generator.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orgpulse/generators/qc.py tests/test_qc_generator.py
git commit -m "feat: QC/lab generator with calibration drift and OOS clustering"
```

---

### Task 5: Supply-chain generator

**Files:**
- Create: `src/orgpulse/generators/supply.py`
- Test: `tests/test_supply_generator.py`

**Interfaces:**
- Consumes: `make_rng`, `date_series`.
- Produces:
  - `generate_supply(start_date: date, days: int, rng: numpy.random.Generator) -> pandas.DataFrame`
  - **Wide format**, one row per date, columns:
    `date` (datetime64), `inventory_units` (int ≥ 0), `lead_time_days` (float ≥ 3.0),
    `supplier_reliability` (float in [0,1]).
  - Injected structure: inventory follows a reorder **sawtooth** (draw down daily, reorder when
    below `REORDER_POINT`); `lead_time_days` has occasional **spikes**; `supplier_reliability`
    is **seasonal** (annual sinusoid).
  - Module constant `REORDER_POINT: int = 200` exported.

- [ ] **Step 1: Write the failing test**

`tests/test_supply_generator.py`:
```python
from datetime import date

import numpy as np
import pandas as pd

from orgpulse.config import make_rng
from orgpulse.generators.supply import generate_supply, REORDER_POINT


def _df(days=365, seed=42):
    return generate_supply(date(2026, 1, 1), days, make_rng(seed))


def test_schema_and_types():
    df = _df()
    assert list(df.columns) == [
        "date", "inventory_units", "lead_time_days", "supplier_reliability",
    ]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_inventory_nonnegative_and_replenishes():
    df = _df()
    assert (df["inventory_units"] >= 0).all()
    # Sawtooth: inventory both draws down and jumps back up.
    diffs = np.diff(df["inventory_units"].to_numpy())
    assert (diffs < 0).any() and (diffs > 0).any()


def test_lead_time_has_floor_and_spikes():
    df = _df()
    assert (df["lead_time_days"] >= 3.0).all()
    assert df["lead_time_days"].max() > df["lead_time_days"].median() * 1.5


def test_reliability_bounded_and_seasonal():
    df = _df()
    assert df["supplier_reliability"].between(0, 1).all()
    # Genuine seasonality: reliability tracks the known annual sinusoid,
    # not just "has spread" (which pure noise would also satisfy).
    i = np.arange(len(df))
    seasonal_signal = np.sin(2 * np.pi * i / 365.0)
    corr = np.corrcoef(df["supplier_reliability"].to_numpy(), seasonal_signal)[0, 1]
    assert corr > 0.5


def test_deterministic_given_seed():
    pd.testing.assert_frame_equal(_df(seed=7), _df(seed=7))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_supply_generator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orgpulse.generators.supply'`

- [ ] **Step 3: Write minimal implementation**

`src/orgpulse/generators/supply.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_supply_generator.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orgpulse/generators/supply.py tests/test_supply_generator.py
git commit -m "feat: supply-chain generator with sawtooth inventory, lead-time spikes, seasonality"
```

---

### Task 6: `orgpulse generate` CLI

**Files:**
- Create: `src/orgpulse/cli.py`
- Test: `tests/test_cli_generate.py`

**Interfaces:**
- Consumes: all three generators, `OrgPulseConfig`, `make_rng`.
- Produces:
  - A Typer app `orgpulse.cli.app` with a `generate` command:
    `generate(days: int = 365, seed: int = 42, out: Path = Path("data"))`.
  - Writes three CSVs to `out/`: `clinical.csv`, `qc.csv`, `supply.csv`.
  - `orgpulse.cli.write_streams(out: Path, days: int, seed: int) -> dict[str, Path]`
    — the testable core (returns the paths written), called by the CLI command.

- [ ] **Step 1: Write the failing test**

`tests/test_cli_generate.py`:
```python
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from orgpulse.cli import app, write_streams


def test_write_streams_creates_three_csvs(tmp_path: Path):
    paths = write_streams(out=tmp_path, days=60, seed=42)
    assert set(paths) == {"clinical", "qc", "supply"}
    for name, p in paths.items():
        assert p.exists(), f"{name} not written"
        df = pd.read_csv(p)
        assert "date" in df.columns and len(df) > 0


def test_cli_generate_runs(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(app, ["generate", "--days", "30", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "clinical.csv").exists()
    assert (tmp_path / "qc.csv").exists()
    assert (tmp_path / "supply.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_generate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orgpulse.cli'`

- [ ] **Step 3: Write minimal implementation**

`src/orgpulse/cli.py`:
```python
from __future__ import annotations

from datetime import date
from pathlib import Path

import typer

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical
from orgpulse.generators.qc import generate_qc
from orgpulse.generators.supply import generate_supply

app = typer.Typer(help="OrgPulse AI — PharmaOps CLI")


def write_streams(out: Path, days: int, seed: int) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    start = date(2026, 1, 1)
    frames = {
        "clinical": generate_clinical(start, days, make_rng(seed)),
        "qc": generate_qc(start, days, make_rng(seed + 1)),
        "supply": generate_supply(start, days, make_rng(seed + 2)),
    }
    paths: dict[str, Path] = {}
    for name, df in frames.items():
        p = out / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
    return paths


@app.command()
def generate(
    days: int = typer.Option(365, help="Number of days to generate"),
    seed: int = typer.Option(42, help="Random seed"),
    out: Path = typer.Option(Path("data"), help="Output directory"),
) -> None:
    """Generate synthetic data for all three pharma-ops streams."""
    paths = write_streams(out=out, days=days, seed=seed)
    for name, p in paths.items():
        typer.echo(f"wrote {name}: {p}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_generate.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full suite + the CLI for real**

Run: `uv run pytest -q`
Expected: all tests pass.
Run: `uv run orgpulse generate --days 30 --out /tmp/orgpulse-smoke`
Expected: prints three `wrote ...` lines; the three CSVs exist.

- [ ] **Step 6: Commit**

```bash
git add src/orgpulse/cli.py tests/test_cli_generate.py
git commit -m "feat: orgpulse generate CLI writing all three streams"
```

---

## Self-Review

**Spec coverage (P0 + P1):**
- P0 scaffolding → Task 1 (pyproject, package, config). ✓
- P1 clinical generator (hero, injected breaks) → Task 3. ✓
- P1 QC generator (calibration drift, OOS clusters) → Task 4. ✓
- P1 supply generator (sawtooth, spikes, seasonality) → Task 5. ✓
- Determinism-given-seed constraint → tested in every generator task. ✓
- CLI entrypoint (`orgpulse generate`) → Task 6. ✓
- Deferred correctly to later plans: detection (P2), forecasting (P3), LangGraph/governance (P4/P5), Hermes (P6), Streamlit/story (P7/P8). ✓

**Placeholder scan:** No TBD/TODO; every code step has real, runnable code; every test step has real assertions. ✓

**Type consistency:** `make_rng` returns `np.random.Generator` (Task 1) and is consumed with that type everywhere. Every `generate_*` takes `(start_date, days, rng[, n_sites])` and returns a `pd.DataFrame` with a `datetime64` `date` column. `write_streams` returns `dict[str, Path]` matching its test. `PROTOCOL_AMENDMENT_DAY` / `REORDER_POINT` exported where tests import them. ✓
