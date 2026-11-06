# Genealogy-Aware Forecast Trust Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give OrgPulse's forecast a data-lineage-derived confidence signal, so a forecast built on unverified or scanned-paper-sourced records gets a mechanically wider published interval and an explicitly downgraded, human-gated recommendation, instead of presenting a falsely confident number.

**Architecture:** One additive per-row provenance column in the clinical generator feeds a new, single-chokepoint confidence module. That confidence is consumed only inside the existing `forecast_node`/`recommendation_node` chokepoints (right next to the existing statistical calibration), never threaded through the tested detection/forecast internals. A domain that doesn't implement the new optional `Domain.lineage_extract` publishes no lineage claim at all; one that does and fails to deliver usable data falls back to the unsafe (zero-confidence) default.

**Tech Stack:** Python 3.11, pandas/numpy (lineage aggregation), existing Prophet/LangGraph pipeline, pytest. No new dependencies.

**Spec:** This plan's design was agreed directly in chat with the project owner (three alternatives presented, hybrid "additive column + new node-level chokepoint" approach approved) rather than a separate written spec document; the decisions below are the spec.

## Global Constraints

- TDD throughout: write the failing test before the implementation for every task.
- The full existing test suite (278 tests as of branch creation) must stay green after every task; re-run the FULL suite (`uv run pytest -q`), not just new tests, before considering any task done.
- No em dashes anywhere (docs, commit messages, code comments): use periods, commas, or colons.
- Default-config behavior (no `paper_site_ids`/`missing_source_site_ids`) must produce numerically identical `daily_enrollment`/`cumulative_enrollment`/`screen_fail_rate`/`dropout_rate` values to today, for the same seed. The existing byte-identical committed `runs/` artifacts WILL need re-seeding (the findings/stream schema is gaining new fields, exactly as every prior schema-changing phase in this repo's Change Log has re-seeded `runs/`); what must NOT change is the underlying generator math for the default config.
- Scope is the hero KPI (`daily_enrollment`) and the clinical stream only. Extending genealogy-awareness to QC/supply streams or other domains is explicitly deferred; say so in the roadmap, don't build it.
- Follow this repo's existing `ROADMAP.md` phase-table format and Change Log convention exactly (read the existing phases before adding Phase 9).
- Every new fail-safe default follows this repo's established rule: missing, corrupt, or unparseable lineage data reads as the LOWEST confidence, never silently as trusted.

## Review Focus

- An unrecognized/typo'd source-tier string (not one of the three known values) reaching the new confidence function or flowing through `forecast_node`'s integration must fail to the unsafe default, not crash and not be silently ignored. (Task 2 unit tests + a `forecast_node`-level integration test in Task 6.)
- When the hero's detection already ruled it `DATA_QUALITY` (forecast suppressed entirely, the existing detect/forecast parity guard), no `lineage` key must appear anywhere in the published forecast. Nothing should be computed or claimed about data that was never forecast. (Task 6.)
- A findings artifact from a domain that never implements `lineage_extract` (no `lineage` key at all in its forecast block) must still pass the dashboard's schema validator, since absence of a claim is not the same as a failed claim. (Task 8.)
- `source_tier` must actually change the Article-12 `input_data_hash` computed in `agent/tick.py` when the site-tier configuration changes, so two ticks with different lineage configs are genuinely distinguishable in the audit trail, not silently hashing the same. Verify this by reading `agent/tick.py`'s hash computation and writing a direct test, not by assuming it. (Task 10.)
- Overlapping or out-of-range site ids passed to `paper_site_ids`/`missing_source_site_ids` must raise `ValueError` at generation time, not silently misclassify a site or crash deeper in the pipeline. (Task 3.)

---

## Task 1: Register the roadmap phase

**Files:**
- Modify: `ROADMAP.md`

**Interfaces:**
- None (documentation only).

- [ ] **Step 1: Read the current `ROADMAP.md` Status Overview and the most recent phase table** to match its exact column layout and emoji status conventions.
- [ ] **Step 2: Append a new "Phase 9, Genealogy-Aware Forecast Trust Layer" section** after Phase 8, with one row per Task 2 to Task 11 below (Pending, with estimates), a one-line Goal, and a note that QC/supply streams and non-pharma domains are explicitly deferred.
- [ ] **Step 3: Update the Status Overview counts and Remaining estimate** at the top of the file to include the new Pending tasks.
- [ ] **Step 4: Append a Change Log row** recording that this phase was added, why (the Novartis/Dario conversation and the approved "additive column + new chokepoint" design), and that it extends rather than replaces the existing roadmap.
- [ ] **Step 5: Commit**

```bash
git add ROADMAP.md
git commit -m "docs: register Phase 9, genealogy-aware forecast trust layer"
```

---

## Task 2: Lineage confidence module

**Files:**
- Create: `src/orgpulse/lineage/__init__.py`
- Test: `tests/lineage/__init__.py` (empty), `tests/lineage/test_confidence.py`

**Interfaces:**
- Produces: `SourceTier(str, Enum)` with values `"system_of_record"`, `"manual_scanned"`, `"missing"`; `TIER_WEIGHT: dict[str, float]` mapping those three values to `1.0`, `0.4`, `0.0`; `LOW_TRUST_THRESHOLD: float = 0.8`; `MAX_EXTRA_WIDEN: float = 1.0`; `LineageResult` frozen dataclass with fields `confidence: float`, `tier_counts: dict[str, int]`, `reason: str`, `low_trust: bool`, `widen_factor: float`; `UNSAFE_LINEAGE: LineageResult` module-level constant (`confidence=0.0, tier_counts={}, reason="lineage data unavailable or corrupt", low_trust=True, widen_factor=2.0`); `lineage_confidence(dates, tiers) -> LineageResult`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/lineage/test_confidence.py
import pandas as pd
import pytest
from orgpulse.lineage import lineage_confidence, LOW_TRUST_THRESHOLD, UNSAFE_LINEAGE

def test_all_system_of_record_gives_full_confidence_no_widening():
    dates = pd.date_range("2026-01-01", periods=10)
    tiers = ["system_of_record"] * 10
    r = lineage_confidence(dates, tiers)
    assert r.confidence == 1.0
    assert r.low_trust is False
    assert r.widen_factor == 1.0

def test_per_day_minimum_then_mean_across_days():
    # 5 days with one manual_scanned record alongside a system_of_record record (day weight 0.4),
    # 5 pure system_of_record days (day weight 1.0) -> mean == 0.7
    degraded_days = pd.date_range("2026-01-01", periods=5)
    clean_days = pd.date_range("2026-01-06", periods=5)
    dates = list(degraded_days) * 2 + list(clean_days)
    tiers = (["manual_scanned"] * 5 + ["system_of_record"] * 5) + ["system_of_record"] * 5
    r = lineage_confidence(dates, tiers)
    assert r.confidence == pytest.approx(0.7)
    assert r.low_trust is True          # 0.7 < LOW_TRUST_THRESHOLD (0.8)
    assert r.widen_factor == pytest.approx(1.3)   # 1.0 + 1.0*(1-0.7)

@pytest.mark.parametrize("dates,tiers", [
    (None, None),
    ([], []),
    (pd.date_range("2026-01-01", periods=3), ["system_of_record"]),       # length mismatch
    (pd.date_range("2026-01-01", periods=2), ["system_of_record", "typo_tier"]),  # unknown tier
    ([pd.NaT, "2026-01-02"], ["system_of_record", "system_of_record"]),   # unparseable date
])
def test_missing_or_corrupt_lineage_data_is_unsafe(dates, tiers):
    r = lineage_confidence(dates, tiers)
    assert r == UNSAFE_LINEAGE

def test_low_trust_threshold_is_honest_about_its_boundary():
    # confidence exactly at the threshold is NOT low_trust (< is strict)
    dates = pd.date_range("2026-01-01", periods=1)
    tiers = ["system_of_record"]
    assert lineage_confidence(dates, tiers).confidence >= LOW_TRUST_THRESHOLD
```

- [ ] **Step 2: Run tests to verify they fail** (`ImportError`/`ModuleNotFoundError`: `uv run pytest tests/lineage/test_confidence.py -v`, expect collection failure).
- [ ] **Step 3: Implement `src/orgpulse/lineage/__init__.py`.** Algorithm for `lineage_confidence`: coerce `dates` to a `pd.Series` of datetimes and `tiers` to a `pd.Series` of strings; return `UNSAFE_LINEAGE` if either is `None`, lengths differ, either is empty, any date fails to parse, or any tier value is outside `TIER_WEIGHT`'s keys. Otherwise map each record to its tier weight, group by the normalized date and take the per-day MINIMUM weight (one untrustworthy record taints that whole day, mirroring the existing `validated_clinical_daily`'s any-NaN-in-a-group rule in `pipeline/nodes.py`), then take the mean of all days' minimums as `confidence`. `tier_counts` is the raw count of each of the three tier values across all input records (for display, not used in the formula). `low_trust = confidence < LOW_TRUST_THRESHOLD`. `widen_factor = 1.0 + MAX_EXTRA_WIDEN * (1.0 - confidence)`. `reason` is a short human-readable string naming the percentage of records that are not `system_of_record` and the resulting confidence (exact wording is the implementer's call; tests only check the numeric fields).
- [ ] **Step 4: Run tests to verify they pass.**
- [ ] **Step 5: Commit**

```bash
git add src/orgpulse/lineage tests/lineage
git commit -m "feat: add lineage confidence module (single chokepoint, fails unsafe on corrupt input)"
```

---

## Task 3: Additive `source_tier` column on the clinical generator

**Files:**
- Modify: `src/orgpulse/generators/clinical.py`
- Modify: `tests/test_clinical_generator.py`

**Interfaces:**
- Consumes: `SourceTier` from Task 2 (`src/orgpulse/lineage`).
- Produces: `generate_clinical(start_date, days, rng, n_sites=8, paper_site_ids: frozenset[int] = frozenset(), missing_source_site_ids: frozenset[int] = frozenset()) -> pd.DataFrame` with one new column `source_tier` appended after `dropout_rate`. Raises `ValueError` if any id in either set is `< 0` or `>= n_sites`, or if the two sets overlap.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_clinical_generator.py additions
def test_schema_and_types():
    df = _df()
    assert list(df.columns) == [
        "date", "site_id", "daily_enrollment", "cumulative_enrollment",
        "screen_fail_rate", "dropout_rate", "source_tier",
    ]
    # ... (existing dtype/nunique assertions unchanged)

def test_source_tier_defaults_to_system_of_record_for_all_sites():
    df = _df()
    assert (df["source_tier"] == "system_of_record").all()

def test_paper_and_missing_site_ids_tag_correctly():
    df = generate_clinical(
        date(2026, 1, 1), 60, make_rng(42), n_sites=8,
        paper_site_ids=frozenset({0}), missing_source_site_ids=frozenset({1}),
    )
    assert (df.loc[df.site_id == 0, "source_tier"] == "manual_scanned").all()
    assert (df.loc[df.site_id == 1, "source_tier"] == "missing").all()
    assert (df.loc[~df.site_id.isin([0, 1]), "source_tier"] == "system_of_record").all()

@pytest.mark.parametrize("paper,missing", [
    (frozenset({8}), frozenset()),          # out of range (n_sites=8, valid ids 0..7)
    (frozenset({-1}), frozenset()),         # negative
    (frozenset({0}), frozenset({0})),       # overlap
])
def test_rejects_bad_site_tier_config(paper, missing):
    with pytest.raises(ValueError):
        generate_clinical(date(2026, 1, 1), 30, make_rng(0), n_sites=8,
                           paper_site_ids=paper, missing_source_site_ids=missing)

def test_source_tier_config_does_not_change_any_numeric_column():
    plain = _df(seed=7)
    tagged = generate_clinical(
        date(2026, 1, 1), 200, make_rng(7), n_sites=8,
        paper_site_ids=frozenset({2}), missing_source_site_ids=frozenset({5}),
    )
    numeric_cols = ["date", "site_id", "daily_enrollment", "cumulative_enrollment",
                    "screen_fail_rate", "dropout_rate"]
    pd.testing.assert_frame_equal(plain[numeric_cols], tagged[numeric_cols])
```

- [ ] **Step 2: Run tests to verify they fail.**
- [ ] **Step 3: Implement.** Add the two new parameters; validate the two id sets (range + disjointness) before the generation loop and raise `ValueError` with a clear message on violation. Compute each site's tier once per site (not per day, it's constant): `SourceTier.MISSING.value` if `site_id in missing_source_site_ids`, else `SourceTier.MANUAL_SCANNED.value` if `site_id in paper_site_ids`, else `SourceTier.SYSTEM_OF_RECORD.value`. Add `"source_tier": tier` to each row dict in the existing per-site/per-day loop. Do not change any existing numeric column, RNG draw, or draw order.
- [ ] **Step 4: Run tests to verify they pass; also run the full existing file** (`uv run pytest tests/test_clinical_generator.py -v`) to confirm every pre-existing test (amendment surge, holiday dip, determinism, bad-args) still passes unmodified.
- [ ] **Step 5: Commit**

```bash
git add src/orgpulse/generators/clinical.py tests/test_clinical_generator.py
git commit -m "feat: add additive source_tier provenance column to the clinical generator"
```

---

## Task 4: `Domain.lineage_extract` and `Domain.lineage_caveat`

**Files:**
- Modify: `src/orgpulse/core/domain.py`
- Modify: `tests/core/test_domain.py`

**Interfaces:**
- Produces: new `Domain` fields `lineage_extract: Callable[[dict], tuple] | None = None` and `lineage_caveat: Callable[[str], str] = _default_lineage_caveat`; new module-level `_default_lineage_caveat(kpi: str) -> str` (domain-neutral default text, following the exact precedent of `_default_dq_recommendation`).
- Consumes: nothing new.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_domain.py additions
def test_lineage_extract_defaults_to_none_and_lineage_caveat_has_a_default():
    d = _domain([KpiSpec("x", is_hero=True)])
    assert d.lineage_extract is None
    assert isinstance(d.lineage_caveat("x"), str) and d.lineage_caveat("x").strip()

def test_lineage_extract_must_be_callable_or_none():
    with pytest.raises(ValueError):
        Domain(
            name="t", generate=lambda **k: {}, extract=lambda s, n: None,
            kpi_specs=(KpiSpec("x", is_hero=True),), analyst_system="a", recommend_system="r",
            lineage_extract="not callable",
        )

def test_lineage_caveat_must_be_callable():
    with pytest.raises(ValueError):
        Domain(
            name="t", generate=lambda **k: {}, extract=lambda s, n: None,
            kpi_specs=(KpiSpec("x", is_hero=True),), analyst_system="a", recommend_system="r",
            lineage_caveat="not callable",
        )
```

- [ ] **Step 2: Run tests to verify they fail.**
- [ ] **Step 3: Implement.** Add `_default_lineage_caveat(kpi: str) -> str` near `_default_dq_recommendation`, returning domain-neutral caveat text (e.g. naming that the forecast's source records could not be fully verified and recommending the governed recommendation be reviewed with that in mind). Add the two new `Domain` fields in the dataclass body (after `hero_frame`, before `offline_framing`, matching the existing field grouping). In `__post_init__`, extend the existing `hero_frame` callable-or-None check to also cover `lineage_extract`, and extend the existing `for cb in ("generate", "extract", "offline_recommend", "dq_recommendation")` callable-required loop to include `"lineage_caveat"`.
- [ ] **Step 4: Run tests to verify they pass; run the full `tests/core/` directory** to confirm no existing `Domain` construction test breaks.
- [ ] **Step 5: Commit**

```bash
git add src/orgpulse/core/domain.py tests/core/test_domain.py
git commit -m "feat: add optional Domain.lineage_extract and Domain.lineage_caveat capabilities"
```

---

## Task 5: Pharma's lineage implementation and the `make_pharma_domain` factory

**Files:**
- Modify: `src/orgpulse/domains/pharma/__init__.py`
- Modify: `tests/domains/test_pharma_domain.py`

**Interfaces:**
- Consumes: `Domain.lineage_extract`/`lineage_caveat` (Task 4); `generate_clinical`'s new parameters (Task 3).
- Produces: `make_pharma_domain(paper_site_ids: frozenset[int] = frozenset(), missing_source_site_ids: frozenset[int] = frozenset()) -> Domain`; `PHARMA = make_pharma_domain()` (unchanged singleton, now built by the factory).

- [ ] **Step 1: Write the failing tests**

```python
# tests/domains/test_pharma_domain.py additions
def test_pharma_lineage_extract_returns_aligned_dates_and_tiers():
    streams = PHARMA.generate(date(2026, 1, 1), 200, seed=42)
    dates, tiers = PHARMA.lineage_extract(streams)
    assert len(dates) == len(tiers) == len(streams["clinical"])
    assert set(tiers) == {"system_of_record"}   # default PHARMA has no degraded sites

def test_make_pharma_domain_with_paper_sites_tags_degraded_tier():
    degraded = make_pharma_domain(paper_site_ids=frozenset({0}))
    streams = degraded.generate(date(2026, 1, 1), 60, seed=42)
    _, tiers = degraded.lineage_extract(streams)
    assert "manual_scanned" in set(tiers)
    assert degraded.hero() == PHARMA.hero()   # same KPI config, only lineage differs

def test_pharma_lineage_caveat_is_a_nonempty_pharma_framed_string():
    text = PHARMA.lineage_caveat("daily_enrollment")
    assert isinstance(text, str) and text.strip()
```

- [ ] **Step 2: Run tests to verify they fail.**
- [ ] **Step 3: Implement.** Add `_lineage_extract(streams: dict) -> tuple[np.ndarray, np.ndarray]` returning `(streams["clinical"]["date"].to_numpy(), streams["clinical"]["source_tier"].to_numpy())`. Add `_pharma_lineage_caveat(kpi: str) -> str` with pharma-framed GxP language (parallel in tone to `_pharma_dq_recommendation`). Convert the module-level `PHARMA = Domain(...)` literal into `make_pharma_domain(paper_site_ids=frozenset(), missing_source_site_ids=frozenset()) -> Domain`, whose inner `_generate` closure passes the two new parameters through to `generate_clinical`, and which wires `lineage_extract=_lineage_extract, lineage_caveat=_pharma_lineage_caveat` into the returned `Domain`; keep every other field identical to today. Set `PHARMA = make_pharma_domain()` so the default singleton is unchanged.
- [ ] **Step 4: Run tests to verify they pass; run `tests/domains/` and `tests/core/test_second_domain.py`** to confirm the existing "pharma matches the previously-hardcoded config" test and the domain-agnostic proof suite still pass unchanged.
- [ ] **Step 5: Commit**

```bash
git add src/orgpulse/domains/pharma/__init__.py tests/domains/test_pharma_domain.py
git commit -m "feat: wire pharma lineage_extract/lineage_caveat; add make_pharma_domain factory"
```

---

## Task 6: `forecast_node` propagation

**Files:**
- Modify: `src/orgpulse/pipeline/nodes.py`
- Create: `tests/pipeline/test_nodes_lineage.py`

**Interfaces:**
- Consumes: `lineage_confidence`, `UNSAFE_LINEAGE` (Task 2); `domain.lineage_extract` (Task 4/5).
- Produces: `forecast_node` now writes `forecast[hero]["lineage"] = {"confidence": float, "tier_counts": dict, "reason": str, "low_trust": bool, "widen_factor": float}` whenever `domain.lineage_extract is not None` and the hero was actually forecast; omits the key entirely otherwise. The published `lower`/`upper` bounds are widened by `scale * widen_factor` in place of the current `scale`-only widening (the existing `calibration` block's `scale_factor` is unchanged and still reports the pure statistical factor).

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_nodes_lineage.py
from datetime import date
from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical
from orgpulse.generators.qc import generate_qc
from orgpulse.generators.supply import generate_supply
from orgpulse.domains.pharma import PHARMA, make_pharma_domain
from orgpulse.pipeline.nodes import detect_node, forecast_node

def _streams(domain, days=260, seed=42):
    return domain.generate(date(2026, 1, 1), days, seed)

def _state(domain, days=260, seed=42):
    streams = _streams(domain, days, seed)
    kfs = detect_node({"streams": streams}, domain)["kpi_findings"]
    return {"streams": streams, "kpi_findings": kfs}

def test_forecast_node_publishes_full_confidence_lineage_for_clean_pharma():
    out = forecast_node(_state(PHARMA), PHARMA)
    lineage = out["forecast"]["daily_enrollment"]["lineage"]
    assert lineage["confidence"] == 1.0
    assert lineage["low_trust"] is False
    assert lineage["widen_factor"] == 1.0

def test_forecast_node_widens_interval_when_lineage_confidence_is_low():
    clean_fc = forecast_node(_state(PHARMA), PHARMA)["forecast"]["daily_enrollment"]
    degraded = make_pharma_domain(paper_site_ids=frozenset({0, 1, 2}))
    degraded_fc = forecast_node(_state(degraded), degraded)["forecast"]["daily_enrollment"]
    assert degraded_fc["lineage"]["low_trust"] is True
    clean_width = clean_fc["h30"]["upper"] - clean_fc["h30"]["lower"]
    degraded_width = degraded_fc["h30"]["upper"] - degraded_fc["h30"]["lower"]
    assert degraded_width > clean_width   # direction, not a brittle exact magnitude

def test_forecast_node_omits_lineage_block_when_domain_has_no_lineage_extract():
    from dataclasses import replace
    no_lineage_domain = replace(PHARMA, lineage_extract=None)
    out = forecast_node(_state(no_lineage_domain), no_lineage_domain)
    assert "lineage" not in out["forecast"]["daily_enrollment"]

def test_forecast_node_falls_back_to_unsafe_lineage_when_lineage_extract_raises():
    from dataclasses import replace
    def _boom(streams):
        raise RuntimeError("boom")
    broken = replace(PHARMA, lineage_extract=_boom)
    out = forecast_node(_state(broken), broken)
    lineage = out["forecast"]["daily_enrollment"]["lineage"]
    assert lineage["confidence"] == 0.0
    assert lineage["low_trust"] is True

def test_forecast_node_publishes_no_lineage_key_when_hero_is_data_quality():
    # A hero series too short to assess (< MIN_ASSESSMENT_DAYS) is DATA_QUALITY; no forecast,
    # therefore no lineage claim either.
    streams = _streams(PHARMA, days=10)
    kfs = detect_node({"streams": streams}, PHARMA)["kpi_findings"]
    out = forecast_node({"streams": streams, "kpi_findings": kfs}, PHARMA)
    assert out["forecast"] == {}

def test_forecast_node_falls_back_to_unsafe_lineage_when_tiers_are_unrecognized():
    # Integration-level twin of the module-level unknown-tier test in Task 2: a domain whose
    # lineage_extract hands back a tier value outside the three known ones must still reach
    # forecast_node as the unsafe default, not raise and not silently pass through.
    from dataclasses import replace
    streams = _streams(PHARMA)
    def _bad_tiers(streams):
        dates, _ = PHARMA.lineage_extract(streams)
        return dates, ["not_a_real_tier"] * len(dates)
    bad_domain = replace(PHARMA, lineage_extract=_bad_tiers)
    kfs = detect_node({"streams": streams}, bad_domain)["kpi_findings"]
    out = forecast_node({"streams": streams, "kpi_findings": kfs}, bad_domain)
    lineage = out["forecast"]["daily_enrollment"]["lineage"]
    assert lineage["confidence"] == 0.0
    assert lineage["low_trust"] is True
```

- [ ] **Step 2: Run tests to verify they fail.**
- [ ] **Step 3: Implement.** Immediately after the existing `cal_meta` try/except block in `forecast_node` (both branches already set `scale`), compute: if `domain.lineage_extract is None`, skip lineage entirely (no key, `combined_scale = scale`). Otherwise call `domain.lineage_extract(streams)` inside a `try/except Exception` (symmetric with the existing `hero_frame` guard elsewhere in this function); on success pass its two return values into `lineage_confidence(...)`; on any exception use `UNSAFE_LINEAGE` directly. Build the `lineage` dict from the resulting `LineageResult`'s fields and set `combined_scale = scale * result.widen_factor`. Change the existing per-horizon loop to use `combined_scale` wherever it currently uses `scale` for the `lower`/`upper` computation (leave `yhat` untouched, leave `cal_meta["scale_factor"]` reporting the pure statistical `scale`, unchanged). Assign `forecast_result[hero]["lineage"] = lineage_dict` only when a lineage result was computed; never for a domain with `lineage_extract is None`.
- [ ] **Step 4: Run tests to verify they pass; run `tests/pipeline/` in full.**
- [ ] **Step 5: Commit**

```bash
git add src/orgpulse/pipeline/nodes.py tests/pipeline/test_nodes_lineage.py
git commit -m "feat: forecast_node publishes and applies lineage confidence widening"
```

---

## Task 7: `recommendation_node` downgrade

**Files:**
- Modify: `src/orgpulse/pipeline/nodes.py`
- Modify: `tests/pipeline/test_nodes_lineage.py`

**Interfaces:**
- Consumes: `forecast[hero]["lineage"]["low_trust"]` (Task 6); `domain.lineage_caveat` (Task 4/5).
- Produces: `recommendation_node` forces `alert=True` whenever the hero's lineage is low-trust (in addition to the existing `overall_severity in ("CRITICAL","DATA_QUALITY")` condition), and always appends exactly one additional `Recommendation(kpi=hero, capa_frame="LINEAGE-RISK", ...)` in that case, independent of the existing anomaly/data-quality recommendation logic.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_nodes_lineage.py additions
from orgpulse.pipeline.nodes import recommendation_node
from orgpulse.pipeline.llm import make_llm
from orgpulse.pipeline.state import KpiFinding

def _calm_state_with_lineage(low_trust: bool):
    return {
        "kpi_findings": [KpiFinding(kpi="daily_enrollment", severity="SAFE", detail="d", latest_value=1.0)],
        "overall_severity": "SAFE",
        "forecast": {"daily_enrollment": {"lineage": {"low_trust": low_trust, "confidence": 0.3 if low_trust else 1.0}}},
    }

def test_low_trust_hero_forces_alert_and_adds_lineage_risk_recommendation():
    llm = make_llm(offline=True, domain=PHARMA)
    out = recommendation_node(_calm_state_with_lineage(True), llm, PHARMA)
    assert out["alert"] is True
    matches = [r for r in out["recommendations"] if r.capa_frame == "LINEAGE-RISK" and r.kpi == "daily_enrollment"]
    assert len(matches) == 1 and matches[0].text.strip()

def test_high_trust_hero_does_not_add_lineage_risk_recommendation():
    llm = make_llm(offline=True, domain=PHARMA)
    out = recommendation_node(_calm_state_with_lineage(False), llm, PHARMA)
    assert out["alert"] is False
    assert not any(r.capa_frame == "LINEAGE-RISK" for r in out["recommendations"])

def test_lineage_risk_recommendation_coexists_with_an_anomaly_recommendation():
    llm = make_llm(offline=True, domain=PHARMA)
    state = _calm_state_with_lineage(True)
    state["kpi_findings"] = [KpiFinding(kpi="daily_enrollment", severity="WARNING", detail="d", latest_value=1.0)]
    state["overall_severity"] = "WARNING"
    out = recommendation_node(state, llm, PHARMA)
    kinds = [r.capa_frame for r in out["recommendations"] if r.kpi == "daily_enrollment"]
    assert "LINEAGE-RISK" in kinds and len(kinds) == 2   # both the anomaly rec and the lineage rec
```

- [ ] **Step 2: Run tests to verify they fail.**
- [ ] **Step 3: Implement.** Near the top of `recommendation_node`, compute `hero = domain.hero()` and `hero_low_trust = bool(state.get("forecast", {}).get(hero, {}).get("lineage", {}).get("low_trust", False))`. Change the existing `alert = overall_severity in ("CRITICAL", "DATA_QUALITY")` line to `or hero_low_trust`. After the existing anomaly/data-quality recommendation-building loops (so it never interferes with their kpi-membership logic), if `hero_low_trust`: call `domain.lineage_caveat(hero)` inside a `try/except Exception`, falling back to `_default_lineage_caveat(hero)` on raise or on a non-string/blank return (same guard pattern already used for `dq_recommendation`), and append one `Recommendation(kpi=hero, text=caveat, capa_frame="LINEAGE-RISK", disclaimer=HUMAN_OVERSIGHT_DISCLAIMER, review_status="PENDING_HUMAN_REVIEW")`.
- [ ] **Step 4: Run tests to verify they pass; run `tests/pipeline/` and `tests/agent/` in full** (the tick path calls these nodes too).
- [ ] **Step 5: Commit**

```bash
git add src/orgpulse/pipeline/nodes.py tests/pipeline/test_nodes_lineage.py
git commit -m "feat: recommendation_node downgrades and alerts on low lineage trust"
```

---

## Task 8: Dashboard schema validation

**Files:**
- Modify: `streamlit_app/data.py`
- Modify: `tests/test_dashboard_data.py`

**Interfaces:**
- Consumes: the `forecast[hero]["lineage"]` shape produced by Task 6/7.
- Produces: `_findings_schema_reason` validates the `lineage` block when present, accepts its absence, and rejects the artifact if `low_trust` is True but `alert` is false or no `LINEAGE-RISK` recommendation exists for that KPI.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dashboard_data.py additions
from streamlit_app.data import _findings_schema_reason

def _base_findings(**overrides):
    fj = {
        "run_id": "r", "generated_at": "2026-01-01T00:00:00", "overall_severity": "SAFE",
        "alert": False,
        "kpi_findings": [{"kpi": "daily_enrollment", "severity": "SAFE", "detail": "d", "latest_value": 1.0}],
        "forecast": {}, "analysis": "a", "recommendations": [],
    }
    fj.update(overrides)
    return fj

def test_schema_accepts_absent_lineage_block():
    fj = _base_findings(forecast={"daily_enrollment": {"h30": {"yhat": 1, "lower": 0, "upper": 2},
                                                         "h60": {"yhat": 1, "lower": 0, "upper": 2},
                                                         "h90": {"yhat": 1, "lower": 0, "upper": 2}}})
    assert _findings_schema_reason(fj) is None

def test_schema_rejects_confidence_out_of_range():
    hero = {"h30": {"yhat": 1, "lower": 0, "upper": 2}, "h60": {"yhat": 1, "lower": 0, "upper": 2},
            "h90": {"yhat": 1, "lower": 0, "upper": 2},
            "lineage": {"confidence": 1.5, "tier_counts": {}, "reason": "x", "low_trust": False, "widen_factor": 1.0}}
    fj = _base_findings(forecast={"daily_enrollment": hero})
    assert _findings_schema_reason(fj) is not None

def test_schema_rejects_low_trust_without_alert_or_without_lineage_risk_rec():
    hero = {"h30": {"yhat": 1, "lower": 0, "upper": 2}, "h60": {"yhat": 1, "lower": 0, "upper": 2},
            "h90": {"yhat": 1, "lower": 0, "upper": 2},
            "lineage": {"confidence": 0.1, "tier_counts": {}, "reason": "x", "low_trust": True, "widen_factor": 2.0}}
    # alert True but no LINEAGE-RISK rec
    fj = _base_findings(forecast={"daily_enrollment": hero}, alert=True)
    assert _findings_schema_reason(fj) is not None
    # has the rec but alert is False
    fj2 = _base_findings(
        forecast={"daily_enrollment": hero}, alert=False,
        recommendations=[{"kpi": "daily_enrollment", "text": "t", "capa_frame": "LINEAGE-RISK",
                           "review_status": "PENDING_HUMAN_REVIEW", "disclaimer": "d"}],
    )
    assert _findings_schema_reason(fj2) is not None

def test_schema_accepts_consistent_low_trust_artifact():
    hero = {"h30": {"yhat": 1, "lower": 0, "upper": 2}, "h60": {"yhat": 1, "lower": 0, "upper": 2},
            "h90": {"yhat": 1, "lower": 0, "upper": 2},
            "lineage": {"confidence": 0.1, "tier_counts": {}, "reason": "x", "low_trust": True, "widen_factor": 2.0}}
    fj = _base_findings(
        forecast={"daily_enrollment": hero}, alert=True,
        recommendations=[{"kpi": "daily_enrollment", "text": "t", "capa_frame": "LINEAGE-RISK",
                           "review_status": "PENDING_HUMAN_REVIEW", "disclaimer": "d"}],
    )
    assert _findings_schema_reason(fj) is None

def test_verify_run_true_on_a_real_degraded_lineage_tick(tmp_path):
    from orgpulse.agent.baseline_memory import BaselineMemory
    from orgpulse.agent.tick import run_tick
    from orgpulse.domains.pharma import make_pharma_domain
    from streamlit_app import data
    runs = tmp_path / "runs"
    mem = BaselineMemory(tmp_path / "m.json")
    degraded = make_pharma_domain(paper_site_ids=frozenset({0, 1, 2}))
    run_tick(mem, 0, out_root=runs, now="2026-08-04T07:00:00", offline=True, domain=degraded)
    ok, reason = data.verify_run(runs / "latest")
    assert ok, reason
```

- [ ] **Step 2: Run tests to verify they fail.**
- [ ] **Step 3: Implement.** In `_findings_schema_reason`, inside the existing `if fc:` block, right after the existing `cal = hero.get("calibration")` validation, initialize `hero_low_trust = False` (declare this and `hero_key` before the `if fc:` block so they're always in scope later) and add: if `hero.get("lineage")` is not `None`, validate it's a dict with `confidence` a finite number in `[0,1]`, `tier_counts` a dict of non-negative ints, `reason` a non-empty string, `low_trust` a bool, `widen_factor` a finite number `>= 1.0`; on any violation return a reason string (following the exact style of the surrounding checks). Set `hero_low_trust = lineage.get("low_trust", False)` when the block is valid. After the existing `recs`/`alert` cross-field checks near the end of the function, add: if `hero_low_trust` is True, `fj["alert"]` must be `True` and `recs` must contain an entry with `kpi == hero_key and capa_frame == "LINEAGE-RISK"`; otherwise return a reason string.
- [ ] **Step 4: Run tests to verify they pass; run `tests/test_dashboard_data.py` in full** (confirms the anti-drift pairing with `orgpulse.governance.audit` still holds).
- [ ] **Step 5: Commit**

```bash
git add streamlit_app/data.py tests/test_dashboard_data.py
git commit -m "feat: dashboard schema validates lineage block and its alert/recommendation consistency"
```

---

## Task 9: Dashboard rendering

**Files:**
- Modify: `streamlit_app/app.py`

**Interfaces:**
- Consumes: `forecast[hero]["lineage"]` (may be absent).
- Produces: a caption near the existing forecast/calibration caption showing the lineage confidence and whether the interval was additionally widened; `LINEAGE-RISK` recommendations render through the existing generic recommendation loop (no structural change needed there, since no other `capa_frame` value gets distinct styling today either).

- [ ] **Step 1: No automated test exists for `app.py` today (no AppTest harness in this repo); verification for this task is the manual browser check in Step 3, per this project's own rule that UI changes must be exercised in a running browser before being called done.**
- [ ] **Step 2: Implement.** In the `with left:` forecast block of `streamlit_app/app.py`, immediately after the existing `st.caption(cal_txt + ...)` call, add: if `fc.get("lineage")` is present, render a caption naming the confidence percentage, the plain-language reason string already computed by the compute plane, and (when `low_trust`) the additional widen factor applied; when not `low_trust`, say plainly that no additional widening was needed.
- [ ] **Step 3: Manually verify.** Regenerate `runs/latest` for a degraded-lineage tick locally (`uv run python -c "..."` calling `run_tick` with `make_pharma_domain(paper_site_ids=frozenset({0,1,2}))`, or wait for Task 10's seeded degraded tick), run `uv run --extra dashboard streamlit run streamlit_app/app.py`, and visually confirm: the lineage caption appears, the forecast fan is visibly wider than a clean-lineage run's, the `LINEAGE-RISK` recommendation card appears, and the alert banner fires. Then point `ORGPULSE_RUNS` at a clean-lineage run and confirm the caption shows full confidence with no widening and no `LINEAGE-RISK` card.
- [ ] **Step 4: Commit**

```bash
git add streamlit_app/app.py
git commit -m "feat: dashboard shows lineage confidence and widening in the forecast panel"
```

---

## Task 10: Seeded demo pair and audit-hash regression check

**Files:**
- Modify: `scripts/seed_runs.py`
- Create: `tests/agent/test_tick_lineage.py`

**Interfaces:**
- Consumes: `make_pharma_domain` (Task 5).
- Produces: `runs/` regenerated with 7 ticks (the existing 6 clean-lineage monthly ticks plus one new degraded-lineage tick); a direct test proving `source_tier` changes `agent.tick.run_tick`'s recorded `input_data_hash`.

- [ ] **Step 1: Write the failing regression test**

```python
# tests/agent/test_tick_lineage.py
import json
from orgpulse.agent.baseline_memory import BaselineMemory
from orgpulse.agent.tick import run_tick
from orgpulse.domains.pharma import PHARMA, make_pharma_domain

def _input_hash(runs_root, run_id_prefix):
    entries = [json.loads(l) for l in (runs_root / "latest" / "audit.jsonl").read_text().splitlines()]
    return next(e["payload"]["input_data_hash"] for e in entries if e["kind"] == "run")

def test_source_tier_changes_the_recorded_input_data_hash(tmp_path):
    clean_runs, degraded_runs = tmp_path / "clean", tmp_path / "degraded"
    run_tick(BaselineMemory(tmp_path / "m1.json"), 0, out_root=clean_runs,
              now="2026-08-04T07:00:00", offline=True, domain=PHARMA)
    degraded = make_pharma_domain(paper_site_ids=frozenset({0, 1, 2}))
    run_tick(BaselineMemory(tmp_path / "m2.json"), 0, out_root=degraded_runs,
              now="2026-08-04T07:00:00", offline=True, domain=degraded)
    assert _input_hash(clean_runs, "tick-0") != _input_hash(degraded_runs, "tick-0")

def test_degraded_lineage_tick_produces_a_low_trust_alerting_run(tmp_path):
    degraded = make_pharma_domain(paper_site_ids=frozenset({0, 1, 2}))
    findings = run_tick(BaselineMemory(tmp_path / "m.json"), 0, out_root=tmp_path / "runs",
                          now="2026-08-04T07:00:00", offline=True, domain=degraded)
    assert findings.alert is True
    assert any(r.capa_frame == "LINEAGE-RISK" for r in findings.recommendations)
    assert findings.forecast["daily_enrollment"]["lineage"]["low_trust"] is True
```

- [ ] **Step 2: Run tests to verify they fail (or pass trivially wrong if Task 5/6/7 have a latent bug), then confirm against the implementation** (`uv run pytest tests/agent/test_tick_lineage.py -v`).
- [ ] **Step 3: Modify `scripts/seed_runs.py`.** Import `make_pharma_domain`. Add a 7th fixed date to `_DATES` (e.g. `"2026-09-01T07:00:00"`). In the seeding loop, construct `domain = make_pharma_domain(paper_site_ids=frozenset({0, 1, 2}) if tick == len(_DATES) - 1 else frozenset())` for every tick and pass `domain=domain` to `run_tick` (the first six ticks get an all-system-of-record domain, functionally identical to `PHARMA`; the seventh gets the degraded one). Update the `_seed_dir_names` call sites if `n` can now exceed 6 (it already parameterizes over `len(_DATES)`, so this should need no further change; verify by reading the function).
- [ ] **Step 4: Regenerate the committed runs.** Run `uv run python scripts/seed_runs.py --force`. Confirm the printed summary shows 7 ticks, the last one `alert=True`. Run `uv run pytest -q` and confirm the full suite (278 + all new tests from Tasks 2 to 10) passes.
- [ ] **Step 5: Commit**

```bash
git add scripts/seed_runs.py tests/agent/test_tick_lineage.py runs/ baseline_memory.json
git commit -m "feat: seed a clean/degraded lineage demo pair; prove source_tier changes the audit input hash"
```

---

## Task 11: Honest documentation and roadmap close-out

**Files:**
- Modify: `docs/CASE_STUDY.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `README.md`
- Modify: `ROADMAP.md`

**Interfaces:** None (documentation only).

- [ ] **Step 1: Add a short section to `docs/CASE_STUDY.md`** (near the existing "Honest results" section, matching its voice) stating plainly: what this phase proves (a lineage/provenance signal mechanically widens a published forecast interval and gates the recommendation, demonstrated with a seeded clean/degraded scenario pair) and what it explicitly does not prove (the tier weights `1.0/0.4/0.0` and the widen-factor formula are illustrative choices, not empirically validated against real clinical operations; the mechanism covers the hero KPI and the clinical stream only in this phase; it is a demonstration of the mechanism, not a validated risk model).
- [ ] **Step 2: Add a row to `docs/ARCHITECTURE.md`'s "Domain configuration" table** for the new `lineage_extract`/`lineage_caveat` capability, and one sentence in the DAG description noting that `forecast` now optionally publishes a lineage confidence alongside the statistical calibration.
- [ ] **Step 3: Add one or two sentences to `README.md`** (in the "What it does" bullet list) naming the new capability in the same honest tone as the existing bullets.
- [ ] **Step 4: Update `ROADMAP.md`.** Mark every Phase 9 task Done with timestamps (per this repo's existing auto-update convention), update the Status Overview counts, and append a Change Log row summarizing the phase: what was added, why (the Dario/Novartis conversation), and that QC/supply streams and non-pharma domains are explicitly deferred.
- [ ] **Step 5: Run the full suite one last time** (`uv run pytest -q`) and record the new total test count in `README.md`'s "Tests" section and `docs/CASE_STUDY.md`'s "By the numbers" table.
- [ ] **Step 6: Commit**

```bash
git add docs/CASE_STUDY.md docs/ARCHITECTURE.md README.md ROADMAP.md
git commit -m "docs: document the genealogy-aware forecast trust layer honestly; close out Phase 9"
```

---
