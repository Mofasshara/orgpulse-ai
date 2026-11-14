from datetime import date

import numpy as np

from orgpulse.detect.validation import KPI_DOMAINS
from orgpulse.domains.pharma import PHARMA, make_pharma_domain


def test_pharma_domain_matches_the_previously_hardcoded_config():
    assert PHARMA.hero() == "daily_enrollment"
    # bounds must exactly equal the old hardcoded KPI_DOMAINS (unbounded KPIs simply absent there)
    for kpi, (lo, hi) in PHARMA.bounds().items():
        if kpi in KPI_DOMAINS:
            assert (lo, hi) == KPI_DOMAINS[kpi], kpi
        else:
            assert (lo, hi) == (None, None), kpi   # e.g. calibration_offset is unbounded
    assert PHARMA.analyst_system == "You are a clinical data quality analyst."
    assert PHARMA.recommend_system == "You are a quality assurance CAPA expert."


def test_pharma_extract_produces_the_expected_kpi_series():
    streams = PHARMA.generate(date(2026, 1, 1), 200, seed=42)
    assert set(streams) == {"clinical", "qc", "supply"}
    for spec in PHARMA.kpi_specs:
        arr = np.asarray(PHARMA.extract(streams, spec.name), dtype=float)
        assert arr.ndim == 1 and arr.size >= 120 and np.isfinite(arr).all(), spec.name
    # hero aggregates clinical to one value per day
    assert PHARMA.extract(streams, "daily_enrollment").shape[0] == 200


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


def test_pharma_lineage_extract_is_row_for_row_bound_to_the_raw_clinical_frame():
    # Codex r1 Critical: the engine-level coverage check only binds lineage to the forecast's
    # DATE window; it cannot, in general, detect a domain that drops SOME rows for an
    # otherwise-covered day. Pharma's own implementation must close that gap by construction:
    # prove lineage_extract returns the clinical frame's OWN columns, row-for-row and in
    # order, not an independently re-derived view that could silently diverge (e.g. omit one
    # site's rows while keeping the date present via another site's row).
    degraded = make_pharma_domain(paper_site_ids=frozenset({2}))
    streams = degraded.generate(date(2026, 1, 1), 200, seed=42)
    clinical = streams["clinical"]
    dates, tiers = degraded.lineage_extract(streams)
    assert list(dates) == list(clinical["date"].to_numpy())
    assert list(tiers) == list(clinical["source_tier"].to_numpy())


def test_pharma_lineage_caveat_is_a_nonempty_pharma_framed_string():
    text = PHARMA.lineage_caveat("daily_enrollment")
    assert isinstance(text, str) and text.strip()
