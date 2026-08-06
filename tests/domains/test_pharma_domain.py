from datetime import date

import numpy as np

from orgpulse.detect.validation import KPI_DOMAINS
from orgpulse.domains.pharma import PHARMA


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
