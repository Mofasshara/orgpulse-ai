"""Tests for the dashboard's lineage caption text (streamlit_app/lineage_caption.py).

Pulled into its own pure, importable module (not left inline in app.py) because app.py is a
live Streamlit script that executes page-rendering side effects at import time -- this keeps
the wording logic unit-testable without standing up an AppTest harness.
"""
from streamlit_app.lineage_caption import lineage_caption


def test_describes_widening_when_widen_factor_exceeds_one_even_if_not_low_trust():
    # Codex r1 Minor: nine clean days plus one manual day produces confidence~0.94,
    # low_trust=False, widen_factor~1.06 -- real widening occurred, but the old wording
    # branched on low_trust alone and claimed "no additional widening was needed".
    lineage = {"confidence": 0.94, "reason": "x", "low_trust": False, "widen_factor": 1.06}
    text = lineage_caption(lineage)
    assert "ADDITIONALLY widened" in text
    assert "No additional widening" not in text


def test_describes_no_widening_when_widen_factor_is_exactly_one():
    lineage = {"confidence": 1.0, "reason": "x", "low_trust": False, "widen_factor": 1.0}
    text = lineage_caption(lineage)
    assert "No additional widening" in text
    assert "ADDITIONALLY widened" not in text


def test_describes_widening_for_a_genuinely_low_trust_case_too():
    lineage = {"confidence": 0.4, "reason": "x", "low_trust": True, "widen_factor": 1.6}
    text = lineage_caption(lineage)
    assert "ADDITIONALLY widened ×1.60" in text
