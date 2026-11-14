"""Pure formatting for the dashboard's lineage caption text.

Split out of app.py (a live Streamlit script that executes page-rendering side effects at
import time) so this wording logic is unit-testable without an AppTest harness.
"""
from __future__ import annotations


def lineage_caption(lineage: dict) -> str:
    """One caption line describing a forecast's lineage confidence and widening.

    Whether widening is described is decided by `widen_factor` (the actual mechanical effect
    on the published interval), not by `low_trust` (the governance-gate flag) -- the two are
    related but not identical: a confidence just above the low_trust threshold still applies
    real, non-trivial widening and must say so, not claim "no additional widening was needed".
    """
    conf = lineage.get("confidence")
    widen = lineage.get("widen_factor", 1.0)
    widened = isinstance(widen, (int, float)) and widen > 1.0 + 1e-9
    base = f"🧬 **Lineage confidence: {conf:.0%}**: {lineage.get('reason', '')}. "
    if widened:
        return base + (
            f"The interval above is ADDITIONALLY widened ×{widen:.2f} for this "
            "data-provenance risk, on top of the statistical calibration."
        )
    return base + "No additional widening was needed; the source records are trusted."
