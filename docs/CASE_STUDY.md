# Case Study — OrgPulse AI: an autonomous agent you can run in regulated pharma

## The problem

Pharmaceutical operations generate continuous signals — patient enrollment, QC pass rates and
instrument calibration, supplier lead times — and the expensive failures are the ones nobody
notices for weeks: an enrollment surge after a protocol amendment, a QC instrument drifting out
of calibration, a supplier quietly slipping. The obvious answer is "an AI that watches
everything and flags problems." The reason that answer keeps failing in pharma isn't the
detection — it's everything around it:

- A model that **silently treats missing/corrupt data as "all clear"** is worse than no model.
- A recommendation with **no audit trail** is unusable under GxP and the EU AI Act.
- An agent that **takes action on its own** is a non-starter in a regulated environment.
- A demo that needs a **running model and an API key** to show anything is dead on arrival.

So the interesting engineering problem wasn't "can you detect an anomaly." It was: **can you
build an agent that runs itself, and is still safe to run where regulators are watching?**
Autonomy and compliance, fused — that's the wedge.

## The architecture (and the one decision that shaped everything)

The system splits into two planes that share nothing but a directory of committed artifacts
(full diagram in [ARCHITECTURE.md](ARCHITECTURE.md)):

- a **compute plane** — a Hermes-scheduled agent that runs the full `detect → forecast →
  analyst → recommend` LangGraph DAG locally and writes small, tamper-evident run artifacts;
- a **presentation plane** — a Streamlit dashboard that *only reads* those artifacts.

That split was the highest-leverage decision in the project. Because the dashboard never
imports the compute stack, it deploys to Streamlit Cloud with three dependencies
(`streamlit`, `pandas`, `plotly`), holds no API key, and is **always-populated** (once
deployed) — it renders
committed results, so there's nothing to spin up and nothing to fail on first load. The agent's
autonomy lives entirely on the compute plane; the public surface is a pure, cheap view.

The "autonomous" part is real: each daily tick detects the **recent 30-day period** against an
**established 120-day baseline it remembered from the previous run**, then writes the updated
baseline back. It has memory across runs, not just a schedule.

## Where governance lives: everywhere, on purpose

Most projects bolt compliance on at the end as a disclaimer. Here it's a cross-cut:

- **Human oversight (Art. 14)** is *architectural*, not advisory: the strongest thing the
  system can emit is a recommendation stamped `PENDING_HUMAN_REVIEW`. It cannot act.
- **Logging (Art. 12)** is a **hash-chained** `audit.jsonl`: the run entry records the input
  lineage — the data-generation `seed` and parameters, a `input_data_hash` of the actual input
  frames, and (for a remembered-baseline tick) a `baseline_input_hash` of the exact baseline that
  drove detection — plus output-integrity hashes (`findings_hash`, `report_hash`), the
  model+version, and a timestamp. The dashboard doesn't just re-verify the chain — it verifies
  that the `findings.json` and `report.md` on screen still hash to what the chain attests to (so
  editing either is caught) and that the recommendation count matches. The standalone verifier is
  proven equivalent to the compute-plane implementation in tests, and both fail closed on
  missing/empty/malformed logs, so the presentation plane can attest to integrity without
  importing the system it's attesting to. **Threat model, stated plainly:**
  an unkeyed chain in a writable file catches accidental corruption and naïve tampering; it does
  not stop an adversary who rewrites the whole file and recomputes the chain — that needs keyed
  signing or an external anchor, which is noted as future work, not claimed as done.
- **Annex III self-classification** is machine-readable and rendered to a document mapping
  Articles 9/11/12/13/14 to concrete mechanisms in the code.

## The part most people skip: telling the truth about the model

The forecaster is deterministic Prophet with a rolling-origin backtest, and the honest result
is that **it does not beat a naïve baseline** (MASE ≈ 1.4, MAPE ≈ 17%). Rather than hide that,
the project foregrounds it — because the deliverable isn't point accuracy, it's a **calibrated
uncertainty interval** plus the audit trail around it. Raw Prophet intervals were
over-confident (~70% coverage against a nominal 90%); an interval-calibration factor is
**applied to the published bounds** (not merely diagnosed) to widen them toward nominal. That
calibration is fit in-sample, so the case study says so out loud: the widened interval's
*out-of-sample* coverage is not claimed — a production system would use split-conformal
prediction. The anomaly detector gets the same treatment: CUSUM thresholds are calibrated to a
documented ~1% in-control false-alarm rate *per window length* via a Monte-Carlo table, with a
≥60-sample baseline floor.

That discipline was enforced, not assumed. The codebase went through **seven rounds of external
adversarial review**, each of which found real defects — inert data corruptions, tests that
fired on the wrong signal, false-alarm calibration that collapsed on short baselines, fail-open
SAFE paths, and impossible-value gaps — the last of which forced a root-cause redesign rather than
another patch. The
lesson that stuck: *internal reviews that measure comfortable margins miss what mutation-style
and multi-seed tests catch.* A later defect proved the point — seeding the demo history exposed
that the agent's remembered baseline was being fit on a 30-sample window, below CUSUM's
60-sample floor, silently degrading every post-bootstrap run to `DATA_QUALITY`. The fix was at
the root (decouple the baseline window from the detection window), locked with a regression test
— not a patch over the symptom.

## What it demonstrates

- **Agentic autonomy** with genuine cross-run memory and self-scheduling.
- **Statistically honest ML** — calibrated intervals, calibrated false-alarm rates, and results
  reported straight, including where the model loses to naïve.
- **LLM reasoning** with a keyless deterministic fallback, so the system is fully testable and
  demonstrable without a key or network, and the audit trail records which model spoke.
- **Regulatory engineering** — EU AI Act Annex III and GxP treated as a first-class,
  cross-cutting concern that shaped the architecture, not a paragraph at the bottom of a README.

## By the numbers

| | |
|---|---|
| Data streams | 3 (clinical hero + QC + supply) |
| Analysis DAG | 4 nodes (LangGraph) |
| Forecast horizons | 30 / 60 / 90 days, 90% in-sample-calibrated interval |
| Tests | 152, keyless & offline |
| External adversarial review rounds | 7 (all findings fixed & reproduction-verified; round 5 forced a redesign) |
| Deployed dashboard deps | 3 (`streamlit`, `pandas`, `plotly`) |
