from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from orgpulse.agent.baseline_memory import BaselineMemory
from orgpulse.core.domain import DetectionMode, Domain
from orgpulse.detect.baseline import fit_baseline
from orgpulse.detect.cusum import _MIN_BASELINE_N, cusum_detect
from orgpulse.detect.validation import (
    MIN_ASSESSMENT_DAYS,
    baseline_domain_feasible,
    kpi_series_reason,
)
from orgpulse.detect.orchestrator import overall_severity
from orgpulse.detect.rules import datemath_detect, threshold_detect
from orgpulse.detect.types import Severity
from orgpulse.detect.zscore import zscore_detect
from orgpulse.domains.pharma import PHARMA
from orgpulse.governance.audit import HashChainedAuditLog
from orgpulse.pipeline.graph import render_report
from orgpulse.pipeline.llm import make_llm
from orgpulse.pipeline.nodes import analyst_node, forecast_node, recommendation_node
from orgpulse.pipeline.state import Findings, KpiFinding, data_quality_finding


def run_tick(
    memory: BaselineMemory,
    tick_number: int,
    *,
    days_per_period: int = 30,
    baseline_days: int = 120,
    base_days: int = 200,
    seed: int = 42,
    offline: bool = True,
    out_root: Path = Path("runs"),
    now: str | None = None,
    domain: Domain = PHARMA,
) -> Findings:
    """One autonomous tick: detect the recent PERIOD (``days_per_period``) against the
    REMEMBERED baseline (bootstrap on the first tick), forecast + reason, write run
    artifacts, and save an established trailing baseline (``baseline_days``) back to
    memory for the next tick.

    The baseline window is deliberately longer than the detection window: CUSUM's
    false-alarm calibration requires a baseline of at least 60 samples, so fitting the
    baseline on the short detection window alone would make every remembered baseline
    unusable (degrading every post-bootstrap tick to DATA_QUALITY).

    Parameters are validated so the statistical preconditions the baseline window depends
    on cannot be silently violated: ``days_per_period > 0``, ``baseline_days >= 120`` (the
    minimum-history contract), and ``base_days >= max(baseline_days, days_per_period)`` so the
    very first tick has both a full detection window and a full baseline window (matching the run
    pipeline's rigor: an ingested stream shorter than ``baseline_days`` is DATA_QUALITY, not a
    free bootstrap SAFE). These preconditions apply only when the domain has a TIMESERIES KPI; a
    pure rule-mode domain (all THRESHOLD/DATEMATH) assesses the latest value only and needs just
    one readable row.
    """
    for name, val in (("tick_number", tick_number), ("days_per_period", days_per_period),
                      ("baseline_days", baseline_days), ("base_days", base_days)):
        if not isinstance(val, int) or isinstance(val, bool):
            raise TypeError(f"{name} must be an int, got {type(val).__name__}: {val!r}")
    if tick_number < 0:
        raise ValueError(f"tick_number must be >= 0, got {tick_number}")
    if days_per_period <= 0:
        raise ValueError(f"days_per_period must be > 0, got {days_per_period}")
    # The 120-day baseline contract exists ONLY for TIMESERIES KPIs (z-score + CUSUM need it).
    # A pure rule-mode domain (all THRESHOLD/DATEMATH) assesses the latest value only
    # (`min_history_for` == 1), so forcing baseline_days >= 120 on it is wrong — and was a
    # run-vs-tick divergence (run_pipeline already handled a one-point rule domain). Enforce the
    # full contract when any KPI is TIMESERIES; otherwise require just a single readable row.
    has_timeseries = any(s.detection_mode is DetectionMode.TIMESERIES for s in domain.kpi_specs)
    if has_timeseries:
        if baseline_days < MIN_ASSESSMENT_DAYS:
            raise ValueError(
                f"baseline_days must be >= {MIN_ASSESSMENT_DAYS} (the shared minimum-history contract; "
                f"the run pipeline needs the same 120 to make an assessment), got {baseline_days}"
            )
        if base_days < max(baseline_days, days_per_period):
            raise ValueError(
                f"base_days must be >= max(baseline_days={baseline_days}, days_per_period={days_per_period}) "
                f"so tick 0 has a full detection window and a usable baseline, got {base_days}"
            )
    elif base_days < 1:
        raise ValueError(f"base_days must be >= 1 (need at least the latest value), got {base_days}")
    days = base_days + tick_number * days_per_period
    start = date(2026, 1, 1)
    # generate() is a Domain-supplied callable. Validate its output BEFORE any assessment or writing
    # so a malformed generator (returns None/non-dict, empty, or non-DataFrame members, or raises)
    # fails fast with a clear error and leaves NO partial, unverifiable run directory behind.
    try:
        streams = domain.generate(start, days, seed)
    except Exception as exc:
        raise ValueError(f"domain {domain.name!r} generate() failed: {exc}") from exc
    if not isinstance(streams, dict) or not streams or not all(
        isinstance(df, pd.DataFrame) for df in streams.values()
    ):
        raise ValueError(
            f"domain {domain.name!r} generate() must return a non-empty dict of DataFrames, "
            f"got {type(streams).__name__}"
        )
    bounds = domain.bounds()

    remembered = memory.load()
    n = days_per_period
    findings: list[KpiFinding] = []
    new_baselines = {}
    used_bootstrap = False

    for spec in domain.kpi_specs:
        kpi = spec.name
        # Extraction goes through the domain (clinical is validated per-row before aggregation,
        # so per-site corruption masked by the daily sum is caught). Any structural failure
        # (missing stream/column, ragged data) is DATA_QUALITY, not a crash.
        try:
            series = domain.extract(streams, kpi)
        except (KeyError, ValueError, TypeError, IndexError, AttributeError, OverflowError) as exc:
            findings.append(data_quality_finding(kpi, f"unusable input ({exc})"))
            continue
        # Validate the WHOLE ingested series through the shared gate (numeric, 1-D, >= a full
        # baseline window, all-finite, in-domain). Requiring baseline_days (not just the
        # detection window) matches the run pipeline's minimum and closes the run/tick
        # divergence where a too-short stream read a free bootstrap SAFE. A violation
        # ANYWHERE — including in the baseline history — is DATA_QUALITY, and NO baseline is
        # persisted from corrupt data (a bad baseline would poison the next tick's detection).
        # Timeseries needs a full baseline window; threshold/deadline checks need only the latest.
        min_len = baseline_days if spec.detection_mode is DetectionMode.TIMESERIES else 1
        reason = kpi_series_reason(kpi, series, min_len=min_len, bounds=bounds)
        if reason is not None:
            findings.append(data_quality_finding(kpi, reason))
            continue
        s = np.asarray(series, dtype=float)

        # Non-timeseries modes assess the LATEST value directly — no baseline, no remembered state.
        if spec.detection_mode is not DetectionMode.TIMESERIES:
            r = (threshold_detect(s, spec.thresholds, kpi) if spec.detection_mode is DetectionMode.THRESHOLD
                 else datemath_detect(s, spec.thresholds, kpi))
            v = float(s[-1])
            findings.append(KpiFinding(kpi=kpi, severity=r.severity.name, detail=r.detail,
                                       latest_value=v if math.isfinite(v) else None))
            continue

        # Established trailing baseline (long window, now validated finite + in-domain) becomes
        # the NEXT tick's baseline; kept longer than the detection window so CUSUM's ≥60-sample
        # calibration holds. Persisted only when the window meets the detector minimum.
        baseline_window = s[-baseline_days:]
        if baseline_window.size >= _MIN_BASELINE_N:
            new_baselines[kpi] = fit_baseline(baseline_window)

        current = s[-n:]
        prior = remembered.get(kpi)
        # The remembered baseline DRIVES detection and is loaded from a writable file, so a
        # corrupt/impossible baseline must not silently produce SAFE. Reject it if its mean is
        # out of domain, its std is infeasible for a bounded KPI (e.g. std=100 for pass_rate,
        # which flattens every z-score toward 0), or its sample count doesn't match the window
        # it claims to summarise (a legitimate save always has n == baseline_days).
        if prior is not None and (
            not baseline_domain_feasible(kpi, prior.mean, prior.std, bounds) or prior.n != baseline_days
        ):
            findings.append(data_quality_finding(kpi, "remembered baseline is corrupt/implausible (memory)"))
            continue
        if prior is not None:
            results = [
                zscore_detect(float(current[-1]), prior, kpi=kpi),
                cusum_detect(current, prior, kpi=kpi),
            ]
            sev = overall_severity(results)
            detail = next((r for r in results if r.severity == sev), results[0]).detail
        else:
            used_bootstrap = True
            sev = Severity.SAFE   # earned: current period validated finite + sufficient
            detail = "bootstrap baseline (no prior period remembered)"

        v = float(current[-1])
        latest = v if math.isfinite(v) else None
        findings.append(KpiFinding(kpi=kpi, severity=sev.name, detail=detail, latest_value=latest))

    # DATA_QUALITY dominates; else max of SAFE/WARNING/CRITICAL.
    if any(f.severity == "DATA_QUALITY" for f in findings):
        overall = Severity.DATA_QUALITY
    elif findings:
        overall = max(Severity[f.severity] for f in findings)
    else:
        overall = Severity.SAFE

    # Pass the streams AND the detection findings so forecast_node can enforce detect/forecast
    # parity (no forecast for a hero detection ruled DATA_QUALITY).
    forecast = forecast_node({"streams": streams, "kpi_findings": findings}, domain)["forecast"]
    llm = make_llm(offline, domain=domain)
    state = {"kpi_findings": findings, "overall_severity": overall.name, "forecast": forecast}
    analysis = analyst_node(state, llm, domain)["analysis"]
    rec_out = recommendation_node(state, llm, domain)

    generated_at = now or datetime.now().isoformat(timespec="seconds")
    run_id = f"tick-{tick_number}-{generated_at}"
    findings_obj = Findings(
        run_id=run_id, generated_at=generated_at, overall_severity=overall.name,
        alert=rec_out["alert"], kpi_findings=findings, forecast=forecast, analysis=analysis,
        recommendations=rec_out["recommendations"],
    )

    model = "offline-stub" if offline else "claude-sonnet-5"
    # The remembered baseline is a direct decision input (it drives detection) and is
    # overwritten after each tick — hash it into the lineage so a remembered-baseline tick's
    # decision is genuinely regenerable from what the audit records (Article 12).
    baseline_input_hash = hashlib.sha256(
        json.dumps(
            {kpi: [b.mean, b.std, b.n] for kpi, b in sorted(remembered.items())},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    # Article 12 lineage input hash of the actual input frames. Computed ONCE here (streams is
    # validated above) — never inside _write, so the write path can't fail on it after creating files.
    input_data_hash = hashlib.sha256(
        "".join(df.to_csv(index=False) for df in streams.values()).encode("utf-8")
    ).hexdigest()

    def _write(out_dir: Path, *, is_latest: bool) -> None:
        # Timestamped run dirs are durable, append-only history: never overwrite one (a
        # same-second, same-tick re-run would otherwise silently destroy a prior run).
        # `latest/` is a reusable mirror and is expected to be overwritten each tick.
        if not is_latest and out_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite existing run directory {out_dir} "
                "(durable run history); use a distinct timestamp or tick_number"
            )
        # Compute ALL content + integrity hashes BEFORE creating the directory or writing any file,
        # so a failure can never leave a partial, unverifiable run dir (findings.json/report.md
        # without a matching audit.jsonl).
        fj = json.dumps(findings_obj.to_json_dict(), indent=2, allow_nan=False)
        report_text = render_report(findings_obj)
        findings_hash = hashlib.sha256(fj.encode("utf-8")).hexdigest()
        report_hash = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "findings.json").write_text(fj)
        (out_dir / "report.md").write_text(report_text)
        # Each run dir holds exactly THIS run: clear any prior audit log (the reusable
        # `latest/` dir would otherwise accumulate across ticks and diverge from
        # findings.json/report.md). Durable per-run records live in the timestamped dirs.
        (out_dir / "audit.jsonl").unlink(missing_ok=True)
        audit = HashChainedAuditLog(out_dir / "audit.jsonl")
        audit.append(
            "run",
            {"overall_severity": overall.name, "alert": findings_obj.alert, "generated_at": generated_at,
             "model": model, "offline": offline, "findings_hash": findings_hash,
             "report_hash": report_hash, "input_data_hash": input_data_hash,
             "baseline_input_hash": baseline_input_hash,
             "seed": seed, "base_days": base_days, "days_per_period": days_per_period,
             "baseline_days": baseline_days, "tick_number": tick_number, "used_bootstrap": used_bootstrap},
            trace_id=run_id, timestamp=generated_at,
        )
        for i, rec in enumerate(findings_obj.recommendations):
            audit.append(
                "recommendation", {**asdict(rec), "model": model, "findings_hash": findings_hash},
                trace_id=f"{run_id}:{rec.kpi}:{i}", timestamp=generated_at,
            )

    # Run-dir name includes the tick number so two ticks that share a wall-clock second
    # (default timestamps have only second precision) don't collide onto the same directory.
    run_dir_name = run_id.replace(":", "-")
    _write(out_root / run_dir_name, is_latest=False)
    _write(out_root / "latest", is_latest=True)
    memory.save(new_baselines)  # cross-run state: this period is the next baseline
    return findings_obj
