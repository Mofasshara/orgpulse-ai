from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

from orgpulse.agent.baseline_memory import BaselineMemory
from orgpulse.config import make_rng
from orgpulse.detect.baseline import fit_baseline
from orgpulse.detect.cusum import cusum_detect
from orgpulse.detect.orchestrator import overall_severity
from orgpulse.detect.types import Severity
from orgpulse.detect.zscore import zscore_detect
from orgpulse.forecast.prophet_forecaster import series_from_generator
from orgpulse.generators.clinical import generate_clinical
from orgpulse.generators.qc import generate_qc
from orgpulse.generators.supply import generate_supply
from orgpulse.governance.audit import HashChainedAuditLog
from orgpulse.pipeline.graph import render_report
from orgpulse.pipeline.llm import make_llm
from orgpulse.pipeline.nodes import analyst_node, forecast_node, recommendation_node
from orgpulse.pipeline.state import Findings, KpiFinding


def _kpi_series(clinical, qc, supply) -> list[tuple[str, np.ndarray]]:
    return [
        ("daily_enrollment", series_from_generator(clinical, "date", "daily_enrollment")["y"].to_numpy()),
        ("pass_rate", qc["pass_rate"].to_numpy()),
        ("calibration_offset", qc["calibration_offset"].to_numpy()),
        ("lead_time_days", supply["lead_time_days"].to_numpy()),
    ]


def run_tick(
    memory: BaselineMemory,
    tick_number: int,
    *,
    days_per_period: int = 30,
    base_days: int = 200,
    seed: int = 42,
    offline: bool = True,
    out_root: Path = Path("runs"),
    now: str | None = None,
) -> Findings:
    """One autonomous tick: detect the current period against the REMEMBERED baseline
    (bootstrap on the first tick), forecast + reason, write run artifacts, and save the
    current period's baseline back to memory for the next tick.
    """
    days = base_days + tick_number * days_per_period
    start = date(2026, 1, 1)
    clinical = generate_clinical(start, days, make_rng(seed))
    qc = generate_qc(start, days, make_rng(seed + 1))
    supply = generate_supply(start, days, make_rng(seed + 2))

    remembered = memory.load()
    n = days_per_period
    findings: list[KpiFinding] = []
    new_baselines = {}
    used_bootstrap = False

    for kpi, series in _kpi_series(clinical, qc, supply):
        s = np.asarray(series, dtype=float)
        if len(s) < n:
            continue
        current = s[-n:]
        finite = current[np.isfinite(current)]
        if finite.size >= 1:
            new_baselines[kpi] = fit_baseline(finite)  # this period becomes the next baseline

        prior = remembered.get(kpi)
        if prior is not None:
            results = [
                zscore_detect(float(current[-1]), prior, kpi=kpi),
                cusum_detect(current, prior, kpi=kpi),
            ]
            sev = overall_severity(results)
            detail = next((r for r in results if r.severity == sev), results[0]).detail
        else:
            used_bootstrap = True
            sev = Severity.SAFE
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

    forecast = forecast_node({"clinical": clinical})["forecast"]
    llm = make_llm(offline)
    state = {"kpi_findings": findings, "overall_severity": overall.name, "forecast": forecast}
    analysis = analyst_node(state, llm)["analysis"]
    rec_out = recommendation_node(state, llm)

    generated_at = now or datetime.now().isoformat(timespec="seconds")
    run_id = f"tick-{tick_number}-{generated_at}"
    findings_obj = Findings(
        run_id=run_id, generated_at=generated_at, overall_severity=overall.name,
        alert=rec_out["alert"], kpi_findings=findings, forecast=forecast, analysis=analysis,
        recommendations=rec_out["recommendations"],
    )

    model = "offline-stub" if offline else "claude-sonnet-5"

    def _write(out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fj = json.dumps(findings_obj.to_json_dict(), indent=2, allow_nan=False)
        (out_dir / "findings.json").write_text(fj)
        (out_dir / "report.md").write_text(render_report(findings_obj))
        input_hash = hashlib.sha256(fj.encode("utf-8")).hexdigest()
        audit = HashChainedAuditLog(out_dir / "audit.jsonl")
        audit.append(
            "run",
            {"overall_severity": overall.name, "alert": findings_obj.alert, "generated_at": generated_at,
             "model": model, "offline": offline, "input_hash": input_hash,
             "tick_number": tick_number, "used_bootstrap": used_bootstrap},
            trace_id=run_id, timestamp=generated_at,
        )
        for i, rec in enumerate(findings_obj.recommendations):
            audit.append(
                "recommendation", {**asdict(rec), "model": model, "input_hash": input_hash},
                trace_id=f"{run_id}:{rec.kpi}:{i}", timestamp=generated_at,
            )

    _write(out_root / generated_at.replace(":", "-"))
    _write(out_root / "latest")
    memory.save(new_baselines)  # cross-run state: this period is the next baseline
    return findings_obj
