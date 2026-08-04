from __future__ import annotations

from datetime import date
from pathlib import Path

import typer

from orgpulse.config import make_rng
from orgpulse.generators.clinical import generate_clinical
from orgpulse.generators.qc import generate_qc
from orgpulse.generators.supply import generate_supply

app = typer.Typer(help="OrgPulse AI — PharmaOps CLI")


@app.callback()
def main_callback():
    """OrgPulse AI — PharmaOps CLI"""
    pass


def write_streams(out: Path, days: int, seed: int) -> dict[str, Path]:
    if days < 1:
        raise ValueError("days must be >= 1")
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
    days: int = typer.Option(365, min=1, help="Number of days to generate"),
    seed: int = typer.Option(42, help="Random seed"),
    out: str = typer.Option("data", help="Output directory"),
) -> None:
    """Generate synthetic data for all three pharma-ops streams."""
    paths = write_streams(out=Path(out), days=days, seed=seed)
    for name, p in paths.items():
        typer.echo(f"wrote {name}: {p}")


def write_run(out: Path, days: int, seed: int, offline: bool):
    """Generate the three streams, run the analysis DAG, and write findings.json + report.md.

    Returns the Findings object. Offline (keyless) by default.
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    import hashlib
    import json
    from dataclasses import asdict

    # Lazy import so `orgpulse generate` doesn't pull in Prophet/LangGraph.
    from orgpulse.governance.audit import HashChainedAuditLog
    from orgpulse.pipeline.graph import render_report, run_pipeline

    out.mkdir(parents=True, exist_ok=True)
    start = date(2026, 1, 1)
    clinical = generate_clinical(start, days, make_rng(seed))
    qc = generate_qc(start, days, make_rng(seed + 1))
    supply = generate_supply(start, days, make_rng(seed + 2))
    findings = run_pipeline(clinical, qc, supply, offline=offline)

    findings_json = json.dumps(findings.to_json_dict(), indent=2, allow_nan=False)
    (out / "findings.json").write_text(findings_json)
    (out / "report.md").write_text(render_report(findings))

    # EU AI Act Article 12: hash-chained, reconstructable audit log of the run and
    # every AI-generated recommendation (trace_id, timestamp, model, input hash).
    model = "offline-stub" if offline else "claude-sonnet-5"
    input_hash = hashlib.sha256(findings_json.encode("utf-8")).hexdigest()
    audit = HashChainedAuditLog(out / "audit.jsonl")
    audit.append(
        "run",
        {
            "overall_severity": findings.overall_severity, "alert": findings.alert,
            "generated_at": findings.generated_at, "model": model, "offline": offline,
            "input_hash": input_hash,
        },
        trace_id=findings.run_id, timestamp=findings.generated_at,
    )
    for i, rec in enumerate(findings.recommendations):
        audit.append(
            "recommendation",
            {**asdict(rec), "model": model, "input_hash": input_hash},
            trace_id=f"{findings.run_id}:{rec.kpi}:{i}", timestamp=findings.generated_at,
        )
    return findings


@app.command()
def run(
    days: int = typer.Option(260, min=1, help="Days of data to generate and analyze"),
    seed: int = typer.Option(42, help="Random seed"),
    offline: bool = typer.Option(
        True, "--offline/--online",
        help="Offline deterministic LLM stub (keyless, default) vs Claude via API key",
    ),
    out: str = typer.Option("runs/latest", help="Output directory for findings.json + report.md"),
) -> None:
    """Run the analysis DAG end-to-end and write findings.json + report.md."""
    findings = write_run(out=Path(out), days=days, seed=seed, offline=offline)
    typer.echo(f"overall severity: {findings.overall_severity}   alert: {findings.alert}")
    typer.echo(f"wrote {out}/findings.json and {out}/report.md")


if __name__ == "__main__":
    app()
