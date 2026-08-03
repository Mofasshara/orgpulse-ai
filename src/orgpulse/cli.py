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


if __name__ == "__main__":
    app()
