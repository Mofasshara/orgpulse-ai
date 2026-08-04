"""Seed a deterministic committed run history for the dashboard demo.

Runs the autonomous agent for several monthly ticks with fixed timestamps so the
deployed Streamlit dashboard (`runs/latest` + run history) is populated and
reproducible on first load. The compute plane produces these artifacts locally;
only the artifacts are committed (split-plane).

Usage:  uv run python scripts/seed_runs.py [n_ticks]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from orgpulse.agent.baseline_memory import BaselineMemory
from orgpulse.agent.tick import run_tick
from orgpulse.governance.audit import HashChainedAuditLog

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
MEMORY = REPO / "baseline_memory.json"

# One tick per month, fixed dates → reproducible timestamps in the committed tree.
_DATES = [
    "2026-03-02T07:00:00", "2026-04-01T07:00:00", "2026-05-01T07:00:00",
    "2026-06-01T07:00:00", "2026-07-01T07:00:00", "2026-08-03T07:00:00",
]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(_DATES)
    n = max(1, min(n, len(_DATES)))

    for path in (RUNS, MEMORY):
        if path.exists():
            shutil.rmtree(path) if path.is_dir() else path.unlink()

    mem = BaselineMemory(MEMORY)
    for tick, now in enumerate(_DATES[:n]):
        findings = run_tick(mem, tick, out_root=RUNS, now=now, offline=True)
        assert HashChainedAuditLog(RUNS / "latest" / "audit.jsonl").verify(), f"tick {tick} audit broken"
        print(f"tick {tick} @ {now[:10]}: {findings.overall_severity:<12} alert={findings.alert}")

    runs = sorted(p.name for p in RUNS.iterdir() if p.is_dir() and p.name != "latest")
    print(f"\nSeeded {len(runs)} run(s) + latest under {RUNS.relative_to(REPO)}/; memory at {MEMORY.name}")


if __name__ == "__main__":
    main()
