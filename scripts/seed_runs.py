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
from orgpulse.domains.pharma import make_pharma_domain
from orgpulse.governance.audit import HashChainedAuditLog

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
MEMORY = REPO / "baseline_memory.json"

# One tick per month, fixed dates → reproducible timestamps in the committed tree. The last
# tick is the genealogy-trust-layer demo scenario: sites 0-2 tagged manual_scanned lineage
# (degraded), every earlier tick is all system_of_record (clean) via make_pharma_domain below.
_DATES = [
    "2026-03-02T07:00:00", "2026-04-01T07:00:00", "2026-05-01T07:00:00",
    "2026-06-01T07:00:00", "2026-07-01T07:00:00", "2026-08-03T07:00:00",
    "2026-09-01T07:00:00",
]
_DEGRADED_LINEAGE_TICK = len(_DATES) - 1


def _seed_dir_names(n: int) -> list[str]:
    # Must match run_tick's run-dir naming: f"tick-{tick}-{now}" with colons → dashes.
    return [f"tick-{tick}-{now}".replace(":", "-") for tick, now in enumerate(_DATES[:n])]


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv
    n = int(args[0]) if args else len(_DATES)
    n = max(1, min(n, len(_DATES)))

    # Delete ONLY what this script owns — never rmtree the shared runs/ tree, which may hold
    # real durable runs or unrelated operator artifacts. Refuse (unless --force) if runs/
    # contains run dirs we did not seed.
    seeded = set(_seed_dir_names(n))
    if RUNS.exists():
        foreign = [
            p.name for p in RUNS.iterdir()
            if p.is_dir() and p.name != "latest" and p.name not in seeded
        ]
        if foreign and not force:
            raise SystemExit(
                f"runs/ contains {len(foreign)} run(s) not created by this seeder "
                f"(e.g. {foreign[:3]}). Refusing to touch them. Re-run with --force to seed anyway "
                "(only the seeded dirs + latest are replaced; foreign runs are left intact)."
            )
        for name in list(seeded) + ["latest"]:
            target = RUNS / name
            if target.exists():
                shutil.rmtree(target)
    if MEMORY.exists():
        MEMORY.unlink()

    mem = BaselineMemory(MEMORY)
    for tick, now in enumerate(_DATES[:n]):
        paper = frozenset({0, 1, 2}) if tick == _DEGRADED_LINEAGE_TICK else frozenset()
        domain = make_pharma_domain(paper_site_ids=paper)
        findings = run_tick(mem, tick, out_root=RUNS, now=now, offline=True, domain=domain)
        assert HashChainedAuditLog(RUNS / "latest" / "audit.jsonl").verify(), f"tick {tick} audit broken"
        print(f"tick {tick} @ {now[:10]}: {findings.overall_severity:<12} alert={findings.alert}")

    runs = sorted(p.name for p in RUNS.iterdir() if p.is_dir() and p.name != "latest")
    print(f"\nSeeded {len(runs)} run(s) + latest under {RUNS.relative_to(REPO)}/; memory at {MEMORY.name}")


if __name__ == "__main__":
    main()
