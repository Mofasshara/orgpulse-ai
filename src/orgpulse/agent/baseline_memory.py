from __future__ import annotations

import json
from pathlib import Path

from orgpulse.detect.baseline import Baseline


class BaselineMemory:
    """JSON-backed store of per-KPI baselines, persisted across agent runs.

    This is the agent's cross-run memory: each tick detects the current period
    against the baseline saved by the previous tick, then writes the current
    period's baseline back for the next tick to use.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Baseline]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text())
        return {
            kpi: Baseline(mean=float(v["mean"]), std=float(v["std"]), n=int(v["n"]))
            for kpi, v in raw.items()
        }

    def save(self, baselines: dict[str, Baseline]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {kpi: {"mean": b.mean, "std": b.std, "n": b.n} for kpi, b in baselines.items()},
                indent=2,
            )
        )
