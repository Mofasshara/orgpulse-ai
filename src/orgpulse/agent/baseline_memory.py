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
        # Fail closed with a clear diagnosis on a corrupt/unreadable memory file, rather than
        # letting a raw JSONDecodeError/KeyError surface mid-way through the autonomous tick.
        try:
            raw = json.loads(self.path.read_text())
            if not isinstance(raw, dict):
                raise ValueError("expected a JSON object of per-KPI baselines")
            return {
                kpi: Baseline(mean=float(v["mean"]), std=float(v["std"]), n=int(v["n"]))
                for kpi, v in raw.items()
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"baseline memory at {self.path} is corrupt or unreadable ({exc}); "
                "delete it to re-bootstrap, or restore a valid copy"
            ) from exc

    def save(self, baselines: dict[str, Baseline]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {kpi: {"mean": b.mean, "std": b.std, "n": b.n} for kpi, b in baselines.items()},
                indent=2,
            )
        )
