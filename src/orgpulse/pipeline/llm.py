from __future__ import annotations

import json
import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def analyze(self, system: str, prompt: str) -> str: ...
    def recommend(self, system: str, prompt: str, kpis: list[str]) -> list[dict]: ...


class OfflineLLM:
    """Deterministic, keyless LLM stub — the public-demo default."""

    def analyze(self, system: str, prompt: str) -> str:
        head = prompt.strip().splitlines()[0] if prompt.strip() else "the monitored KPIs"
        return (
            "Automated interpretation (offline mode): "
            f"the pipeline observed — {head}. These movements are assessed against "
            "GxP quality expectations; where a KPI is Warning/Critical the trend and "
            "forecast suggest a deviation trigger may apply and should be reviewed by "
            "the quality team."
        )

    def recommend(self, system: str, prompt: str, kpis: list[str]) -> list[dict]:
        out = []
        for i, kpi in enumerate(kpis, start=1):
            out.append({
                "kpi": kpi,
                "capa_frame": f"CAPA-{i:03d}",
                "text": (
                    f"Open a CAPA review for '{kpi}': confirm the signal against source "
                    "data, assess whether an ICH E6(R3) deviation threshold is crossed, "
                    "and, if confirmed, document root cause and corrective action. "
                    "Escalate to the responsible quality owner for sign-off."
                ),
            })
        return out


def _anthropic_client():  # indirection so tests can monkeypatch
    import anthropic

    return anthropic.Anthropic()


class ClaudeLLM:
    def __init__(self, model: str = "claude-sonnet-5") -> None:
        self.model = model
        self._client = _anthropic_client()

    def analyze(self, system: str, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self.model, max_tokens=1024, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

    def recommend(self, system: str, prompt: str, kpis: list[str]) -> list[dict]:
        schema = {
            "type": "object",
            "properties": {
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kpi": {"type": "string"},
                            "capa_frame": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["kpi", "capa_frame", "text"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["recommendations"],
            "additionalProperties": False,
        }
        msg = self._client.messages.create(
            model=self.model, max_tokens=2048, system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return json.loads(text)["recommendations"]


def make_llm(offline: bool, model: str = "claude-sonnet-5") -> LLMClient:
    if offline or not os.environ.get("ANTHROPIC_API_KEY"):
        return OfflineLLM()
    return ClaudeLLM(model=model)
