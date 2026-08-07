from __future__ import annotations

import json
import os
from typing import Protocol, runtime_checkable

from orgpulse.core.domain import Domain
from orgpulse.domains.pharma import PHARMA


@runtime_checkable
class LLMClient(Protocol):
    def analyze(self, system: str, prompt: str) -> str: ...
    def recommend(self, system: str, prompt: str, kpis: list[str]) -> list[dict]: ...


class OfflineLLM:
    """Deterministic, keyless LLM stub — the public-demo default. Its framing comes from the
    active domain (the stub can't be prompted), so offline output is domain-correct (pharma GxP,
    banking FINMA, …) without a key."""

    def __init__(self, domain: "Domain" = PHARMA) -> None:
        self._domain = domain

    def analyze(self, system: str, prompt: str) -> str:
        head = prompt.strip().splitlines()[0] if prompt.strip() else "the monitored KPIs"
        return (
            "Automated interpretation (offline mode): "
            f"the pipeline observed — {head}. {self._domain.offline_framing}"
        )

    def recommend(self, system: str, prompt: str, kpis: list[str]) -> list[dict]:
        return self._domain.offline_recommend(kpis)


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


def make_llm(offline: bool, model: str = "claude-sonnet-5", domain: "Domain" = PHARMA) -> LLMClient:
    if offline or not os.environ.get("ANTHROPIC_API_KEY"):
        return OfflineLLM(domain)
    return ClaudeLLM(model=model)
