import os

from orgpulse.pipeline.llm import OfflineLLM, make_llm


def test_offline_analyze_is_deterministic_and_keyless():
    llm = OfflineLLM()
    a = llm.analyze("sys", "enroll dropped 40%")
    b = llm.analyze("sys", "enroll dropped 40%")
    assert a == b and len(a) > 0


def test_offline_recommend_one_per_kpi_with_capa():
    recs = OfflineLLM().recommend("sys", "prompt", kpis=["enroll", "pass_rate"])
    assert len(recs) == 2
    assert {r["kpi"] for r in recs} == {"enroll", "pass_rate"}
    assert all("CAPA" in r["capa_frame"] for r in recs)
    assert all(r["text"] for r in recs)


def test_make_llm_defaults_to_offline_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(make_llm(offline=False), OfflineLLM)   # no key -> offline
    assert isinstance(make_llm(offline=True), OfflineLLM)


def test_claude_llm_uses_sonnet_and_parses(monkeypatch):
    # Mock the Anthropic SDK so no network/key is needed.
    import orgpulse.pipeline.llm as m

    class _Block:
        type = "text"; text = "analysis text"

    class _Msg:
        content = [_Block()]

    calls = {}

    class _Messages:
        def create(self, **kw):
            calls.update(kw); return _Msg()

    class _FakeClient:
        def __init__(self, *a, **k): self.messages = _Messages()

    monkeypatch.setattr(m, "_anthropic_client", lambda: _FakeClient())
    llm = m.ClaudeLLM(model="claude-sonnet-5")
    out = llm.analyze("sys", "prompt")
    assert out == "analysis text"
    assert calls["model"] == "claude-sonnet-5"
