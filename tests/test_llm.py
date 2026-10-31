"""OllamaClient tests with a mock HTTP transport (no Ollama needed)."""
from __future__ import annotations

import httpx
import pytest

from ai_engine.llm import LLMUnavailableError, OllamaClient


def _client(handler, url="http://localhost:11434", model="qwen2.5:3b") -> tuple[OllamaClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return OllamaClient(url, model, transport=httpx.MockTransport(record)), seen


def test_localhost_is_rewritten_to_ipv4_loopback():
    """Regression: on Windows "localhost" tried ::1 first and added ~2 s to every Ollama call."""
    c, seen = _client(lambda r: httpx.Response(200, json={"message": {"content": "Hi [1]"}}))
    assert c.chat("sys", "user") == "Hi [1]"
    assert seen[0].url.host == "127.0.0.1" and seen[0].url.port == 11434
    assert OllamaClient("http://192.168.1.5:11434/", "m").url == "http://192.168.1.5:11434"


def test_client_is_reused_and_ignores_proxy_env(monkeypatch):
    """Regression: a new httpx client per call cost ~0.7 s on Windows; proxies must not see LLM traffic."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    c, seen = _client(lambda r: httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]}))
    http = c._http
    assert c.status()["model_installed"] and c.status()["reachable"]
    assert c._http is http and len(seen) == 2
    assert http._trust_env is False


def test_missing_model_and_errors_raise_llm_unavailable():
    c, _ = _client(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(LLMUnavailableError, match="ollama pull"):
        c.chat("s", "u")
    c, _ = _client(lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    with pytest.raises(LLMUnavailableError, match="Cannot reach Ollama"):
        c.chat("s", "u")
    assert c.status()["reachable"] is False
