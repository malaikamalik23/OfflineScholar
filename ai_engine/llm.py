"""Local LLM client for Ollama (http://localhost:11434). No cloud APIs."""
from __future__ import annotations

from typing import Protocol

import httpx


class LLMUnavailableError(Exception):
    """Ollama is not running, the model is missing, or generation failed."""


class LLMClient(Protocol):
    model: str

    def chat(self, system: str, user: str) -> str: ...
    def status(self) -> dict: ...


def _prefer_ipv4_loopback(url: str) -> str:
    """Ollama listens on 127.0.0.1 only. On Windows "localhost" resolves to ::1 first, and a refused
    connection there is retried for ~2 s before falling back, which added ~2 s to every request."""
    u = httpx.URL(url)
    return str(u.copy_with(host="127.0.0.1")).rstrip("/") if u.host == "localhost" else url.rstrip("/")


class OllamaClient:
    def __init__(self, url: str, model: str, timeout: float = 180.0,
                 temperature: float = 0.1, num_ctx: int = 4096, transport: httpx.BaseTransport | None = None):
        self.url = _prefer_ipv4_loopback(url)
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx
        # One reusable client: creating one costs ~0.7 s (SSL setup). trust_env=False keeps local LLM
        # traffic away from any system HTTP proxy.
        self._http = httpx.Client(trust_env=False, transport=transport)

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        try:
            r = self._http.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self.url}. Start it with `ollama serve`."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(
                f"Ollama did not answer within {self.timeout:.0f}s. A smaller model may help."
            ) from exc
        if r.status_code == 404:
            raise LLMUnavailableError(f"Model '{self.model}' is not installed. Run `ollama pull {self.model}`.")
        if r.status_code != 200:
            raise LLMUnavailableError(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")
        content = (r.json().get("message") or {}).get("content", "")
        return content.strip()

    def status(self) -> dict:
        try:
            r = self._http.get(f"{self.url}/api/tags", timeout=3.0)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
            return {"reachable": False, "model": self.model, "model_installed": False, "detail": str(exc)[:200]}
        installed = any(n == self.model or n.split(":")[0] == self.model for n in names) or \
            (":" not in self.model and any(n.startswith(self.model + ":") for n in names))
        return {"reachable": True, "model": self.model, "model_installed": installed}
