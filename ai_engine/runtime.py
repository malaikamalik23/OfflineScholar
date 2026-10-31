"""Shared singletons (embedder, vector store, LLM). Swappable for tests via set_services()."""
from __future__ import annotations

import threading
from dataclasses import dataclass

from backend.config import Settings, get_settings

from .embeddings import Embedder, build_embedder
from .llm import LLMClient, OllamaClient
from .vector_store import VectorStore


@dataclass
class Services:
    settings: Settings
    embedder: Embedder
    vector_store: VectorStore
    llm: LLMClient


_services: Services | None = None
_lock = threading.Lock()


def build_services(settings: Settings, *, embedder: Embedder | None = None, llm: LLMClient | None = None) -> Services:
    embedder = embedder or build_embedder(settings)
    return Services(
        settings=settings,
        embedder=embedder,
        vector_store=VectorStore(settings.index_dir, lambda: embedder.dim),
        llm=llm or OllamaClient(
            settings.ollama_url, settings.ollama_model, settings.ollama_timeout,
            settings.llm_temperature, settings.llm_num_ctx,
        ),
    )


def set_services(services: Services | None) -> None:
    global _services
    _services = services


def get_services() -> Services:
    global _services
    with _lock:
        if _services is None:
            settings = get_settings()
            settings.ensure_dirs()
            _services = build_services(settings)
        return _services
