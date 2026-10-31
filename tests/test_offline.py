"""Offline mode: a missing embedding model must fail fast with a clear fix, never try to download."""
from __future__ import annotations

import socket
import time

import pytest
from fastapi.testclient import TestClient

from ai_engine.embeddings import EmbeddingModelUnavailableError, SentenceTransformerEmbedder
from ai_engine.runtime import build_services
from backend.config import Settings
from backend.main import create_app

from .conftest import upload


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*_a, **_kw):
        raise AssertionError("tried to use the network in offline mode")
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture
def offline_settings(tmp_path) -> Settings:  # sentence-transformers backend, but nothing in data/models
    return Settings(data_dir=tmp_path / "data", offline=True, _env_file=None)


def test_missing_model_fails_fast_without_network(offline_settings, no_network):
    """Regression: with OS_OFFLINE=1 and no data/models copy, Hugging Face retried for ~13 s and then
    told the user to check their internet connection."""
    emb = SentenceTransformerEmbedder(offline_settings.resolved_embedding_model, offline=True)
    assert emb.available is False
    t = time.perf_counter()
    with pytest.raises(EmbeddingModelUnavailableError, match="download_models.py"):
        emb.embed(["hello"])
    assert time.perf_counter() - t < 1.0


def test_local_model_dir_is_preferred(offline_settings):
    local = offline_settings.models_dir / "all-MiniLM-L6-v2"
    local.mkdir(parents=True)
    assert offline_settings.resolved_embedding_model == str(local)
    assert SentenceTransformerEmbedder(str(local), offline=True).available


def test_api_reports_missing_model(offline_settings, llm, pdfs, no_network):
    with TestClient(create_app(offline_settings, build_services(offline_settings, llm=llm))) as c:
        health = c.get("/api/health").json()
        assert health["status"] == "degraded" and health["embeddings"]["available"] is False
        doc = upload(c, pdfs["biology"], subject_name="Biology")
        status = c.get(f"/api/documents/{doc['id']}/status").json()
        assert status["status"] == "failed" and "download_models.py" in status["error"]
        r = c.post("/api/chat/ask", json={"question": "Where does photosynthesis happen?",
                                          "subject_id": doc["subject_id"]})
        assert r.status_code == 503 and r.json()["error"]["code"] == "embedding_model_unavailable"
