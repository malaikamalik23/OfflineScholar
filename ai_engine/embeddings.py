"""Embedding backends. Production: Sentence Transformers (local). Tests/CI: hashing fallback."""
from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from backend.config import Settings


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 array of shape (len(texts), dim)."""


class EmbeddingModelUnavailableError(RuntimeError):
    """The embedding model is not on disk and offline mode forbids downloading it."""


class SentenceTransformerEmbedder:
    """Lazy-loads the model on first use (loading takes a few seconds)."""

    def __init__(self, model: str, offline: bool = False):
        self.name = model
        self._offline = offline
        self._model = None
        self._lock = threading.Lock()
        self._dim: int | None = None

    @property
    def available(self) -> bool:
        """False when offline and the model isn't saved locally (scripts/download_models.py not run)."""
        return self._model is not None or not self._offline or Path(self.name).is_dir()

    def _load(self):
        with self._lock:
            if self._model is None:
                if not self.available:
                    # Fail fast with a fix, instead of Hugging Face's "check your internet connection"
                    # after a ~13 s retry loop.
                    raise EmbeddingModelUnavailableError(
                        f"Embedding model '{self.name}' is not saved in data/models and offline mode "
                        "(OS_OFFLINE=1) forbids downloading it. Run `python scripts/download_models.py` "
                        "once while online.")
                if self._offline:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.name, device="cpu")
                get_dim = getattr(self._model, "get_embedding_dimension", None) or \
                    self._model.get_sentence_embedding_dimension  # name before sentence-transformers 6
                self._dim = int(get_dim())
        return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def dim(self) -> int:  # type: ignore[override]
        self._load()
        return self._dim  # type: ignore[return-value]

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vecs = model.encode(
            texts, batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


_STOP = frozenset(
    "a an the of to in on at for and or is are was were be been it this that these those with as by from "
    "what which who how why when where do does did can could should would explain define describe tell me about".split()
)


class HashingEmbedder:
    """Bag-of-words hashing embedder. Deterministic, no downloads: NOT semantic. Tests/CI only."""

    loaded = True
    available = True

    def __init__(self, dim: int = 384):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                if tok in _STOP or len(tok) < 2:
                    continue
                tok = tok.rstrip("s") if len(tok) > 3 else tok
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.resolved_embedding_model, offline=settings.offline)
