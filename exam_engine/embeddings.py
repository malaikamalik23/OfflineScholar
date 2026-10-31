"""Embedding adapter.

Member 1 owns the shared embedding service. This module does NOT build another
vector store; it only adapts whatever embedder is handed to us to one tiny
interface:

    embedder.kind               -> "dense" | "lexical"
    embedder.prepare(texts)     -> optional; lets corpus-fitted embedders (TF-IDF) fit
    embedder.encode(texts)      -> L2-normalised float32 matrix (n, d)

Resolution order in :func:`get_embedder`:
  1. an embedder passed in by the caller (Member 1's shared service),
  2. a local sentence-transformers model that is already on disk (no download),
  3. a TF-IDF fallback (scikit-learn, always available, fully offline).

Similarity thresholds depend on the embedder family, so they live here too.
"""
from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np

from .text_utils import lexical_analyzer

log = logging.getLogger(__name__)

# Cosine thresholds per embedder family. Dense values are sensible starting points
# for MiniLM-class models; calibrate on real data (see README, "Calibration").
THRESHOLDS = {
    "dense": {
        "question_group": 0.72,      # average-linkage cut for "same question, different words"
        "question_near_duplicate": 0.92,
        "topic_map": 0.45,           # question -> topic assignment
        "semantic_ref": 0.60,        # cosine treated as "strong semantic match" (=1.0)
    },
    "lexical": {
        "question_group": 0.50,
        "question_near_duplicate": 0.85,
        "topic_map": 0.20,
        "semantic_ref": 0.35,
    },
}


def thresholds_for(embedder) -> dict:
    return THRESHOLDS["dense" if getattr(embedder, "kind", "lexical") == "dense" else "lexical"]


def _l2(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class TfidfEmbedder:
    """Offline lexical embedder. Stemmed unigrams + bigrams, sublinear TF, L2 norm.

    Used only when no neural embedder is available. It matches *shared key terms*
    (after stemming and removing exam command words), so it handles
    "neural network" vs "artificial neural networks" but not true synonyms.
    """

    kind = "lexical"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(analyzer=lexical_analyzer, sublinear_tf=True, norm="l2")
        self._fitted = False

    def prepare(self, texts: Sequence[str]) -> None:
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return
        self._vec.fit(texts)
        self._fitted = True

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not self._fitted:
            self.prepare(texts)
        if not self._fitted:
            return np.zeros((len(texts), 1), dtype=np.float32)
        return _l2(self._vec.transform(texts).toarray())


class SentenceTransformerEmbedder:
    """Local sentence-transformers model (never downloads unless allowed)."""

    kind = "dense"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 local_files_only: bool = True) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name, local_files_only=local_files_only)

    def prepare(self, texts: Sequence[str]) -> None:  # nothing to fit
        return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return _l2(self._model.encode(list(texts), show_progress_bar=False))


class CallableEmbedder:
    """Wrap Member 1's shared embedding service.

    Accepts an object exposing ``encode(list[str])`` or ``embed(list[str])``,
    or a plain callable ``fn(list[str]) -> array-like``.
    """

    def __init__(self, service, kind: str = "dense") -> None:
        if hasattr(service, "encode"):
            self._fn: Callable = service.encode
        elif hasattr(service, "embed"):
            self._fn = service.embed
        elif callable(service):
            self._fn = service
        else:
            raise TypeError("embedder must have .encode/.embed or be callable")
        self.kind = getattr(service, "kind", kind)
        self._service = service

    def prepare(self, texts: Sequence[str]) -> None:
        prep = getattr(self._service, "prepare", None)
        if callable(prep):
            prep(list(texts))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return _l2(np.asarray(self._fn(list(texts))))


def get_embedder(embedder=None, prefer_dense: bool = True):
    """Resolve to an object with kind/prepare/encode (see module docstring)."""
    if embedder is not None:
        if hasattr(embedder, "kind") and hasattr(embedder, "prepare") and hasattr(embedder, "encode"):
            return embedder
        return CallableEmbedder(embedder)
    if prefer_dense:
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:  # not installed / model not cached / offline
            log.info("Dense embedder unavailable (%s); using TF-IDF fallback.", type(exc).__name__)
    return TfidfEmbedder()


def embed_all(embedder, *groups: Sequence[str]) -> list[np.ndarray]:
    """Fit once on the union of all texts, then encode each group in one space."""
    union = [t for g in groups for t in g]
    embedder.prepare(union)
    return [embedder.encode(list(g)) if len(g) else np.zeros((0, 1), np.float32) for g in groups]


def cosine_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    b = a if b is None else b
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return np.clip(a @ b.T, -1.0, 1.0)
