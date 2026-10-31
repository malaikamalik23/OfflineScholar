"""FAISS vector store: one flat inner-product index per subject, persisted to disk.

Vectors are L2-normalised, so inner product == cosine similarity. The FAISS vector id is the
`chunks.id` primary key, which is how a hit is mapped back to document name and page number.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import faiss
import numpy as np

REPLACE_ATTEMPTS = 5


class VectorStore:
    def __init__(self, index_dir: Path, dim_provider):
        self._dir = Path(index_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._dim_provider = dim_provider  # callable -> int (lazy, avoids loading model early)
        self._indexes: dict[int, faiss.Index] = {}
        self._lock = threading.RLock()

    def _path(self, subject_id: int) -> Path:
        return self._dir / f"subject_{subject_id}.faiss"

    def _get(self, subject_id: int, create: bool = True) -> faiss.Index | None:
        if subject_id in self._indexes:
            return self._indexes[subject_id]
        path = self._path(subject_id)
        if path.exists():
            index = faiss.read_index(str(path))
        elif create:
            index = faiss.IndexIDMap(faiss.IndexFlatIP(self._dim_provider()))
        else:
            return None
        self._indexes[subject_id] = index
        return index

    def _save(self, subject_id: int) -> None:
        tmp = self._path(subject_id).with_suffix(".tmp")
        faiss.write_index(self._indexes[subject_id], str(tmp))
        for attempt in range(REPLACE_ATTEMPTS):
            try:
                tmp.replace(self._path(subject_id))
                return
            except PermissionError:  # Windows: antivirus/indexer/sync briefly holding the file
                if attempt == REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(0.1 * (attempt + 1))

    def add(self, subject_id: int, ids: list[int], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        if not ids:
            return
        with self._lock:
            index = self._get(subject_id)
            index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), np.asarray(ids, dtype=np.int64))
            self._save(subject_id)

    def remove(self, subject_id: int, ids: list[int]) -> None:
        if not ids:
            return
        with self._lock:
            index = self._get(subject_id, create=False)
            if index is None:
                return
            index.remove_ids(np.asarray(ids, dtype=np.int64))
            self._save(subject_id)

    def search(self, vector: np.ndarray, subject_ids: list[int], k: int) -> list[tuple[int, float]]:
        """Top-k (chunk_id, cosine score) across the given subjects, best first."""
        q = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        hits: list[tuple[int, float]] = []
        with self._lock:
            for sid in subject_ids:
                index = self._get(sid, create=False)
                if index is None or index.ntotal == 0:
                    continue
                scores, ids = index.search(q, min(k, index.ntotal))
                hits += [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
        hits.sort(key=lambda h: h[1], reverse=True)
        return hits[:k]

    def count(self, subject_id: int) -> int:
        with self._lock:
            index = self._get(subject_id, create=False)
            return 0 if index is None else int(index.ntotal)

    def reset(self, subject_id: int) -> None:
        with self._lock:
            self._indexes.pop(subject_id, None)
            self._path(subject_id).unlink(missing_ok=True)

    def drop(self, subject_id: int) -> None:
        self.reset(subject_id)
