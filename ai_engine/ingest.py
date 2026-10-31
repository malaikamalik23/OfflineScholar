"""Ingestion pipeline: PDF -> pages -> chunks -> embeddings -> SQLite + FAISS."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from backend.database import session_scope
from backend.models import Chunk, Document, Subject

from .chunking import chunk_pages
from .extraction import ExtractionError, extract_document
from .runtime import Services, get_services

log = logging.getLogger(__name__)


def ingest_document(document_id: int, services: Services | None = None) -> None:
    """Process one uploaded document. Never raises: failures are recorded on the document row."""
    svc = services or get_services()
    cfg = svc.settings
    subject_id: int | None = None
    added_ids: list[int] = []
    try:
        with session_scope() as db:
            doc = db.get(Document, document_id)
            if doc is None:
                return
            doc.status, doc.error = "processing", None
            subject_id, path = doc.subject_id, doc.stored_path

        extracted = extract_document(
            path, ocr_enabled=cfg.ocr_enabled, ocr_language=cfg.ocr_language, ocr_dpi=cfg.ocr_dpi,
            tesseract_cmd=cfg.tesseract_cmd,
        )
        pieces = chunk_pages(extracted.pages, cfg.chunk_size, cfg.chunk_overlap)
        if not pieces:
            raise ExtractionError("no_extractable_text", "No readable text was found in this PDF.")

        with session_scope() as db:
            rows = [Chunk(document_id=document_id, page_number=p.page_number,
                          chunk_index=p.chunk_index, text=p.text) for p in pieces]
            db.add_all(rows)
            db.flush()  # assigns chunk ids == FAISS vector ids
            ids = [r.id for r in rows]
            vectors = svc.embedder.embed([r.text for r in rows])
            svc.vector_store.add(subject_id, ids, vectors)
            added_ids = ids

            doc = db.get(Document, document_id)
            doc.status, doc.error = "ready", None
            doc.page_count, doc.chunk_count = extracted.page_count, len(rows)
            doc.ocr_pages, doc.warnings = extracted.ocr_pages, extracted.warnings
            doc.processed_at = datetime.now(timezone.utc)
    except ExtractionError as exc:
        _mark_failed(document_id, exc.message, svc, subject_id, added_ids)
    except Exception as exc:  # noqa: BLE001 - recorded, not propagated (runs in background)
        log.exception("Ingestion failed for document %s", document_id)
        _mark_failed(document_id, f"Processing failed: {exc}", svc, subject_id, added_ids)


def _mark_failed(document_id: int, message: str, svc: Services, subject_id: int | None, added_ids: list[int]) -> None:
    if subject_id is not None and added_ids:
        try:
            svc.vector_store.remove(subject_id, added_ids)  # DB rows roll back with the transaction
        except Exception:  # noqa: BLE001
            log.exception("Could not roll back vectors for document %s", document_id)
    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is not None:
            doc.status, doc.error, doc.chunk_count = "failed", message, 0


def ensure_index_consistent(services: Services | None = None) -> list[int]:
    """At startup: rebuild any subject index whose size differs from the chunks in SQLite.

    Returns the subject ids that were rebuilt.
    """
    svc = services or get_services()
    rebuilt: list[int] = []
    with session_scope() as db:
        for subject in db.query(Subject).all():
            chunks = (db.query(Chunk).join(Document).filter(
                Document.subject_id == subject.id, Document.status == "ready").order_by(Chunk.id).all())
            if svc.vector_store.count(subject.id) == len(chunks):
                continue
            svc.vector_store.reset(subject.id)
            for i in range(0, len(chunks), 64):
                batch = chunks[i:i + 64]
                svc.vector_store.add(subject.id, [c.id for c in batch], svc.embedder.embed([c.text for c in batch]))
            rebuilt.append(subject.id)
            log.warning("Rebuilt vector index for subject %s (%d chunks)", subject.id, len(chunks))
    return rebuilt
