from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from ai_engine.ingest import ingest_document
from ai_engine.runtime import get_services
from backend.database import get_db
from backend.errors import AppError
from backend.models import Chunk, Document, Subject
from backend.schemas import DocType, DocumentOut, DocumentStatus
from backend.storage import remove_file

router = APIRouter(prefix="/documents", tags=["documents"])

DEFAULT_SUBJECT = "General"


def _get_document(db: Session, document_id: int) -> Document:
    doc = db.get(Document, document_id)
    if doc is None:
        raise AppError(404, "document_not_found", f"Document {document_id} not found")
    return doc


def _resolve_subject(db: Session, subject_id: int | None, subject_name: str | None) -> Subject:
    if subject_id is not None:
        subject = db.get(Subject, subject_id)
        if subject is None:
            raise AppError(404, "subject_not_found", f"Subject {subject_id} not found")
        return subject
    name = (subject_name or DEFAULT_SUBJECT).strip() or DEFAULT_SUBJECT
    subject = db.query(Subject).filter(Subject.name == name).one_or_none()
    if subject is None:
        subject = Subject(name=name)
        db.add(subject)
        db.flush()
    return subject


@router.post("/upload", response_model=DocumentOut, status_code=202)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    subject_id: int | None = Form(None),
    subject_name: str | None = Form(None, max_length=200),
    doc_type: DocType = Form(DocType.lecture),
    db: Session = Depends(get_db),
):
    """Upload a PDF. Processing (extract -> chunk -> embed -> index) runs in the background;
    poll GET /documents/{id}/status until status is `ready` or `failed`."""
    settings = get_services().settings  # the running app's settings, not the global default
    original = Path(file.filename or "").name
    if not original.lower().endswith(".pdf"):
        raise AppError(400, "unsupported_file_type", "Only .pdf files are accepted.")

    limit = settings.max_upload_mb * 1024 * 1024
    stored = settings.uploads_dir / f"{uuid.uuid4().hex}.pdf"
    settings.ensure_dirs()
    digest, size = hashlib.sha256(), 0
    try:
        with stored.open("wb") as out:
            while block := await file.read(1024 * 1024):
                size += len(block)
                if size > limit:
                    raise AppError(413, "file_too_large", f"File exceeds {settings.max_upload_mb} MB.")
                if size == len(block) and not block.startswith(b"%PDF"):
                    raise AppError(400, "invalid_pdf", "File content is not a PDF.")
                digest.update(block)
                out.write(block)
        if size == 0:
            raise AppError(400, "empty_file", "The uploaded file is empty.")

        subject = _resolve_subject(db, subject_id, subject_name)
        duplicate = db.query(Document).filter(
            Document.subject_id == subject.id, Document.sha256 == digest.hexdigest()).first()
        if duplicate:
            raise AppError(409, "duplicate_document",
                           f"This file was already uploaded as '{duplicate.filename}' (id {duplicate.id}).")
        doc = Document(subject_id=subject.id, filename=original, doc_type=doc_type.value,
                       stored_path=str(stored), sha256=digest.hexdigest(), size_bytes=size, status="pending")
        db.add(doc)
        db.commit()
    except BaseException:
        db.rollback()
        stored.unlink(missing_ok=True)
        raise

    background.add_task(ingest_document, doc.id)
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(subject_id: int | None = None, db: Session = Depends(get_db)):
    q = db.query(Document).order_by(Document.created_at.desc(), Document.id.desc())
    if subject_id is not None:
        q = q.filter(Document.subject_id == subject_id)
    return q.all()


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, db: Session = Depends(get_db)):
    return _get_document(db, document_id)


@router.get("/{document_id}/status", response_model=DocumentStatus)
def document_status(document_id: int, db: Session = Depends(get_db)):
    return _get_document(db, document_id)


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = _get_document(db, document_id)
    if doc.status == "processing":
        raise AppError(409, "document_processing", "Wait for processing to finish before deleting.")
    chunk_ids = [cid for (cid,) in db.query(Chunk.id).filter(Chunk.document_id == doc.id)]
    get_services().vector_store.remove(doc.subject_id, chunk_ids)
    path = doc.stored_path
    db.delete(doc)
    db.commit()
    remove_file(path)  # a locked file (Windows) must not turn a completed delete into a 500
