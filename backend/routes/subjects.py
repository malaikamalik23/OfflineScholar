from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ai_engine.runtime import get_services
from backend.database import get_db
from backend.errors import AppError
from backend.models import Chunk, Document, Subject
from backend.schemas import SubjectCreate, SubjectOut

router = APIRouter(prefix="/subjects", tags=["subjects"])


@router.get("", response_model=list[SubjectOut])
def list_subjects(db: Session = Depends(get_db)):
    return db.query(Subject).order_by(Subject.name).all()


@router.post("", response_model=SubjectOut, status_code=201)
def create_subject(body: SubjectCreate, db: Session = Depends(get_db)):
    if db.query(Subject).filter(Subject.name == body.name).first():
        raise AppError(409, "subject_exists", f"Subject '{body.name}' already exists.")
    subject = Subject(name=body.name)
    db.add(subject)
    db.commit()
    return subject


@router.delete("/{subject_id}", status_code=204)
def delete_subject(subject_id: int, db: Session = Depends(get_db)):
    subject = db.get(Subject, subject_id)
    if subject is None:
        raise AppError(404, "subject_not_found", f"Subject {subject_id} not found")
    paths = [d.stored_path for d in subject.documents]
    get_services().vector_store.drop(subject_id)
    db.delete(subject)
    db.commit()
    from pathlib import Path
    for p in paths:
        Path(p).unlink(missing_ok=True)
