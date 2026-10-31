from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend import integration
from backend.database import get_db
from backend.errors import AppError
from backend.models import Subject

router = APIRouter(tags=["exam-engine"])


def _require_subject(db: Session, subject_id: int) -> None:
    if db.get(Subject, subject_id) is None:
        raise AppError(404, "subject_not_found", f"Subject {subject_id} not found")


@router.get("/analysis/{subject_id}")
def analysis(subject_id: int, db: Session = Depends(get_db)):
    """Exam/past-paper analysis, produced by exam_engine (Member 2)."""
    _require_subject(db, subject_id)
    return integration.get_analysis(subject_id, db)


@router.get("/revision/{subject_id}")
def revision(subject_id: int, db: Session = Depends(get_db)):
    """Revision recommendations, produced by exam_engine (Member 2)."""
    _require_subject(db, subject_id)
    return integration.get_revision(subject_id, db)
