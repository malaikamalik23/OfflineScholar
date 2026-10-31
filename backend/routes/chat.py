from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ai_engine.rag import answer_question
from backend.database import get_db
from backend.errors import AppError
from backend.models import Subject
from backend.schemas import AskRequest, AskResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, db: Session = Depends(get_db)):
    """Answer a question from the uploaded materials, with document/page citations.

    If the materials don't contain the answer, `grounded` is false and `sources` is empty.
    """
    if req.subject_id is not None and db.get(Subject, req.subject_id) is None:
        raise AppError(404, "subject_not_found", f"Subject {req.subject_id} not found")
    result = answer_question(req.question, req.subject_id, req.top_k)  # may raise LLMUnavailableError -> 503
    return AskResponse(
        answer=result.answer, grounded=result.grounded,
        sources=[asdict(s) for s in result.sources], confidence=result.confidence,
    )
