from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend import integration
from backend.database import get_db
from backend.errors import AppError
from backend.models import QuizAnswer, QuizAttempt, Subject, TopicPerformance
from backend.schemas import QuizGenerate, QuizResult, QuizSubmit, TopicStat

router = APIRouter(prefix="/quiz", tags=["quiz"])


@router.post("/generate")
def generate_quiz(body: QuizGenerate, db: Session = Depends(get_db)):
    """Multiple-choice questions on `topic`, built by exam_engine from this subject's lecture notes.

    Each question carries its source document and page. Grade on the client, then send the results to
    POST /quiz/submit with `topic` and `question_ref` = the question's id.
    """
    if db.get(Subject, body.subject_id) is None:
        raise AppError(404, "subject_not_found", f"Subject {body.subject_id} not found")
    return integration.generate_quiz(body.subject_id, body.topic, body.n_questions, db)


@router.post("/submit", response_model=QuizResult, status_code=201)
def submit_quiz(body: QuizSubmit, db: Session = Depends(get_db)):
    """Record a finished quiz and update per-topic performance.

    The client (or Member 2's quiz generator) grades each question; we store the outcome so
    revision recommendations can be computed from real performance.
    """
    if db.get(Subject, body.subject_id) is None:
        raise AppError(404, "subject_not_found", f"Subject {body.subject_id} not found")

    total = len(body.answers)
    correct = sum(a.correct for a in body.answers)
    attempt = QuizAttempt(subject_id=body.subject_id, total=total, correct=correct, score=correct / total)
    attempt.answers = [QuizAnswer(topic=a.topic.strip(), question_ref=a.question_ref, correct=a.correct)
                       for a in body.answers]
    db.add(attempt)

    touched: dict[str, TopicPerformance] = {}
    for a in body.answers:
        topic = a.topic.strip()
        perf = touched.get(topic) or db.query(TopicPerformance).filter_by(
            subject_id=body.subject_id, topic=topic).one_or_none()
        if perf is None:
            perf = TopicPerformance(subject_id=body.subject_id, topic=topic, attempts=0, correct=0)
            db.add(perf)
        perf.attempts += 1
        perf.correct += int(a.correct)
        touched[topic] = perf
    db.commit()

    return QuizResult(
        attempt_id=attempt.id, total=total, correct=correct, score=attempt.score,
        topics=[TopicStat(topic=t, attempts=p.attempts, correct=p.correct, accuracy=p.correct / p.attempts)
                for t, p in sorted(touched.items())],
    )
