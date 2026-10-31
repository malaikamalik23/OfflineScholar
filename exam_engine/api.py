"""Entry points called by Member 1's backend (see docs/INTEGRATION.md).

    GET /api/analysis/{subject_id}  ->  analyze_subject(subject_id, db)
    GET /api/revision/{subject_id}  ->  recommend_revision(subject_id, db)

plus ``generate_quiz_for_subject`` for the quiz route Member 1 can add.

Document roles come from ``Document.doc_type``: ``past_paper`` (questions), ``syllabus`` (the
topic list), ``lecture`` (notes compared against the syllabus, and quiz material). Quiz answers
come from the ``quiz_answers`` table. Nothing here needs the LLM except optional quiz writing.
Both hooks return JSON-serialisable dicts; a missing input never raises — the response says what to
upload instead (``notes``), so the frontend can show a helpful empty state.
"""
from __future__ import annotations

from typing import Any

from . import bridge

DEFAULT_DAYS = 7
DEFAULT_MINUTES_PER_DAY = 90

_UPLOAD_HINT = {
    "past_paper": "No past papers uploaded yet — upload PDFs with doc_type=past_paper to see repeated questions and important topics.",
    "syllabus": "No syllabus uploaded yet — upload it with doc_type=syllabus so lecture notes can be checked for gaps.",
    "lecture": "No lecture notes uploaded yet — upload them with doc_type=lecture to check syllabus coverage and to generate quizzes.",
}


def _subject(db, subject_id: int):
    from backend.models import Subject

    return db.get(Subject, subject_id)


def _gather_and_analyse(subject_id: int, db) -> dict[str, Any]:
    """Run every analysis that the uploaded material supports."""
    engine = bridge.build_engine(use_llm=False)
    subject = _subject(db, subject_id)
    notes: list[str] = []

    paper_pages, n1 = bridge.load_pages(db, subject_id, "past_paper")
    syl_pages, n2 = bridge.load_pages(db, subject_id, "syllabus")
    lecture_chunks = bridge.load_chunks(db, subject_id, "lecture")
    quiz_records = bridge.load_quiz_records(db, subject_id)
    notes += n1 + n2

    syllabus = bridge.syllabus_topics_from(syl_pages) if syl_pages else []
    if not syl_pages:
        notes.append(_UPLOAD_HINT["syllabus"])
    elif not syllabus:
        notes.append("A syllabus was uploaded but no topics could be read from it.")

    past = None
    if paper_pages:
        past = engine.analyze_past_papers(paper_pages, syllabus_topics=syllabus or None)
        if not past["questions"]:
            notes.append("Past papers were read, but no questions could be recognised in them "
                         "(unusual numbering or a scanned page without OCR?).")
    else:
        notes.append(_UPLOAD_HINT["past_paper"])

    coverage = None
    if syllabus and lecture_chunks:
        coverage = engine.syllabus_coverage(syllabus, lecture_chunks)
    elif syllabus:
        notes.append(_UPLOAD_HINT["lecture"])

    weak = None
    if quiz_records:
        importance = {}
        if past:
            importance = {t["topic"]: t["importance_score"] / 100
                          for t in past["topic_frequency"]["topics"] if not t.get("one_off")}
        weak = engine.weak_topics(quiz_records, topic_importance=importance or None)
    else:
        notes.append("No quiz results yet — take a quiz so weak topics can be found.")

    return {
        "engine": engine,
        "subject": {"id": subject_id, "name": subject.name if subject else None},
        "documents": {"past_paper_pages": len(paper_pages), "syllabus_pages": len(syl_pages),
                      "lecture_chunks": len(lecture_chunks), "quiz_answers": len(quiz_records)},
        "past_papers": past, "syllabus_coverage": coverage, "weak_topics": weak, "notes": notes,
        "matching": engine.embedder.kind,
    }


def analyze_subject(subject_id: int, db) -> dict:
    """GET /api/analysis/{subject_id}: past-paper analysis + syllabus coverage + weak topics."""
    r = _gather_and_analyse(subject_id, db)
    return {
        "subject": r["subject"],
        "matching": r["matching"],  # "dense" (semantic model) or "lexical" (word-overlap fallback)
        "documents": r["documents"],
        "past_papers": r["past_papers"],
        "syllabus_coverage": r["syllabus_coverage"],
        "weak_topics": r["weak_topics"],
        "notes": r["notes"],
    }


def recommend_revision(subject_id: int, db, days: int = DEFAULT_DAYS,
                       minutes_per_day: int = DEFAULT_MINUTES_PER_DAY) -> dict:
    """GET /api/revision/{subject_id}: revision plan; also saved to ``revision_recommendations``."""
    from backend.models import RevisionRecommendation

    r = _gather_and_analyse(subject_id, db)
    analysis = {
        "topic_frequency": (r["past_papers"] or {}).get("topic_frequency"),
        "syllabus_coverage": r["syllabus_coverage"],
        "weak_topics": r["weak_topics"],
    }
    plan = r["engine"].revision_plan(analysis, days=days, minutes_per_day=minutes_per_day)

    # replace the stored recommendations for this subject with the current plan
    db.query(RevisionRecommendation).filter(RevisionRecommendation.subject_id == subject_id).delete()
    saved = 0
    for item in plan["priority_topics"]:
        reason = "; ".join(item["reasons"][:3]) or "Part of your revision plan."
        db.add(RevisionRecommendation(subject_id=subject_id, topic=item["topic"][:200],
                                      priority=item["priority_score"], reason=reason))
        saved += 1
    db.commit()

    return {
        "subject": r["subject"],
        "matching": r["matching"],
        "summary": plan["summary"],
        "priority_topics": plan["priority_topics"],
        "schedule": plan["schedule"],
        "notes": r["notes"] + plan["notes"],
        "saved_recommendations": saved,
    }


def generate_quiz_for_subject(subject_id: int, topic: str, db, n_questions: int = 5, top_k: int = 8) -> dict:
    """Quiz on ``topic`` from this subject's *lecture* notes (retrieved with the shared embeddings).

    Uses the local LLM when Ollama is reachable, otherwise falls back to extractive questions.
    Each question carries its source document + page. Submit results via POST /api/quiz/submit with
    ``topic`` and ``question_ref = question_id``.
    """
    from ai_engine.retrieval import retrieve_context

    hits = [h for h in retrieve_context(topic, subject_id, top_k=top_k) if h.doc_type == "lecture"]
    context = [{"document_id": h.document_id, "document_name": h.document_name,
                "page_number": h.page_number, "text": h.text} for h in hits]
    engine = bridge.build_engine(use_llm=True)
    quiz = engine.make_quiz(context, topic, n_questions)
    if not context:
        quiz["generation_report"]["notes"].append(
            "Nothing relevant to this topic was found in the uploaded lecture notes.")
    return quiz
