"""Thin facade over the seven reusable functions — the one object Member 1's backend needs.

    engine = ExamEngine(embedder=shared_embedding_service, llm=local_llm_callable)
    engine.analyze_past_papers(page_records)      # questions + repeated groups + topic ranking
    engine.syllabus_coverage(topics, lecture_pages)
    engine.make_quiz(context_chunks, "Topic", 5)
    engine.grade(quiz, {"question_id": "B"})
    engine.weak_topics(all_answer_records)
    engine.revision_plan(...)

Every method returns plain JSON-serialisable dicts (see docs/JSON_CONTRACTS.md). The
engine stores nothing: persistence (SQLite) stays with Member 1's backend.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .embeddings import get_embedder
from .performance_analysis import analyze_weak_topics
from .question_extraction import extract_questions_from_chunks
from .quiz_generator import generate_quiz, grade_quiz
from .revision_engine import generate_revision_plan
from .similarity import group_similar_questions
from .syllabus_coverage import analyze_syllabus_coverage
from .topic_analysis import calculate_topic_frequency, extract_lecture_topics, get_important_topics


def jsonable(obj):
    """Recursively convert numpy / set / tuple values so ``json.dumps`` always works."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class ExamEngine:
    def __init__(self, embedder=None, llm: Callable[[str], str] | None = None) -> None:
        # resolve once so every module works in the same embedding space
        self.embedder = get_embedder(embedder)
        self.llm = llm

    # Module 1 + 2
    def analyze_past_papers(self, page_records: Sequence[dict], syllabus_topics: Sequence | None = None,
                            top_k: int = 10) -> dict:
        """Modules 1+2. When ``syllabus_topics`` is given, ``topic_frequency`` is computed against
        those topics (so names line up with syllabus coverage and the revision plan) and the
        topics discovered from the questions themselves are returned as ``discovered_topic_frequency``."""
        questions = extract_questions_from_chunks(page_records)
        groups = group_similar_questions(questions, embedder=self.embedder)
        discovered = calculate_topic_frequency(questions, embedder=self.embedder)
        freq = (calculate_topic_frequency(questions, topics=syllabus_topics, embedder=self.embedder)
                if syllabus_topics else discovered)
        out = {
            "questions": questions,
            "similar_question_groups": groups,
            "repeated_questions": [g for g in groups["groups"] if g["repeated"]],
            "topic_frequency": freq,
            "important_topics": get_important_topics(freq, top_k=top_k),
        }
        if syllabus_topics:
            out["discovered_topic_frequency"] = discovered
        return jsonable(out)

    def lecture_topics(self, lecture_chunks: Sequence[dict], top_k: int = 15) -> list[dict]:
        return jsonable(extract_lecture_topics(lecture_chunks, top_k=top_k))

    # Module 3
    def syllabus_coverage(self, syllabus_topics: Sequence, lecture_chunks: Sequence[dict]) -> dict:
        return jsonable(analyze_syllabus_coverage(syllabus_topics, lecture_chunks, embedder=self.embedder))

    # Module 4
    def make_quiz(self, context, topic: str, n_questions: int = 5, seed: int = 0) -> dict:
        return jsonable(generate_quiz(context, topic, n_questions=n_questions, llm=self.llm, seed=seed))

    def grade(self, quiz: dict, answers: dict) -> dict:
        return jsonable(grade_quiz(quiz, answers))

    # Module 5
    def weak_topics(self, quiz_results, **kwargs) -> dict:
        return jsonable(analyze_weak_topics(quiz_results, **kwargs))

    # Module 6
    def revision_plan(self, analysis_results: dict, days: int = 7, minutes_per_day: int = 90) -> dict:
        return jsonable(generate_revision_plan(analysis_results, days=days, minutes_per_day=minutes_per_day,
                                               embedder=self.embedder))
