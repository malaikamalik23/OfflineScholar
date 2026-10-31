"""Offline Scholar — examination intelligence engine (Member 2).

Reusable entry points (see docs/JSON_CONTRACTS.md for every input/output shape):

    extract_questions(text)
    group_similar_questions(questions)
    calculate_topic_frequency(questions)
    analyze_syllabus_coverage(syllabus_topics, lecture_chunks)
    generate_quiz(context, topic)
    analyze_weak_topics(quiz_results)
    generate_revision_plan(analysis_results)

Everything runs locally; the only ML dependency is scikit-learn, and a sentence-transformers
model is used automatically (or Member 1's shared embedder is injected) when available.
"""
from .performance_analysis import analyze_weak_topics
from .question_extraction import extract_questions, extract_questions_from_chunks
from .quiz_generator import generate_quiz, grade_quiz
from .revision_engine import generate_revision_plan
from .service import ExamEngine
from .similarity import find_similar_questions, group_similar_questions
from .syllabus_coverage import analyze_syllabus_coverage, parse_syllabus_text
from .topic_analysis import (calculate_topic_frequency, extract_keywords, extract_lecture_topics,
                             get_important_topics, map_questions_to_topics)

__all__ = [
    "extract_questions", "extract_questions_from_chunks", "group_similar_questions", "find_similar_questions",
    "calculate_topic_frequency", "get_important_topics", "extract_keywords", "extract_lecture_topics",
    "map_questions_to_topics", "analyze_syllabus_coverage", "parse_syllabus_text", "generate_quiz",
    "grade_quiz", "analyze_weak_topics", "generate_revision_plan", "ExamEngine",
]
__version__ = "0.1.0"
