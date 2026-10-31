"""The answer prompt's refusal rules (behaviour on real models: docs/EVALUATION.md)."""
from ai_engine.rag import NOT_FOUND_TOKEN, SYSTEM_PROMPT


def test_partly_covered_topics_get_a_partial_answer_not_a_refusal():
    assert "do not fully answer" in SYSTEM_PROMPT and "notes do not cover the rest" in SYSTEM_PROMPT
    assert "mention the topic but do not contain what is asked, reply" not in SYSTEM_PROMPT


def test_unrelated_questions_are_still_refused_and_outside_knowledge_forbidden():
    assert f"If no passage is about the topic of the question, reply with exactly {NOT_FOUND_TOKEN}" in SYSTEM_PROMPT
    assert "Do not add outside knowledge" in SYSTEM_PROMPT
