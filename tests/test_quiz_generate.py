"""POST /api/quiz/generate: the desktop UI's quiz source (exam_engine.api.generate_quiz_for_subject)."""
from __future__ import annotations

from .conftest import make_pdf, upload

from .exam_fixtures import LECTURES

# The exam-engine fixture lecture pages (neural networks, gradient descent, ...), one PDF page each.
NOTES = [page["text"] for page in LECTURES]


def _subject_with_notes(client, tmp_path, llm, settings) -> int:
    # The test's hashing embedder scores real paraphrases low; let every page reach the quiz generator.
    settings.min_score = 0.0
    sid = client.post("/api/subjects", json={"name": "ML"}).json()["id"]
    upload(client, make_pdf(tmp_path / "Lecture 1.pdf", NOTES), subject_id=sid, doc_type="lecture")
    llm.down = True  # Ollama unavailable: the engine must fall back to questions taken from the notes
    return sid


def test_generates_questions_with_sources_from_lecture_notes(client, tmp_path, llm, settings):
    sid = _subject_with_notes(client, tmp_path, llm, settings)
    r = client.post("/api/quiz/generate", json={"subject_id": sid, "topic": "Neural networks", "n_questions": 2})
    assert r.status_code == 200, r.text
    quiz = r.json()
    assert quiz["questions"], quiz["generation_report"]
    q = quiz["questions"][0]
    assert {o["key"] for o in q["options"]} >= {q["correct_option"]}
    assert q["source"]["document_name"] == "Lecture 1.pdf" and 1 <= q["source"]["page_number"] <= len(NOTES)
    assert quiz["generation_report"]["generated"] == len(quiz["questions"]) <= 2


def test_result_of_a_generated_quiz_can_be_submitted(client, tmp_path, llm, settings):
    sid = _subject_with_notes(client, tmp_path, llm, settings)
    quiz = client.post("/api/quiz/generate", json={"subject_id": sid, "topic": "Neural networks", "n_questions": 1}).json()
    answers = [{"topic": quiz["topic"], "correct": True, "question_ref": q["question_id"]} for q in quiz["questions"]]
    r = client.post("/api/quiz/submit", json={"subject_id": sid, "answers": answers})
    assert r.status_code == 201 and r.json()["correct"] == len(answers)


def test_unknown_subject_and_bad_input(client):
    r = client.post("/api/quiz/generate", json={"subject_id": 999, "topic": "perceptron"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "subject_not_found"
    assert client.post("/api/quiz/generate", json={"subject_id": 1, "topic": " "}).status_code == 422
    assert client.post("/api/quiz/generate", json={"subject_id": 1, "topic": "x" * 3, "n_questions": 50}).status_code == 422


def test_topic_with_no_notes_returns_an_explanation_not_an_error(client, tmp_path):
    sid = client.post("/api/subjects", json={"name": "Empty"}).json()["id"]
    r = client.post("/api/quiz/generate", json={"subject_id": sid, "topic": "photosynthesis"})
    assert r.status_code == 200
    assert r.json()["questions"] == [] and r.json()["generation_report"]["notes"]
