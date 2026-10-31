"""exam_engine <-> backend integration: real PDFs uploaded through the real API."""
from __future__ import annotations

import json

import pytest

from backend.database import session_scope
from backend.models import RevisionRecommendation
from exam_engine import api as exam_api
from exam_engine import bridge

from .conftest import make_pdf, upload
from .exam_fixtures import LECTURES

PAPERS = {
    "ML Paper 2023.pdf": ["Section A (3 marks each)\n1. Define overfitting.\n2. What is a confusion matrix?\n"
                          "Section B\n3. Explain the working of a neural network. [10 marks]\n"
                          "4. Describe the K-means clustering procedure. [10 marks]"],
    "ML Paper 2024.pdf": ["1. What is overfitting and why does it occur?\n2. Define precision and recall.\n"
                          "3. Describe how artificial neural networks function. [10 marks]\n"
                          "4. Explain support vector machines. [10 marks]"],
    "ML Paper 2025.pdf": ["1. Define overfitting and name one technique to reduce it.\n2. What is a confusion matrix?\n"
                          "3. Explain the working of a neural network and its activation functions. [10 marks]\n"
                          "4. Describe the K-means algorithm. [10 marks]"],
}
SYLLABUS = ["Syllabus\nNeural networks\nK-means clustering\nOverfitting\nReinforcement learning"]


@pytest.fixture(autouse=True)
def _uploads_in_tmp(monkeypatch, settings):
    """backend.routes.documents reads the global get_settings(); point it at the per-test settings so these
    tests write PDFs to a temp dir instead of the real data/uploads."""
    monkeypatch.setattr("backend.routes.documents.get_settings", lambda: settings)


@pytest.fixture()
def subject(client, tmp_path):
    """A subject with 3 past papers, a syllabus and lecture notes, all through the upload API."""
    sid = None
    for name, pages in PAPERS.items():
        d = upload(client, make_pdf(tmp_path / name, pages), subject_name="Machine Learning", doc_type="past_paper")
        sid = d["subject_id"]
    upload(client, make_pdf(tmp_path / "Syllabus.pdf", SYLLABUS), subject_id=sid, doc_type="syllabus")
    pages = [c["text"] for c in LECTURES]
    upload(client, make_pdf(tmp_path / "Lecture notes.pdf", pages), subject_id=sid, doc_type="lecture")
    return sid


def test_analysis_endpoint_end_to_end(client, subject):
    r = client.get(f"/api/analysis/{subject}")
    assert r.status_code == 200, r.text
    body = r.json()
    json.dumps(body)
    assert body["subject"]["name"] == "Machine Learning" and body["matching"] == "lexical"  # hashing test embedder
    past = body["past_papers"]
    assert len(past["questions"]) == 12 and all(q["page_number"] == 1 and q["year"] for q in past["questions"])
    assert any("neural" in g["shared_key_terms"] for g in past["repeated_questions"])
    top = past["important_topics"][0]
    assert top["topic"] == "Neural networks" and top["paper_count"] == 3
    cov = {t["topic"]: t for t in body["syllabus_coverage"]["topics"]}
    assert cov["Neural networks"]["status"] == "Covered" and cov["Neural networks"]["document"] == "Lecture notes.pdf"
    assert cov["Neural networks"]["pages"]
    assert cov["Reinforcement learning"]["status"] == "Not covered"
    assert body["weak_topics"] is None and any("quiz" in n.lower() for n in body["notes"])


def test_empty_subject_returns_guidance_not_errors(client):
    sid = client.post("/api/subjects", json={"name": "Empty"}).json()["id"]
    body = client.get(f"/api/analysis/{sid}").json()
    assert body["past_papers"] is None and body["syllabus_coverage"] is None and body["weak_topics"] is None
    assert any("past_paper" in n for n in body["notes"]) and any("syllabus" in n for n in body["notes"])
    rev = client.get(f"/api/revision/{sid}")
    assert rev.status_code == 200 and rev.json()["priority_topics"] == [] and rev.json()["saved_recommendations"] == 0


def test_revision_uses_quiz_results_and_replaces_stored_rows(client, subject):
    client.post("/api/quiz/submit", json={"subject_id": subject, "answers": (
        [{"topic": "Neural networks", "correct": False, "question_ref": f"q{i}"} for i in range(4)]
        + [{"topic": "Neural networks", "correct": True, "question_ref": "q9"}]
        + [{"topic": "Overfitting", "correct": True}] * 3)})
    first = client.get(f"/api/revision/{subject}").json()
    names = [t["topic"] for t in first["priority_topics"]]
    assert names[0] == "Neural networks" and first["priority_topics"][0]["signals"]["weakness"] > 0.5
    assert "Reinforcement learning" in names                           # syllabus gap with no exam history still listed
    assert first["saved_recommendations"] == len(names) and first["schedule"]

    client.get(f"/api/revision/{subject}")                              # again: rows are replaced, not duplicated
    with session_scope() as db:
        rows = db.query(RevisionRecommendation).filter_by(subject_id=subject).order_by(RevisionRecommendation.priority.desc()).all()
        assert len(rows) == len(names) and rows[0].topic == "Neural networks" and rows[0].reason
    weak = client.get(f"/api/analysis/{subject}").json()["weak_topics"]
    assert weak["weak_topics"] == ["Neural networks"]


def test_falls_back_to_stored_chunks_when_original_file_is_missing(client, subject, settings):
    bridge._PAGE_CACHE.clear()
    stored = list(settings.uploads_dir.glob("*.pdf"))
    assert len(stored) == 5            # 3 papers + syllabus + lecture notes (uuid-named by the backend)
    for f in stored:
        f.unlink()
    body = client.get(f"/api/analysis/{subject}").json()
    assert body["past_papers"] is not None and len(body["past_papers"]["questions"]) >= 8   # estimated, may be a bit lower
    assert any("could not be re-read" in n for n in body["notes"])


def test_pages_from_chunks_removes_overlap_and_restores_question_lines():
    a = "Section A. 1. Define overfitting. 2. What is a confusion matrix? 3. Explain the working of a neural network."
    b = "2. What is a confusion matrix? 3. Explain the working of a neural network. 4. Describe the K-means algorithm. [10 marks]"
    chunks = [{"document_id": 1, "document_name": "p.pdf", "page_number": 1, "text": a},
              {"document_id": 1, "document_name": "p.pdf", "page_number": 1, "text": b}]
    (page,) = bridge.pages_from_chunks(chunks)
    lines = page["text"].split("\n")
    assert sum("confusion matrix" in l for l in lines) == 1                   # overlap not duplicated
    assert [l.split(".")[0] for l in lines if l[:1].isdigit()] == ["1", "2", "3", "4"]


def test_quiz_for_subject_uses_lecture_notes_and_falls_back_without_llm_json(client, subject, llm):
    with session_scope() as db:
        quiz = exam_api.generate_quiz_for_subject(subject, "Neural networks", db, n_questions=3)
    assert quiz["topic"] == "Neural networks" and quiz["generation_report"]["from_llm"] == 0      # FakeLLM returned prose, not JSON
    assert llm.calls                                                                               # the local model was tried first
    for q in quiz["questions"]:
        assert q["source"]["document_name"] == "Lecture notes.pdf" and q["source"]["page_number"]
        assert len({o["text"] for o in q["options"]}) == 4

    llm.reply = json.dumps([{"question": "What does the learning rate control?",
                             "options": ["Size of each weight update", "Number of layers", "Batch size", "Dropout rate"],
                             "correct_index": 0, "explanation": "The notes say it controls the size of each update.",
                             "evidence": "The learning rate controls the size of each update; too large a learning rate makes training diverge.",
                             "difficulty": "basic"}])
    with session_scope() as db:
        quiz = exam_api.generate_quiz_for_subject(subject, "Gradient descent", db, n_questions=2)
    llm_q = [q for q in quiz["questions"] if q["generation_method"] == "llm"]
    assert llm_q and llm_q[0]["source"]["page_number"] >= 1


def test_quiz_for_unknown_topic_says_nothing_was_found(client, subject):
    with session_scope() as db:
        quiz = exam_api.generate_quiz_for_subject(subject, "Quantum chromodynamics gluons", db)
    assert quiz["questions"] == [] and quiz["generation_report"]["notes"]
