from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from ai_engine.interfaces import (
    DocumentNotFoundError, generate_embeddings, get_document_chunks, retrieve_context,
)

from .conftest import BIOLOGY, upload


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["database"]["ok"] and body["llm"]["reachable"]


def test_upload_process_list_status(client, pdfs):
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    status = client.get(f"/api/documents/{doc['id']}/status").json()
    assert status["status"] == "ready", status
    assert status["page_count"] == 2 and status["chunk_count"] >= 2
    listed = client.get("/api/documents").json()
    assert [d["filename"] for d in listed] == ["Biology 01.pdf"]
    assert client.get(f"/api/documents?subject_id={doc['subject_id']}").json()[0]["id"] == doc["id"]


def test_upload_validation(client, tmp_path, pdfs):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    r = client.post("/api/documents/upload", files={"file": ("notes.txt", txt.open("rb"))})
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_file_type"

    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"not a pdf at all")
    r = client.post("/api/documents/upload", files={"file": ("fake.pdf", fake.open("rb"))})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_pdf"

    upload(client, pdfs["biology"], subject_name="Biology")
    with pdfs["biology"].open("rb") as f:
        r = client.post("/api/documents/upload", files={"file": ("Biology 01.pdf", f)}, data={"subject_name": "Biology"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "duplicate_document"


def test_unreadable_pdf_is_marked_failed_not_crashed(client, tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4\ngarbage that is not a valid pdf structure")
    doc = upload(client, bad)
    status = client.get(f"/api/documents/{doc['id']}/status").json()
    assert status["status"] == "failed" and status["error"]
    assert client.get("/api/documents").status_code == 200


def test_ask_returns_grounded_answer_with_page(client, pdfs, llm):
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    r = client.post("/api/chat/ask", json={"question": "Where does photosynthesis take place?",
                                            "subject_id": doc["subject_id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grounded"] is True and body["answer"]
    src = body["sources"][0]
    assert src["document_name"] == "Biology 01.pdf" and src["page_number"] == 1
    assert "chloroplasts" in src["excerpt"]
    # the prompt really contained retrieved context, with its page reference
    assert "Biology 01.pdf, page 1" in llm.calls[0][1]


def test_ask_cites_page_two_for_mitosis(client, pdfs):
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    body = client.post("/api/chat/ask", json={"question": "What are the stages of mitosis?",
                                               "subject_id": doc["subject_id"]}).json()
    assert body["grounded"] and body["sources"][0]["page_number"] == 2


def test_unanswerable_question_skips_llm(client, pdfs, llm):
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    body = client.post("/api/chat/ask", json={"question": "Explain quantum chromodynamics gluons",
                                               "subject_id": doc["subject_id"]}).json()
    assert body["grounded"] is False and body["sources"] == []
    assert "couldn't find" in body["answer"].lower()
    assert llm.calls == []  # no chance to hallucinate


def test_model_can_abstain(client, pdfs, llm):
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    llm.reply = "NOT_FOUND"
    body = client.post("/api/chat/ask", json={"question": "Where does photosynthesis take place?",
                                               "subject_id": doc["subject_id"]}).json()
    assert body["grounded"] is False and body["sources"] == []


def test_only_cited_sources_returned(client, pdfs, llm):
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    llm.reply = "Photosynthesis occurs in chloroplasts [1]."
    body = client.post("/api/chat/ask", json={"question": "photosynthesis chloroplasts", "subject_id": doc["subject_id"]}).json()
    assert len(body["sources"]) == 1


def test_subject_isolation(client, pdfs):
    bio = upload(client, pdfs["biology"], subject_name="Biology")
    his = upload(client, pdfs["history"], subject_name="History")
    q = {"question": "When was the Treaty of Versailles signed?"}
    assert client.post("/api/chat/ask", json={**q, "subject_id": bio["subject_id"]}).json()["grounded"] is False
    hit = client.post("/api/chat/ask", json={**q, "subject_id": his["subject_id"]}).json()
    assert hit["grounded"] and hit["sources"][0]["document_name"] == "History 01.pdf"
    assert client.post("/api/chat/ask", json=q).json()["grounded"]  # all subjects


def test_llm_down_gives_503(client, pdfs, llm):
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    llm.down = True
    r = client.post("/api/chat/ask", json={"question": "Where does photosynthesis take place?", "subject_id": doc["subject_id"]})
    assert r.status_code == 503 and r.json()["error"]["code"] == "llm_unavailable"
    assert client.get("/api/health").json()["status"] == "degraded"


def test_request_validation(client):
    assert client.post("/api/chat/ask", json={"question": "x"}).status_code == 422
    assert client.post("/api/chat/ask", json={"question": "valid question", "top_k": 99}).status_code == 422
    assert client.post("/api/chat/ask", json={"question": "valid question", "subject_id": 999}).status_code == 404


def test_delete_removes_document_and_vectors(client, pdfs, settings):
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert client.get("/api/documents").json() == []
    assert client.get(f"/api/documents/{doc['id']}").status_code == 404
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 404
    body = client.post("/api/chat/ask", json={"question": "Where does photosynthesis take place?"}).json()
    assert body["grounded"] is False
    assert list(settings.uploads_dir.glob("*.pdf")) == []


def test_quiz_submit_updates_topic_performance(client, pdfs):
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    sid = doc["subject_id"]
    payload = {"subject_id": sid, "answers": [
        {"topic": "Mitosis", "correct": True}, {"topic": "Mitosis", "correct": False},
        {"topic": "Photosynthesis", "correct": True}]}
    r = client.post("/api/quiz/submit", json=payload)
    assert r.status_code == 201
    assert r.json()["total"] == 3 and r.json()["correct"] == 2
    again = client.post("/api/quiz/submit", json={"subject_id": sid, "answers": [{"topic": "Mitosis", "correct": True}]}).json()
    assert again["topics"] == [{"topic": "Mitosis", "attempts": 3, "correct": 2, "accuracy": pytest.approx(2 / 3)}]
    assert client.post("/api/quiz/submit", json={"subject_id": sid, "answers": []}).status_code == 422
    assert client.post("/api/quiz/submit", json={"subject_id": 99, "answers": payload["answers"]}).status_code == 404


def test_exam_engine_endpoints(client, pdfs, monkeypatch):
    sid = upload(client, pdfs["biology"], subject_name="Biology")["subject_id"]
    monkeypatch.setitem(sys.modules, "exam_engine", None)  # simulate "not installed yet"
    r = client.get(f"/api/analysis/{sid}")
    assert r.status_code == 501 and r.json()["error"]["code"] == "exam_engine_unavailable"

    pkg = types.ModuleType("exam_engine")
    pkg.__path__ = []
    api = types.ModuleType("exam_engine.api")
    api.analyze_subject = lambda subject_id, db: {"subject_id": subject_id, "topics": ["Mitosis"]}
    api.recommend_revision = lambda subject_id, db: {"subject_id": subject_id, "recommendations": []}
    monkeypatch.setitem(sys.modules, "exam_engine", pkg)
    monkeypatch.setitem(sys.modules, "exam_engine.api", api)
    assert client.get(f"/api/analysis/{sid}").json()["topics"] == ["Mitosis"]
    assert client.get(f"/api/revision/{sid}").json()["recommendations"] == []
    assert client.get("/api/analysis/999").status_code == 404


def test_member2_interfaces(client, pdfs):
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    chunks = get_document_chunks(doc["id"])
    assert chunks and chunks[0].document_name == "Biology 01.pdf" and chunks[0].page_number == 1
    assert [c.chunk_index for c in chunks] == sorted(c.chunk_index for c in chunks)
    with pytest.raises(DocumentNotFoundError):
        get_document_chunks(9999)

    vecs = generate_embeddings(["mitosis", "photosynthesis"])
    assert vecs.shape == (2, 384) and vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)

    hits = retrieve_context("stages of mitosis", doc["subject_id"], top_k=2)
    assert hits and hits[0].page_number == 2 and hits[0].score >= hits[-1].score
    assert hits[0].to_dict()["document_name"] == "Biology 01.pdf"
    with pytest.raises(ValueError):
        retrieve_context("   ")


def test_index_rebuilt_if_missing(settings, llm, pdfs):
    """Deleting the FAISS file must not lose data: startup rebuilds it from SQLite."""
    from fastapi.testclient import TestClient
    from ai_engine.runtime import build_services
    from backend.main import create_app

    with TestClient(create_app(settings, build_services(settings, llm=llm))) as c:
        doc = upload(c, pdfs["biology"], subject_name="Biology")
    for f in settings.index_dir.glob("*.faiss"):
        f.unlink()
    with TestClient(create_app(settings, build_services(settings, llm=llm))) as c:
        body = c.post("/api/chat/ask", json={"question": "Where does photosynthesis take place?",
                                              "subject_id": doc["subject_id"]}).json()
        assert body["grounded"] is True


def test_answer_padded_with_outside_knowledge_is_rejected(client, pdfs, llm):
    """Regression (real qwen2.5:3b run): for a question the notes don't cover, the model answered from
    general knowledge and cited a loosely related passage. Such answers must come back ungrounded."""
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    llm.reply = ("Random forests reduce overfitting by averaging many decorrelated trees trained on "
                 "bootstrap samples, which lowers variance [1].")
    body = client.post("/api/chat/ask", json={"question": "How does photosynthesis relate to random forests?",
                                               "subject_id": doc["subject_id"]}).json()
    assert body["grounded"] is False and body["sources"] == []
    assert "couldn't find" in body["answer"].lower()


def test_paraphrased_answer_from_passages_is_kept(client, pdfs, llm):
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    llm.reply = "It happens in the chloroplasts, where chlorophyll absorbs sunlight [1]."
    body = client.post("/api/chat/ask", json={"question": "Where does photosynthesis take place?",
                                               "subject_id": doc["subject_id"]}).json()
    assert body["grounded"] is True and body["sources"][0]["page_number"] == 1


def test_db_not_locked_while_embedding(settings, llm, pdfs):
    """Regression: chunks were inserted (taking SQLite's write lock) and only then embedded. A big PDF
    embeds for 30 s+ on a laptop CPU, so concurrent uploads/deletes/quiz submits hit 'database is locked'."""
    import sqlite3

    from fastapi.testclient import TestClient

    from ai_engine.embeddings import HashingEmbedder
    from ai_engine.runtime import build_services
    from backend.main import create_app

    db_file = settings.data_dir / "offline_scholar.db"
    lock_errors: list[str] = []

    class ProbingEmbedder(HashingEmbedder):
        def embed(self, texts):
            if len(texts) > 1:  # document chunks, not a query
                con = sqlite3.connect(db_file, timeout=0.2)
                try:
                    con.execute("BEGIN IMMEDIATE")  # what any other writer needs
                    con.execute("ROLLBACK")
                except sqlite3.OperationalError as exc:
                    lock_errors.append(str(exc))
                finally:
                    con.close()
            return super().embed(texts)

    with TestClient(create_app(settings, build_services(settings, embedder=ProbingEmbedder(), llm=llm))) as c:
        doc = upload(c, pdfs["biology"], subject_name="Biology")
        assert c.get(f"/api/documents/{doc['id']}/status").json()["status"] == "ready"
    assert lock_errors == []


def test_abstention_with_explanation_is_not_grounded(client, pdfs, llm):
    """The model sometimes explains before abstaining; stripping the token must not leave an 'answer'."""
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    # Words all come from the passages, so the support guard alone would let this through.
    llm.reply = ("The passages only say that photosynthesis takes place in the chloroplasts, where "
                 "chlorophyll absorbs sunlight. NOT_FOUND")
    body = client.post("/api/chat/ask", json={"question": "Where does photosynthesis take place?",
                                               "subject_id": doc["subject_id"]}).json()
    assert llm.calls, "the model must have been asked"
    assert body["grounded"] is False and body["sources"] == []


def test_general_knowledge_hidden_behind_echoed_passages_is_rejected(client, pdfs, llm):
    """Regression (real phi3:mini run): the model answered from general knowledge and then pasted the
    prompt's passages, whose copied words made the answer look supported."""
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    llm.reply = ("Mitochondria generate ATP using a proton gradient. [1] (Biology 01.pdf, page 1) " + BIOLOGY[0])
    body = client.post("/api/chat/ask", json={"question": "How do the light reactions of photosynthesis make ATP?",
                                               "subject_id": doc["subject_id"]}).json()
    assert llm.calls and body["grounded"] is False


def test_refusal_in_prose_is_not_grounded(client, pdfs, llm):
    """Regression (real phi3:mini run): refused in words instead of NOT_FOUND, then echoed the passages."""
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    llm.reply = ("I'm sorry, but the provided context passages do not contain information about this. "
                 "Context passages: [1] (Biology 01.pdf, page 1) " + BIOLOGY[0])
    body = client.post("/api/chat/ask", json={"question": "Where does photosynthesis take place?",
                                               "subject_id": doc["subject_id"]}).json()
    assert llm.calls and body["grounded"] is False and body["sources"] == []


def test_echoed_page_headers_are_removed_from_answer(client, pdfs, llm):
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    llm.reply = "[1] (Biology 01.pdf, page 1) Photosynthesis takes place in the chloroplasts. [1]"
    body = client.post("/api/chat/ask", json={"question": "Where does photosynthesis take place?",
                                               "subject_id": doc["subject_id"]}).json()
    assert body["grounded"] is True and "page 1)" not in body["answer"]
    assert body["answer"].startswith("[1] Photosynthesis takes place in the chloroplasts")
