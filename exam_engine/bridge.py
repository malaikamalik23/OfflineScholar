"""Adapters between exam_engine and Member 1's backend (ai_engine + SQLAlchemy models).

Everything that touches ``backend`` / ``ai_engine`` is imported lazily inside functions, so the
core engine (all other modules) still imports and tests stand-alone.

Why past papers are re-read from the stored PDF
-----------------------------------------------
Chunks are for retrieval: ``ai_engine.chunking`` joins every paragraph into one line, so a
chunked past paper has no line breaks and question numbers can no longer be told apart from
numbers in the middle of a sentence. ``Document.stored_path`` + ``extract_document`` give back
the page text with its original lines. If the file is gone, we fall back to the chunks:
de-duplicate the chunk overlap and re-insert line breaks before inline question numbers.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Iterable

from .embeddings import CallableEmbedder, TfidfEmbedder
from .service import ExamEngine

log = logging.getLogger(__name__)

# (document id, sha256) -> pages. A given upload never changes, so extraction (and OCR) runs once.
_PAGE_CACHE: dict[tuple[int, str], list[dict]] = {}


# ----------------------------------------------------------------------------- engine
class OllamaQuizLLM:
    """``llm(prompt) -> str`` on top of Member 1's ``LLMClient.chat(system, user)``."""

    SYSTEM = ("You are a careful exam-question writer. Use ONLY the SOURCE text given by the user. "
              "Reply with the requested JSON and nothing else.")

    def __init__(self, client) -> None:
        self.client = client

    def __call__(self, prompt: str) -> str:
        return self.client.chat(self.SYSTEM, prompt)


def build_engine(use_llm: bool = True) -> ExamEngine:
    """A fresh engine wired to the shared embedder / LLM (cheap; one per request, so no shared state)."""
    from ai_engine.runtime import get_services

    svc = get_services()
    emb = svc.embedder
    if str(getattr(emb, "name", "")).startswith("hashing"):
        embedder = TfidfEmbedder()  # the hashing embedder is a test stub, not semantic: TF-IDF is better
    else:
        embedder = CallableEmbedder(emb, kind="dense")
    return ExamEngine(embedder=embedder, llm=OllamaQuizLLM(svc.llm) if use_llm else None)


# ----------------------------------------------------------------------------- database readers
def _documents(db, subject_id: int, doc_type: str):
    from backend.models import Document

    return (db.query(Document)
            .filter(Document.subject_id == subject_id, Document.doc_type == doc_type, Document.status == "ready")
            .order_by(Document.id).all())


def load_chunks(db, subject_id: int, doc_type: str) -> list[dict]:
    """Chunk records in the agreed {document_id, document_name, page_number, text} format."""
    from backend.models import Chunk, Document

    rows = (db.query(Chunk, Document).join(Document, Chunk.document_id == Document.id)
            .filter(Document.subject_id == subject_id, Document.doc_type == doc_type, Document.status == "ready")
            .order_by(Document.id, Chunk.chunk_index).all())
    return [{"document_id": d.id, "document_name": d.filename, "page_number": c.page_number, "text": c.text}
            for c, d in rows]


def load_pages(db, subject_id: int, doc_type: str) -> tuple[list[dict], list[str]]:
    """Page records WITH line breaks for every ready document of this type, plus user-facing notes."""
    from ai_engine.extraction import extract_document
    from ai_engine.runtime import get_services

    cfg = get_services().settings
    pages: list[dict] = []
    notes: list[str] = []
    for d in _documents(db, subject_id, doc_type):
        key = (d.id, d.sha256)
        if key not in _PAGE_CACHE:
            try:
                ex = extract_document(d.stored_path, ocr_enabled=cfg.ocr_enabled,
                                      ocr_language=cfg.ocr_language, ocr_dpi=cfg.ocr_dpi)
                _PAGE_CACHE[key] = [{"document_id": d.id, "document_name": d.filename,
                                     "page_number": p.page_number, "text": p.text} for p in ex.pages]
            except Exception as exc:  # file moved/deleted, unreadable ...
                log.warning("Re-reading %s failed (%s); using stored chunks", d.filename, exc)
                notes.append(f"'{d.filename}': the original file could not be re-read, so question/topic "
                             "boundaries were estimated from the stored text and may be less accurate.")
                pages.extend(pages_from_chunks(load_chunks_for_document(db, d)))
                continue
        pages.extend(_PAGE_CACHE[key])
    return pages, notes


def load_chunks_for_document(db, doc) -> list[dict]:
    from backend.models import Chunk

    rows = db.query(Chunk).filter(Chunk.document_id == doc.id).order_by(Chunk.chunk_index).all()
    return [{"document_id": doc.id, "document_name": doc.filename, "page_number": c.page_number, "text": c.text}
            for c in rows]


def load_quiz_records(db, subject_id: int) -> list[dict]:
    """Every stored quiz answer as {topic, is_correct, question_id, timestamp}."""
    from backend.models import QuizAnswer, QuizAttempt

    rows = (db.query(QuizAnswer, QuizAttempt).join(QuizAttempt, QuizAnswer.attempt_id == QuizAttempt.id)
            .filter(QuizAttempt.subject_id == subject_id)
            .order_by(QuizAttempt.created_at, QuizAnswer.id).all())
    return [{"topic": a.topic, "is_correct": bool(a.correct), "question_id": a.question_ref,
             "timestamp": t.created_at.isoformat() if t.created_at else None} for a, t in rows]


# ----------------------------------------------------------------------------- chunk fallback
_INLINE_BREAK = re.compile(
    r"(?<=[.?!)\]])\s+(?="
    r"(?:(?:Q(?:uestion)?|Ques)\.?\s*)?\d{1,2}\s*[.):]\s+(?=[A-Z(])"      # "3. Explain", "Q4) What"
    r"|\(?[a-h]\)\s+(?=[A-Z])"                                           # "(b) Explain"
    r"|(?:Section|Part)\s+[A-Z0-9]{1,3}\b)"                              # "Section B"
)


def _merge_overlap(prev: str, nxt: str, min_overlap: int = 20) -> str:
    """Join two consecutive chunks of one page, dropping the sentences ai_engine.chunking repeats."""
    for k in range(min(len(prev), len(nxt)), min_overlap - 1, -1):
        if prev.endswith(nxt[:k]):
            return prev + " " + nxt[k:].lstrip()
    return prev + " " + nxt


def pages_from_chunks(chunks: Iterable[dict]) -> list[dict]:
    """Rebuild approximate page text (with line breaks) from overlapping chunks."""
    by_page: dict[tuple, str] = {}
    meta: dict[tuple, dict] = {}
    for c in chunks:
        key = (c["document_id"], c["page_number"])
        by_page[key] = _merge_overlap(by_page[key], c["text"]) if key in by_page else c["text"]
        meta.setdefault(key, c)
    out = []
    for key in sorted(by_page, key=lambda k: (str(k[0]), k[1] or 0)):
        text = _INLINE_BREAK.sub("\n", by_page[key])
        out.append({**{k: meta[key][k] for k in ("document_id", "document_name", "page_number")}, "text": text})
    return out


_SYL_HEADER = re.compile(r"^\s*(?:topics(?:\s+covered)?|course\s+(?:outline|contents?)|calendar|schedule|syllabus\s+contents?)\s*:?\s*$", re.I)
_SYL_NOISE = re.compile(
    r"https?:|www\.|isbn|\bpp?\.\s*\d|\bproblem\s+sets?\b|\bprojects?\s+(?:due|out)\b|\bexams?\s+back\b|"
    r"\b(?:due|out)\b\s*$|\blec(?:ture)?\s*#|key\s+dates|\[\s*\d+\s*[ap]m\s*\]|"
    r"^\s*(?:menu|more info|feedback|description|grading|activities|percentages|text|exams?|project|"
    r"midterm|final|calendar|topics|syllabus|lectures?|problem sets)\s*$|recommended citation|"
    r"creative commons|terms and conditions|accessibility|proud member|freely sharing|learn more|courses?\s*&\s*materials|over\s+\d|©|\bmit opencourseware\b|downloaded on|collaborat|\bgrade\b|\bpercent|\d{1,2}/\d{1,2}/\d{2,4}",
    re.I)
_SYL_FILLER = {"introduction", "wrap up", "overview", "review", "guest lecture", "recap", "summary", "conclusion"}


def _clean_syllabus_text(text: str) -> str:
    """Make a syllabus that was printed from a web page / PDF parse as a plain topic list.

    If a 'Topics' / 'Calendar' / 'Course outline' heading exists, only what follows it is used (the part
    of the page before it is usually logistics: grading, textbooks...). Lines that are clearly not topics
    (URLs, ISBNs, problem-set dates, ...) are dropped, and 'A, B, C' lecture titles become separate topics.
    """
    text = unicodedata.normalize("NFKC", text or "")
    lines = text.splitlines()
    heads = [i for i, ln in enumerate(lines) if _SYL_HEADER.match(ln)]
    if heads:
        lines = lines[heads[-1] + 1:]
    out: list[str] = []
    for ln in lines:
        ln = re.sub(r"\(\s*(?:cont\.?|continued)\s*\)", "", ln, flags=re.I).strip()
        if not ln or _SYL_NOISE.search(ln):
            continue
        if ":" not in ln and "," in ln:
            parts = [p.strip(" .;") for p in ln.split(",")]
        else:
            parts = [ln]
        for p in parts:
            p = re.sub(r"^(?:and|or)\s+", "", p, flags=re.I)
            if len(p) >= 4 and p.lower().strip(" .") not in _SYL_FILLER and not re.fullmatch(r"[\d\W]+", p):
                out.append(p)
    return "\n".join(out)


def syllabus_topics_from(pages: list[dict]) -> list[dict]:
    """Syllabus topics from page text (one topic per line; 'Topic: sub, sub' gives subtopics)."""
    from .syllabus_coverage import parse_syllabus_text

    topics = parse_syllabus_text(_clean_syllabus_text("\n".join(p["text"] for p in pages)))
    seen, out = set(), []
    for t in topics:
        name = t["name"].strip()
        if 3 <= len(name) <= 120 and name.lower() not in seen and len(name.split()) <= 14:
            seen.add(name.lower())
            out.append(t)
    return out[:80]
