"""Retrieval-augmented question answering with a local LLM."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retrieval import RetrievedChunk, retrieve_context
from .runtime import Services, get_services

NOT_FOUND_TOKEN = "NOT_FOUND"
NOT_FOUND_MESSAGE = (
    "I couldn't find this in your uploaded materials. Try rephrasing the question, "
    "or upload notes that cover this topic."
)

SYSTEM_PROMPT = f"""You are Offline Scholar, a study assistant for students.
Answer the student's question using ONLY the numbered context passages provided.

Rules:
- Do not use outside knowledge and do not guess.
- After each claim, cite the passage(s) that support it like [1] or [2].
- If the passages do not contain the answer, reply with exactly: {NOT_FOUND_TOKEN}
- Write clearly and concisely for a student revising for an exam."""

_CITATION = re.compile(r"\[(\d+)\]")
_EXCERPT_CHARS = 400


@dataclass
class Source:
    document_name: str
    page_number: int
    excerpt: str
    score: float = 0.0
    document_id: int | None = None


@dataclass
class Answer:
    answer: str
    grounded: bool
    sources: list[Source] = field(default_factory=list)
    confidence: float = 0.0  # best retrieval cosine score (0 when nothing relevant was found)


def build_prompt(question: str, passages: list[RetrievedChunk]) -> str:
    blocks = [f"[{i}] ({p.document_name}, page {p.page_number})\n{p.text}" for i, p in enumerate(passages, 1)]
    return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question.strip()}\n\nAnswer:"


def _excerpt(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _EXCERPT_CHARS else text[:_EXCERPT_CHARS].rsplit(" ", 1)[0] + "…"


def _source(p: RetrievedChunk) -> Source:
    return Source(p.document_name, p.page_number, _excerpt(p.text), p.score, p.document_id)


def _is_refusal(text: str) -> bool:
    stripped = text.strip().strip(".!`'\" ")
    return not stripped or stripped.upper() == NOT_FOUND_TOKEN or (
        NOT_FOUND_TOKEN in text.upper() and len(stripped) < 60
    )


def answer_question(
    question: str,
    subject_id: int | None = None,
    top_k: int = 4,
    services: Services | None = None,
) -> Answer:
    """Retrieve passages, ask the local LLM, and return a cited answer.

    `grounded` is True only when relevant passages were retrieved AND the model answered from them
    (did not abstain). When nothing relevant exists the LLM is not called at all, so it cannot
    hallucinate an answer.
    Raises LLMUnavailableError if Ollama is unreachable.
    """
    svc = services or get_services()
    passages = retrieve_context(question, subject_id, top_k, services=svc)
    if not passages:
        return Answer(NOT_FOUND_MESSAGE, grounded=False, sources=[], confidence=0.0)

    raw = svc.llm.chat(SYSTEM_PROMPT, build_prompt(question, passages))
    if _is_refusal(raw):
        return Answer(NOT_FOUND_MESSAGE, grounded=False, sources=[], confidence=passages[0].score)

    text = raw.replace(NOT_FOUND_TOKEN, "").strip()
    cited = sorted({int(n) for n in _CITATION.findall(text) if 1 <= int(n) <= len(passages)})
    used = [passages[n - 1] for n in cited] or passages  # no valid citations -> show everything retrieved
    return Answer(text, grounded=True, sources=[_source(p) for p in used], confidence=passages[0].score)
