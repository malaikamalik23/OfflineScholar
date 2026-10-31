"""Retrieval-augmented question answering with a local LLM."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .retrieval import RetrievedChunk, retrieve_context
from .runtime import Services, get_services

log = logging.getLogger(__name__)

NOT_FOUND_TOKEN = "NOT_FOUND"
NOT_FOUND_MESSAGE = (
    "I couldn't find this in your uploaded materials. Try rephrasing the question, "
    "or upload notes that cover this topic."
)

# Tuned on real lecture slides with qwen2.5:3b (see docs/EVALUATION.md). The earlier, stricter prompt
# refused answers that were present but worded differently, and quoted numbers from the wrong passage.
SYSTEM_PROMPT = f"""You are Offline Scholar, a study assistant for students.
Answer the student's question using ONLY the numbered context passages, which come from the student's own
lecture notes (some text was read from images and may contain recognition errors).

Rules:
- Use only facts written in the passages. Do not add outside knowledge, examples or details that the
  passages do not contain, even if you know them.
- The passages may use different words than the question (for example "feature scaling" for
  "normalization"). If a passage answers the question, answer it.
- If the passages mention the topic but do not contain what is asked, reply {NOT_FOUND_TOKEN}.
- Copy numbers, names and formulas exactly as they appear. Never calculate or guess a value.
- After each fact, cite the passage it comes from, like [1] or [2].
- Be brief: at most 4 sentences or a short bullet list.
- If no passage answers the question, reply with exactly {NOT_FOUND_TOKEN}."""

_CITATION = re.compile(r"\[(\d+)\]")
_EXCERPT_CHARS = 400
_WORD = re.compile(r"[^\W\d_][\w-]+|\d+(?:\.\d+)?")
_LATEX = re.compile(r"\\[a-z]+")
_SUPPORT_STOP = frozenset("""
a an the of to in on at for and or is are was were be been being it its this that these those with as by
from what which who whom how why when where do does did can could should would will may might must not no
yes into than then there their they them you your such also has have had more most less many much each
other some any all only both used use uses using called because so if but while about between over under
per one two three way ways example examples passage passages answer question""".split())


def _content_words(text: str) -> set[str]:
    text = _LATEX.sub(" ", text.lower())  # models rewrite garbled OCR formulas as LaTeX: \sum, P(e_i)
    words = set()
    for w in _WORD.findall(text):
        if "_" in w:
            continue
        if w in _SUPPORT_STOP or (len(w) < 3 and not w[0].isdigit()):
            continue
        words.add(w[:-1] if len(w) > 4 and w.endswith("s") else w)  # crude plural folding
    return words


def answer_support(answer: str, question: str, passages: list[RetrievedChunk]) -> float:
    """Fraction of the answer's content words that occur in the question or the given passages.

    A cheap, deterministic hallucination guard. Measured on real lecture slides (qwen2.5:3b, llama3.2:3b,
    phi3:mini; echoed passages cut first): 132 faithful answers scored >= 0.62, the 5 answers built from
    outside knowledge 0.12-0.50. See docs/EVALUATION.md.
    """
    words = _content_words(_CITATION.sub(" ", answer))
    if not words:
        return 1.0
    context = _content_words(question + " " + " ".join(p.text for p in passages))
    return len(words & context) / len(words)


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


# Some models (phi3:mini, llama3.2:3b) paste the prompt's passages back into the answer. The copied text
# matches the passages perfectly and hid a general-knowledge answer from the support check, so it is cut.
_ECHOED_BLOCK = re.compile(r"context passages:|\[\d+\]\s*\([^)\n]*,\s*page\s*\d+\)", re.IGNORECASE)
_PAGE_HEADER = re.compile(r"\(\s*[^()\n]*?\.pdf\s*,\s*page\s*\d+\s*\)", re.IGNORECASE)
# A refusal written in words instead of NOT_FOUND ("I'm sorry, the passages do not contain ...").
_PROSE_REFUSAL = re.compile(
    r"\b(passages?|context|materials?|notes)\b[^.]{0,40}\b(do|does|did)\s+not\s+(contain|mention|include|provide|"
    r"cover|discuss|explain)|\bno (relevant )?information (about|on)\b|\bi'?m sorry\b", re.IGNORECASE)


def _strip_echo(text: str) -> str:
    m = _ECHOED_BLOCK.search(text)
    if m and m.start() > 0:  # an echoed block after the answer: drop it
        text = text[:m.start()]
    return re.sub(r"[ \t]{2,}", " ", _PAGE_HEADER.sub(" ", text)).strip()


def _is_refusal(text: str) -> bool:
    stripped = text.strip().strip(".!`'\" ")
    if not stripped or stripped.upper() == NOT_FOUND_TOKEN:
        return True
    # "The passages don't mention X. NOT_FOUND" must not become a grounded answer once the token is
    # stripped: the token without any citation means the model abstained.
    if NOT_FOUND_TOKEN in text.upper() and (len(stripped) < 60 or not _CITATION.search(text)):
        return True
    first_sentence = re.split(r"(?<=[.!?])\s+", stripped, maxsplit=1)[0]
    return bool(_PROSE_REFUSAL.search(first_sentence))


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

    text = _strip_echo(raw.replace(NOT_FOUND_TOKEN, ""))
    if not text:
        return Answer(NOT_FOUND_MESSAGE, grounded=False, sources=[], confidence=passages[0].score)
    cited = sorted({int(n) for n in _CITATION.findall(text) if 1 <= int(n) <= len(passages)})
    used = [passages[n - 1] for n in cited] or passages  # no valid citations -> show everything retrieved
    support = answer_support(text, question, used)
    if support < svc.settings.min_answer_support:
        # The model wrote things its cited passages don't contain: don't present it as grounded.
        log.info("Answer rejected as unsupported by its passages (support %.2f): %r", support, text[:200])
        return Answer(NOT_FOUND_MESSAGE, grounded=False, sources=[], confidence=passages[0].score)
    return Answer(text, grounded=True, sources=[_source(p) for p in used], confidence=passages[0].score)
