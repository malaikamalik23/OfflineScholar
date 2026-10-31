"""MODULE 1a — Question extraction from processed past-paper text.

Public API
----------
extract_questions(text, *, document_id=None, document_name=None, page_number=None) -> list[dict]
extract_questions_from_chunks(chunks) -> list[dict]      # Member 1's chunk format

Output question record (JSON-serialisable)::

    {
      "question_id": "doc_001-Q3a",
      "number": "3a",              # as printed (sub-parts are separate questions)
      "text": "Explain the working of a neural network.",
      "marks": 5 | null,
      "section": "Section B" | "",
      "question_type": "explain" | "define" | "compare" | "list" | "numerical" | "design" | "short_note" | "other",
      "document_id": "doc_001", "document_name": "Past Paper 2025.pdf",
      "page_number": 3,            # page where the question starts
      "year": 2025 | null,         # parsed from the document name if present
      "confidence": 0.0-1.0        # how sure we are this is a real question
    }

The extractor is rule-based and explainable. It handles ``1.``, ``1)``, ``Q1.``,
``Question 1:``, ``Q.1``, sub-parts ``(a)``/``a)``/``(i)``, marks written as
``[5 marks]``, ``(5)``, ``5M``, and per-section defaults such as
"Section A (2 marks each)". When a paper has no numbering at all it falls back to
sentences that end in ``?`` or begin with an exam command word.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Iterable

from .text_utils import COMMAND_WORDS, content_terms, split_sentences

_MAIN_RE = re.compile(
    r"^\s*(?:(?:question|ques|qn|q)\s*\.?\s*)?(\d{1,3})\s*(?:[.):\-](?!\d)\s*|\s+(?=\([a-h]\)))(.*)$",
    re.IGNORECASE,
)
_SUB_RE = re.compile(r"^\s*\(?([a-h]|iv|vi{0,3}|i{1,3}|ix|x)\)\s*(.*)$", re.IGNORECASE)
_SECTION_RE = re.compile(r"^\s*(?:part|section)\s+([A-Za-z0-9]{1,4})\b(.*)$", re.IGNORECASE)
_EACH_MARKS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*marks?\s*each", re.IGNORECASE)

_INSTRUCTION_RE = re.compile(
    r"(time\s*allowed|time\s*:|max(imum)?\.?\s*marks|total\s*marks|roll\s*(no|number)|"
    r"registration\s*(no|number)|reg\.?\s*no|^\s*name\s*:|^\s*instructions?\b\s*:?|"
    r"attempt\s+(all|any|\w+\s+questions)|answer\s+(all|any)|all\s+questions\s+(are\s+)?compulsory|"
    r"candidates\s+(must|should)|use\s+of\s+(calculator|tables)|invigilator|"
    r"your\s+name|\bmit\s+id\b|student\s+id|grade\s+you\s+would\s+give|"
    r"^\s*note\s*[:\-]|^\s*\*+\s*(end|best)|^\s*[-=_*]{3,}\s*$)",
    re.IGNORECASE,
)
_PAGE_NOISE_RE = re.compile(
    r"^\s*(page\s*\d+(\s*of\s*\d+)?|-\s*\d+\s*-|\d+\s*/\s*\d+|turn\s*over|p\.?\s*t\.?\s*o\.?|continued)\s*$",
    re.IGNORECASE,
)
_MARK_PATTERNS = [
    re.compile(r"[\[(]\s*(\d+(?:\.\d+)?)\s*(?:marks?|mks?|m)\s*[\])]\s*[.]?\s*$", re.IGNORECASE),
    re.compile(r"[\[(]\s*(\d+(?:\.\d+)?)\s*[\])]\s*[.]?\s*$"),
    re.compile(r"[-–—,]?\s*\b(\d+(?:\.\d+)?)\s*(?:marks?|mks?)\s*[.]?\s*$", re.IGNORECASE),
    re.compile(r"\bmarks?\s*[:=]\s*(\d+(?:\.\d+)?)\s*$", re.IGNORECASE),
]
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")

_TYPE_RULES = [
    ("compare", re.compile(r"\b(compare|differentiate|distinguish|contrast|difference)\b", re.I)),
    ("define", re.compile(r"\b(define|what\s+is|what\s+are|meaning\s+of)\b", re.I)),
    ("list", re.compile(r"\b(list|enumerate|mention|state|name|identify)\b", re.I)),
    ("numerical", re.compile(r"\b(calculate|compute|solve|derive|prove|find\s+the|determine|evaluate)\b", re.I)),
    ("design", re.compile(r"\b(design|develop|implement|write\s+a\s+(program|code|algorithm)|construct|propose)\b", re.I)),
    ("short_note", re.compile(r"\b(short\s+notes?|write\s+notes?\s+on|brief\s+note)\b", re.I)),
    ("explain", re.compile(r"\b(explain|describe|discuss|illustrate|elaborate|outline|how\s+does|how\s+do|why)\b", re.I)),
]


def _num(x: str):
    v = float(x)
    return int(v) if v == int(v) else v


def classify_question(text: str) -> str:
    """Earliest-matching command word decides the type."""
    best, best_pos = "other", None
    for label, pat in _TYPE_RULES:
        m = pat.search(text)
        if m and (best_pos is None or m.start() < best_pos):
            best, best_pos = label, m.start()
    if best == "other" and text.strip().endswith("?"):
        return "explain"
    return best


def _extract_marks(text: str) -> tuple[float | None, str]:
    stripped = text.rstrip()
    for pat in _MARK_PATTERNS:
        m = pat.search(stripped)
        if m:
            val = float(m.group(1))
            if 0 < val <= 100:
                cleaned = stripped[: m.start()].rstrip(" -–—,")
                return (int(val) if val == int(val) else val), cleaned
    return None, stripped


def _looks_like_question(text: str) -> bool:
    words = text.split()
    if len(words) < 3 or not content_terms(text):
        return False
    first = re.sub(r"[^a-z]", "", words[0].lower())
    return (
        text.rstrip().endswith("?")
        or first in COMMAND_WORDS
        or bool(re.search(r"\b(explain|describe|discuss|define|compare|what|why|how|write|list|differentiate|derive|calculate|state)\b", text, re.I))
    )


def _tidy(text: str) -> str:
    text = re.sub(r"(?<=\w)-\s+(?=[a-z])", "", text)  # de-hyphenate line wraps
    return re.sub(r"\s+", " ", text).strip()


def _doc_year(document_name: str | None) -> int | None:
    if not document_name:
        return None
    m = _YEAR_RE.search(document_name)
    return int(m.group(1)) if m else None


def _scan(lines: list[tuple[str, int | None]]) -> list[dict]:
    """Core line scanner. ``lines`` = [(line_text, page_number), ...]."""
    blocks: list[dict] = []
    section = ""
    section_default_marks: float | None = None
    cur: dict | None = None
    cur_sub: dict | None = None

    def close():
        nonlocal cur, cur_sub
        cur, cur_sub = None, None

    for raw, page in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if _PAGE_NOISE_RE.match(line):
            continue
        sec = _SECTION_RE.match(line)
        if sec and len(line) < 140 and not _MAIN_RE.match(line):
            section = f"{'Part' if line.lower().lstrip().startswith('part') else 'Section'} {sec.group(1).upper()}"
            each = _EACH_MARKS_RE.search(line)
            section_default_marks = _num(each.group(1)) if each else None
            close()
            continue
        main_probe = _MAIN_RE.match(line)
        probe_text = main_probe.group(2) if main_probe else line
        if _INSTRUCTION_RE.search(probe_text) and len(probe_text) < 120:
            each = _EACH_MARKS_RE.search(line)
            if each:
                section_default_marks = _num(each.group(1))
            close()
            continue

        main = _MAIN_RE.match(line)
        if main:
            rest = main.group(2).strip()
            cur = {"number": main.group(1), "stem": [rest] if rest else [], "subs": [],
                   "page": page, "section": section, "default_marks": section_default_marks}
            cur_sub = None
            blocks.append(cur)
            sub = _SUB_RE.match(rest) if rest else None
            if sub:
                cur["stem"] = []
                cur_sub = {"label": sub.group(1).lower(), "parts": [sub.group(2).strip()], "page": page}
                cur["subs"].append(cur_sub)
            continue

        sub = _SUB_RE.match(line)
        if sub and cur is not None:
            cur_sub = {"label": sub.group(1).lower(), "parts": [sub.group(2).strip()], "page": page}
            cur["subs"].append(cur_sub)
            continue

        if cur is not None:
            (cur_sub["parts"] if cur_sub is not None else cur["stem"]).append(line.strip())

    # -- turn blocks into question records ---------------------------------
    out: list[dict] = []
    for b in blocks:
        stem = _tidy(" ".join(b["stem"]))
        stem_marks, stem_clean = _extract_marks(stem)
        default_marks = b["default_marks"]
        if not b["subs"]:
            marks, text = (stem_marks, stem_clean) if stem_marks is not None else _extract_marks(stem)
            out.append(_record(b["number"], text, marks if marks is not None else default_marks,
                               b["section"], b["page"], marks_explicit=marks is not None))
            continue
        if stem_clean and _looks_like_question(stem_clean) and len(stem_clean.split()) >= 5:
            out.append(_record(b["number"], stem_clean, stem_marks, b["section"], b["page"],
                               marks_explicit=stem_marks is not None))
        for s in b["subs"]:
            sub_text = _tidy(" ".join(s["parts"]))
            m, sub_text = _extract_marks(sub_text)
            if len(content_terms(sub_text)) < 3 and len(content_terms(stem_clean)) >= 3:
                sub_text = _tidy(f"{stem_clean} {sub_text}")
            out.append(_record(f"{b['number']}{s['label']}", sub_text,
                               m if m is not None else default_marks, b["section"], s["page"],
                               marks_explicit=m is not None))
    return [q for q in out if q is not None]


def _record(number, text, marks, section, page, marks_explicit) -> dict | None:
    text = text.strip(" \t-–—")
    words = text.split()
    first = re.sub(r"[^a-z]", "", words[0].lower()) if words else ""
    if len(words) < (2 if first in COMMAND_WORDS else 3) or not content_terms(text):
        return None
    conf = 0.5
    conf += 0.2 if text.rstrip().endswith("?") or classify_question(text) != "other" else 0
    conf += 0.1 if marks is not None and marks_explicit else 0
    conf += 0.1 if 6 <= len(text.split()) <= 80 else 0
    return {
        "number": str(number),
        "text": text,
        "marks": marks,
        "section": section,
        "question_type": classify_question(text),
        "page_number": page,
        "confidence": round(min(conf, 1.0), 2),
    }


def _finalise(records: list[dict], document_id, document_name) -> list[dict]:
    year = _doc_year(document_name)
    out = []
    for i, r in enumerate(records, 1):
        r = dict(r)
        r["document_id"] = document_id
        r["document_name"] = document_name
        r["year"] = year
        r["question_id"] = f"{document_id or 'doc'}-Q{r['number'] or i}"
        out.append(r)
    # make ids unique if numbering restarts by section
    seen: dict[str, int] = defaultdict(int)
    for r in out:
        seen[r["question_id"]] += 1
        if seen[r["question_id"]] > 1:
            tag = (r["section"] or f"dup{seen[r['question_id']]}").replace(" ", "")
            r["question_id"] = f"{r['question_id']}-{tag}"
    return out


def extract_questions(text: str, *, document_id: str | None = None,
                      document_name: str | None = None, page_number: int | None = None) -> list[dict]:
    """Extract and separate the questions in one block of paper text."""
    lines = [(ln, page_number) for ln in unicodedata.normalize("NFKC", text or "").splitlines()]
    records = _scan(lines)
    if not records:
        records = _fallback_unnumbered_safe(lines)
    return _finalise(records, document_id, document_name)


def _fallback_unnumbered_safe(lines):
    out = []
    joined = " ".join(l.strip() for l, _ in lines if l.strip() and not _INSTRUCTION_RE.search(l))
    page = next((p for _, p in lines), None)
    for i, sent in enumerate(split_sentences(joined), 1):
        if _looks_like_question(sent):
            marks, clean = _extract_marks(sent)
            rec = _record(str(i), clean, marks, "", page, marks_explicit=marks is not None)
            if rec:
                rec["confidence"] = round(min(rec["confidence"], 0.6), 2)  # unnumbered => less sure
                out.append(rec)
    return out


def extract_questions_from_chunks(chunks: Iterable[dict]) -> list[dict]:
    """Extract from Member 1's page records, keeping questions that span pages together.

    ``chunks`` = [{"document_id","document_name","page_number","text"}, ...]
    """
    by_doc: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for c in chunks:
        did = c.get("document_id") or c.get("document_name") or "doc"
        if did not in by_doc:
            order.append(did)
        by_doc[did].append(c)
    out: list[dict] = []
    for did in order:
        pages = sorted(by_doc[did], key=lambda c: (c.get("page_number") is None, c.get("page_number") or 0))
        lines: list[tuple[str, int | None]] = []
        for c in pages:
            for ln in unicodedata.normalize("NFKC", c.get("text") or "").splitlines():
                lines.append((ln, c.get("page_number")))
        name = pages[0].get("document_name")
        records = _scan(lines) or _fallback_unnumbered_safe(lines)
        out.extend(_finalise(records, did if did != "doc" else pages[0].get("document_id"), name))
    return out
