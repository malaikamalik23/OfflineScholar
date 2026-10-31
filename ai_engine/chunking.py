"""Sentence-aware chunking. Chunks never cross page boundaries, so page citations stay exact."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .extraction import PageText

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
MIN_CHUNK_CHARS = 20


@dataclass
class TextChunk:
    page_number: int
    chunk_index: int  # order within the whole document
    text: str


def _units(text: str, limit: int) -> list[str]:
    """Split page text into sentence-sized units, each no longer than `limit`."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        for sent in _SENTENCE_END.split(para):
            while len(sent) > limit:  # very long run-on text: cut on a space
                cut = sent.rfind(" ", 0, limit)
                cut = cut if cut > limit // 2 else limit
                out.append(sent[:cut].strip())
                sent = sent[cut:].strip()
            if sent:
                out.append(sent)
    return out


def _split_page(text: str, size: int, overlap: int) -> list[str]:
    units = _units(text, size)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for u in units:
        add = len(u) + (1 if cur else 0)
        if cur and cur_len + add > size:
            chunks.append(" ".join(cur))
            tail: list[str] = []  # carry trailing sentences forward as overlap
            tail_len = 0
            for prev in reversed(cur):
                if tail_len + len(prev) + 1 > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            cur, cur_len = tail, sum(len(t) + 1 for t in tail) - (1 if tail else 0)
            add = len(u) + (1 if cur else 0)
        cur.append(u)
        cur_len += add
    if cur:
        last = " ".join(cur)
        if not chunks or last != chunks[-1]:
            chunks.append(last)
    return chunks


def chunk_pages(pages: list[PageText], chunk_size: int = 900, overlap: int = 150) -> list[TextChunk]:
    if chunk_size < 100:
        raise ValueError("chunk_size must be >= 100")
    overlap = max(0, min(overlap, chunk_size // 2))
    result: list[TextChunk] = []
    for page in pages:
        pieces = _split_page(page.text, chunk_size, overlap)
        for piece in pieces:
            if len(piece) >= MIN_CHUNK_CHARS or len(pieces) == 1:
                result.append(TextChunk(page.page_number, len(result), piece))
    return result
