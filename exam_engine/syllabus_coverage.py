"""MODULE 3 — Syllabus coverage: syllabus topics vs lecture notes.

Public API
----------
analyze_syllabus_coverage(syllabus_topics, lecture_chunks, embedder=None) -> dict
parse_syllabus_text(text) -> list[dict]

``syllabus_topics``: list of ``"Topic name"`` strings or
``{"name": ..., "subtopics": [...], "description": "..."}``.
``lecture_chunks``: Member 1's records ``{document_id, document_name, page_number, text}``.

Per topic we return: ``topic``, ``status`` (Covered | Partially covered | Not covered),
``document`` + ``pages`` (best lecture source), ``evidence_score`` (0-1), ``explanation``
(plain language), plus the matched / missing key terms and a short evidence snippet.

Why a topic is *not* called missing just because a keyword is absent
--------------------------------------------------------------------
Evidence combines (a) embedding similarity — strong semantic matches count on their own,
(b) idf-weighted key-term coverage, and (c) *context*: terms spread over the top-3 most
related chunks (e.g. adjacent pages) are read together. Only when all three are weak is
a topic reported as Not covered, and the report names the closest lecture material so the
student can check it.

Thresholds (evidence score): >= 0.65 Covered, >= 0.35 Partially covered. See README
"Calibration" — with a dense embedder these should be tuned on a few real examples.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Sequence

import numpy as np

from .embeddings import get_embedder
from .matching import MatchMatrices, score_matches
from .text_utils import first_sentences, split_sentences, content_terms

COVERED_MIN = 0.65
PARTIAL_MIN = 0.35          # dense embedders
PARTIAL_MIN_LEXICAL = 0.40  # TF-IDF fallback: term overlap is noisier, so ask for a bit more
CONTEXT_CAP = 0.64  # evidence assembled only from spread-out context can reach Partial, never Covered

COVERED, PARTIAL, NOT_COVERED = "Covered", "Partially covered", "Not covered"


def parse_syllabus_text(text: str) -> list[dict]:
    """Turn a plain syllabus document into topics.

    One topic per line; ``Topic: sub1, sub2; sub3`` becomes a topic with subtopics.
    Leading bullets / numbering / 'Unit 3 -' prefixes are stripped.
    """
    topics: list[dict] = []
    for raw in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-•*·▪]|\d+(?:\.\d+)*[.)]?|(?:unit|chapter|week|module)\s*\w+\s*[:.\-–]?)\s*", "", raw, flags=re.I).strip()
        if len(line) < 3:
            continue
        if ":" in line:
            name, rest = line.split(":", 1)
            subs = [x.strip() for x in re.split(r"[;,]", rest) if len(x.strip()) > 2]
            topics.append({"name": name.strip(), "subtopics": subs})
        else:
            topics.append({"name": line.rstrip(".;,"), "subtopics": []})
    return topics


def _normalise_topics(syllabus_topics: Sequence) -> list[dict]:
    out = []
    for t in syllabus_topics:
        if isinstance(t, str):
            out.append({"name": t.strip(), "subtopics": [], "description": ""})
        else:
            out.append({"name": (t.get("name") or t.get("topic") or "").strip(),
                        "subtopics": list(t.get("subtopics") or []),
                        "description": t.get("description") or ""})
    return [t for t in out if t["name"]]


def _status(score: float, pmin: float = PARTIAL_MIN) -> str:
    return COVERED if score >= COVERED_MIN else PARTIAL if score >= pmin else NOT_COVERED


def _doc_label(c: dict) -> str:
    return str(c.get("document_name") or c.get("document_id") or "document")


def _best_snippet(chunk_text: str, terms: Sequence[str]) -> str:
    sents = split_sentences(chunk_text) or [chunk_text]
    tset = set(terms)
    best = max(sents, key=lambda s: len(tset & set(content_terms(s))))
    return first_sentences(best, 220)


def _evaluate(mm: MatchMatrices, t: int, chunks: Sequence[dict]) -> dict:
    """Evidence for topic row ``t`` of the match matrices."""
    comb = mm.combined[t]
    order = np.argsort(-comb)
    top = [int(i) for i in order[:3]]
    best = top[0]
    score = float(comb[best])
    union_lex = mm.lexical_union(t, top)
    context_used = False
    if union_lex > float(mm.lexical[t, best]) + 1e-6:
        boosted = min(CONTEXT_CAP, 0.5 * score + 0.5 * union_lex)
        if boosted > score:
            score, context_used = boosted, True
    matched, missing = mm.terms_found(t, top if context_used else best)
    return {"best": best, "top": top, "score": score, "matched": matched, "missing": missing,
            "semantic": float(mm.semantic[t, best]), "lexical": float(mm.lexical[t, best]),
            "context_used": context_used, "order": order}


def analyze_syllabus_coverage(syllabus_topics: Sequence, lecture_chunks: Sequence[dict], embedder=None) -> dict:
    """Classify every syllabus topic as Covered / Partially covered / Not covered."""
    topics = _normalise_topics(syllabus_topics)
    chunks = [c for c in lecture_chunks if (c.get("text") or "").strip()]
    emb = get_embedder(embedder)
    pmin = PARTIAL_MIN if emb.kind == "dense" else PARTIAL_MIN_LEXICAL
    result: dict = {"embedder": emb.kind, "thresholds": {"covered": COVERED_MIN, "partial": pmin},
                    "topics": [], "summary": {}}
    if not topics:
        result["summary"] = _summary([])
        return result
    if not chunks:
        for t in topics:
            result["topics"].append(_empty_row(t["name"]))
        result["summary"] = _summary(result["topics"])
        return result

    # whole-topic rows + one row per subtopic, embedded together in one space
    texts, owners = [], []
    for ti, t in enumerate(topics):
        texts.append(" ".join([t["name"], t["description"]]).strip()); owners.append((ti, None))
        for si, sub in enumerate(t["subtopics"]):
            texts.append(str(sub)); owners.append((ti, si))
    mm = score_matches(texts, [c["text"] for c in chunks], emb)
    row_of = {o: r for r, o in enumerate(owners)}

    for ti, t in enumerate(topics):
        main = _evaluate(mm, row_of[(ti, None)], chunks)
        sub_rows = []
        for si, sub in enumerate(t["subtopics"]):
            ev = _evaluate(mm, row_of[(ti, si)], chunks)
            sub_rows.append({"subtopic": sub, "status": _status(ev["score"], pmin),
                             "evidence_score": round(ev["score"], 2),
                             "document": _doc_label(chunks[ev["best"]]),
                             "page": chunks[ev["best"]].get("page_number")})
        score = main["score"]
        if sub_rows:
            credit = [1.0 if s["status"] == COVERED else 0.5 if s["status"] == PARTIAL else 0.0 for s in sub_rows]
            frac = sum(credit) / len(credit)
            status = COVERED if frac >= 0.8 and main["score"] >= pmin else PARTIAL if frac >= 0.25 or main["score"] >= pmin else NOT_COVERED
            if status == COVERED and main["score"] < COVERED_MIN:
                status = PARTIAL if frac < 0.8 else COVERED
            score = float(np.mean([main["score"]] + [s["evidence_score"] for s in sub_rows]))
        else:
            status = _status(score, pmin)

        rows_of_topic = [row_of[(ti, None)]] + [row_of[(ti, si)] for si in range(len(t["subtopics"]))]
        agg = mm.combined[rows_of_topic].max(axis=0)
        agg_order = np.argsort(-agg)
        best_i = int(agg_order[0])
        best_chunk = chunks[best_i]
        doc = _doc_label(best_chunk)
        relevant = [int(i) for i in agg_order if agg[i] >= pmin * 0.9] or [best_i]
        same_doc = [i for i in relevant if _doc_label(chunks[i]) == doc][:5]
        pages = sorted({chunks[i].get("page_number") for i in same_doc if chunks[i].get("page_number") is not None})
        others = OrderedDict()
        for i in relevant:
            if _doc_label(chunks[i]) != doc:
                others.setdefault(_doc_label(chunks[i]), set()).add(chunks[i].get("page_number"))
        row = {
            "topic": t["name"],
            "status": status,
            "document": doc if status != NOT_COVERED else None,
            "document_id": best_chunk.get("document_id") if status != NOT_COVERED else None,
            "pages": pages if status != NOT_COVERED else [],
            "evidence_score": round(score, 2),
            "semantic_similarity": round(main["semantic"], 2),
            "term_coverage": round(main["lexical"], 2),
            "matched_terms": main["matched"],
            "missing_terms": main["missing"],
            "other_documents": [{"document": d, "pages": sorted(p for p in ps if p is not None)} for d, ps in list(others.items())[:3]],
            "evidence_snippet": _best_snippet(best_chunk["text"], main["matched"]) if status != NOT_COVERED else "",
            "closest_material": {"document": doc, "page": best_chunk.get("page_number")} if status == NOT_COVERED else None,
            "subtopics": sub_rows,
        }
        row["explanation"] = _explain(row, main)
        result["topics"].append(row)
    result["summary"] = _summary(result["topics"])
    return result


def _empty_row(name: str) -> dict:
    return {"topic": name, "status": NOT_COVERED, "document": None, "document_id": None, "pages": [],
            "evidence_score": 0.0, "semantic_similarity": 0.0, "term_coverage": 0.0, "matched_terms": [],
            "missing_terms": [], "other_documents": [], "evidence_snippet": "", "closest_material": None,
            "subtopics": [], "explanation": "No lecture notes were uploaded to compare against."}


def _explain(row: dict, ev: dict) -> str:
    where = f"{row['document']}, page{'s' if len(row['pages']) > 1 else ''} {', '.join(map(str, row['pages']))}" if row["pages"] else row["document"]
    if row["status"] == COVERED:
        base = f"Your notes cover this: strong match in {where}"
        if row["matched_terms"]:
            base += f" (key terms found: {', '.join(row['matched_terms'][:5])})"
        if ev["context_used"]:
            base += ", gathered from neighbouring pages"
        return base + "."
    if row["status"] == PARTIAL:
        base = f"Partly covered in {where}."
        bad = [s["subtopic"] for s in row["subtopics"] if s["status"] != COVERED]
        if bad:
            base += f" Subtopics needing material: {', '.join(bad[:4])}."
        elif row["missing_terms"]:
            base += f" Not clearly explained: {', '.join(row['missing_terms'][:5])}."
        return base + " Consider adding notes or reviewing the textbook for the missing part."
    near = row["closest_material"]
    hint = f" The closest material is page {near['page']} of {near['document']}, but it is not similar enough to count." if near and near.get("document") else ""
    return "No lecture content in the uploaded notes addresses this topic." + hint


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    c = sum(r["status"] == COVERED for r in rows)
    p = sum(r["status"] == PARTIAL for r in rows)
    m = sum(r["status"] == NOT_COVERED for r in rows)
    return {"total_topics": n, "covered": c, "partially_covered": p, "not_covered": m,
            "coverage_pct": round(100 * (c + 0.5 * p) / n, 1) if n else 0.0,
            "gaps": [r["topic"] for r in rows if r["status"] != COVERED]}
