"""MODULE 1b — Semantic similarity between questions and similar-question grouping.

Public API
----------
group_similar_questions(questions, embedder=None, threshold=None) -> dict
find_similar_questions(query, questions, embedder=None, top_k=5) -> list[dict]

Grouping uses *average-linkage* agglomerative clustering on cosine distance, so a
group only forms when its members are similar to each other on average (this avoids
the "chaining" problem of plain connected components, where A~B and B~C would pull
an unrelated C into A's group).

A group is flagged ``repeated`` when its members come from two or more *different*
papers. Two similar questions inside one paper are reported but do not count as a
repeat across exams.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Iterable, Sequence

import numpy as np

from .embeddings import cosine_matrix, embed_all, get_embedder, thresholds_for
from .text_utils import content_terms, normalise_text, tokenize, is_content_token, stem


def _as_records(questions: Iterable) -> list[dict]:
    recs = []
    for i, q in enumerate(questions):
        if isinstance(q, str):
            q = {"text": q}
        q = dict(q)
        q.setdefault("question_id", f"Q{i + 1}")
        q.setdefault("document_id", None)
        q.setdefault("document_name", None)
        recs.append(q)
    return recs


def _paper_key(q: dict) -> str:
    return str(q.get("document_id") or q.get("document_name") or "paper")


def _paper_label(q: dict) -> str:
    return str(q.get("document_name") or q.get("document_id") or "paper")


def _surface_for_stems(texts: Sequence[str]) -> dict[str, str]:
    surf: dict[str, str] = {}
    for t in texts:
        for tok in tokenize(t):
            if is_content_token(tok):
                surf.setdefault(stem(tok), tok)
    return surf


def shared_key_terms(texts: Sequence[str], max_terms: int = 6) -> list[str]:
    """Key terms common to (all, else most) of the texts, in readable surface form."""
    if not texts:
        return []
    counts: Counter = Counter()
    for t in texts:
        counts.update(set(content_terms(t)))
    n = len(texts)
    surf = _surface_for_stems(texts)
    common = [s for s, c in counts.items() if c == n]
    if not common and n > 2:
        common = [s for s, c in counts.items() if c >= max(2, (n + 1) // 2)]
    common.sort(key=lambda s: (-counts[s], -len(s)))
    return [surf.get(s, s) for s in common[:max_terms]]


def _cluster(sim: np.ndarray, threshold: float) -> np.ndarray:
    n = sim.shape[0]
    if n == 1:
        return np.zeros(1, dtype=int)
    from sklearn.cluster import AgglomerativeClustering

    dist = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    model = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=1.0 - threshold,
    )
    return model.fit_predict(dist)


def group_similar_questions(questions: Sequence, embedder=None, threshold: float | None = None) -> dict:
    """Group questions that ask for the same thing even when worded differently.

    ``questions`` may be strings or the records from ``extract_questions``.
    Returns a JSON-serialisable dict (see module docstring and docs/JSON_CONTRACTS.md).
    """
    recs = _as_records(questions)
    emb = get_embedder(embedder)
    th = thresholds_for(emb)
    thr = float(threshold) if threshold is not None else th["question_group"]
    n = len(recs)
    papers = {_paper_key(q) for q in recs}
    result = {
        "embedder": emb.kind, "threshold": thr, "total_questions": n,
        "total_papers": len(papers), "groups": [], "unique_questions": [],
        "summary": {},
    }
    if n == 0:
        result["summary"] = {"repeated_groups": 0, "questions_in_groups": 0}
        return result

    (vecs,) = embed_all(emb, [q["text"] for q in recs])
    sim = cosine_matrix(vecs)
    labels = _cluster(sim, thr)

    members_by_label: "OrderedDict[int, list[int]]" = OrderedDict()
    for idx, lab in enumerate(labels):
        members_by_label.setdefault(int(lab), []).append(idx)

    groups = []
    for idxs in members_by_label.values():
        if len(idxs) == 1:
            q = recs[idxs[0]]
            result["unique_questions"].append(_brief(q))
            continue
        sub = sim[np.ix_(idxs, idxs)]
        centrality = (sub.sum(axis=1) - 1.0) / (len(idxs) - 1)
        rep_local = int(np.argmax(centrality))
        rep = idxs[rep_local]
        pair_vals = sub[np.triu_indices(len(idxs), 1)]
        norm_texts = {normalise_text(recs[i]["text"]) for i in idxs}
        if len(norm_texts) == 1:
            match_type = "exact"
        elif pair_vals.min() >= th["question_near_duplicate"]:
            match_type = "near_duplicate"
        else:
            match_type = "reworded"
        paper_keys = {_paper_key(recs[i]) for i in idxs}
        paper_labels = sorted({_paper_label(recs[i]) for i in idxs})
        years = sorted({recs[i].get("year") for i in idxs if recs[i].get("year")})
        terms = shared_key_terms([recs[i]["text"] for i in idxs])
        members = []
        for i in sorted(idxs, key=lambda k: (recs[k].get("year") or 0, k)):
            m = _brief(recs[i])
            m["similarity_to_representative"] = round(float(sim[i, rep]), 3)
            members.append(m)
        repeated = len(paper_keys) >= 2
        groups.append({
            "representative": recs[rep]["text"],
            "size": len(idxs),
            "paper_count": len(paper_keys),
            "papers": paper_labels,
            "years": years,
            "repeated": repeated,
            "match_type": match_type,
            "avg_similarity": round(float(pair_vals.mean()), 3),
            "min_similarity": round(float(pair_vals.min()), 3),
            "shared_key_terms": terms,
            "members": members,
            "explanation": _explain(len(idxs), len(paper_keys), years, match_type, terms),
        })

    groups.sort(key=lambda g: (-g["paper_count"], -g["size"], -g["avg_similarity"]))
    result["groups"] = [{"group_id": f"G{i}", **g} for i, g in enumerate(groups, 1)]
    result["summary"] = {
        "repeated_groups": sum(1 for g in result["groups"] if g["repeated"]),
        "questions_in_groups": sum(g["size"] for g in result["groups"]),
        "unique_question_count": len(result["unique_questions"]),
    }
    return result


def _brief(q: dict) -> dict:
    return {k: q.get(k) for k in ("question_id", "number", "text", "marks", "document_id",
                                   "document_name", "page_number", "year")}


def _explain(size: int, papers: int, years: list, match_type: str, terms: list[str]) -> str:
    how = {"exact": "are worded identically", "near_duplicate": "are almost identical",
           "reworded": "are worded differently but ask about the same thing"}[match_type]
    where = (f"{papers} different papers" + (f" ({', '.join(map(str, years))})" if years else "")
             if papers > 1 else "the same paper")
    key = f" Shared key terms: {', '.join(terms)}." if terms else ""
    return f"{size} questions from {where} {how}.{key}"


def find_similar_questions(query: str, questions: Sequence, embedder=None, top_k: int = 5,
                           min_similarity: float | None = None) -> list[dict]:
    """Rank stored questions by similarity to a new question (e.g. a student's search)."""
    recs = _as_records(questions)
    if not recs:
        return []
    emb = get_embedder(embedder)
    vecs, qv = embed_all(emb, [r["text"] for r in recs], [query])
    sims = cosine_matrix(qv, vecs)[0]
    floor = thresholds_for(emb)["question_group"] * 0.8 if min_similarity is None else min_similarity
    order = np.argsort(-sims)[:top_k]
    out = []
    for i in order:
        if sims[i] >= floor:
            b = _brief(recs[int(i)])
            b["similarity"] = round(float(sims[i]), 3)
            out.append(b)
    return out
