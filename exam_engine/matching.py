"""Evidence scoring shared by topic mapping (Module 2) and syllabus coverage (Module 3).

For a (topic, text) pair we combine two kinds of evidence:

* ``semantic``  — cosine similarity of embeddings (catches paraphrase / synonyms),
* ``lexical``   — idf-weighted share of the topic's key terms present in the text.

The combined score deliberately lets a *strong semantic match alone* count
(a topic must never be called "missing" just because an exact keyword is absent)
while still rewarding explicit term evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .embeddings import cosine_matrix, embed_all, thresholds_for
from .text_utils import content_terms, is_content_token, stem, tokenize


@dataclass
class MatchMatrices:
    semantic: np.ndarray      # (T, D) raw cosine
    semantic_norm: np.ndarray  # (T, D) cosine / semantic_ref, clipped to [0, 1]
    lexical: np.ndarray       # (T, D) idf-weighted term coverage in [0, 1]
    combined: np.ndarray      # (T, D) final evidence in [0, 1]
    topic_terms: list[list[str]]
    doc_terms: list[set[str]]
    weights: list[dict[str, float]]
    acronym_hits: set | None = None  # {(topic_idx, doc_idx)} matched via an acronym such as SVM

    def terms_found(self, t: int, d: int | Sequence[int]) -> tuple[list[str], list[str]]:
        """(matched, missing) topic terms for text(s) d."""
        docs = [d] if isinstance(d, (int, np.integer)) else list(d)
        if self.acronym_hits and any((t, i) in self.acronym_hits for i in docs):
            return list(self.topic_terms[t]), []
        found: set[str] = set()
        for i in docs:
            found |= self.doc_terms[i]
        terms = self.topic_terms[t]
        return [x for x in terms if x in found], [x for x in terms if x not in found]

    def lexical_union(self, t: int, docs: Sequence[int]) -> float:
        """Term coverage of topic ``t`` when several texts are read together."""
        terms = self.topic_terms[t]
        if not terms:
            return 0.0
        if self.acronym_hits and any((t, i) in self.acronym_hits for i in docs):
            return 1.0
        found: set[str] = set()
        for i in docs:
            found |= self.doc_terms[i]
        w = self.weights[t]
        total = sum(w[x] for x in terms)
        return sum(w[x] for x in terms if x in found) / total if total else 0.0


def score_matches(topic_texts: Sequence[str], doc_texts: Sequence[str], embedder) -> MatchMatrices:
    T, D = len(topic_texts), len(doc_texts)
    th = thresholds_for(embedder)
    tv, dv = embed_all(embedder, list(topic_texts), list(doc_texts))
    sem = cosine_matrix(tv, dv) if T and D else np.zeros((T, D), np.float32)
    sem_norm = np.clip(sem / th["semantic_ref"], 0.0, 1.0)

    doc_terms, doc_raw = [], []
    for t in doc_texts:
        toks = tokenize(t)
        terms = set(content_terms(t))
        # "trade-off" / "trade off" == "tradeoff"; "k means" == "kmeans"
        terms |= {stem(a + b) for a, b in zip(toks, toks[1:]) if is_content_token(a) or len(a) == 1}
        doc_terms.append(terms)
        doc_raw.append(set(toks))
    df: dict[str, int] = {}
    for s in doc_terms:
        for term in s:
            df[term] = df.get(term, 0) + 1
    topic_terms, weights = [], []
    for tt in topic_texts:
        terms = list(dict.fromkeys(content_terms(tt)))
        topic_terms.append(terms)
        weights.append({x: math.log((1 + D) / (1 + df.get(x, 0))) + 1.0 for x in terms})

    lex = np.zeros((T, D), np.float32)
    for i, terms in enumerate(topic_terms):
        if not terms:
            continue
        total = sum(weights[i][x] for x in terms)
        for j, s in enumerate(doc_terms):
            lex[i, j] = sum(weights[i][x] for x in terms if x in s) / total
    # acronym evidence: "support vector machines" is evidenced by "SVM" in the text
    hits: set = set()
    for i, tt in enumerate(topic_texts):
        toks = [t for t in tokenize(tt) if is_content_token(t)]
        if 2 <= len(toks) <= 5:
            acr = "".join(t[0] for t in toks)
            if len(acr) >= 3:
                for j, raw in enumerate(doc_raw):
                    if acr in raw or acr + "s" in raw:
                        hits.add((i, j))
                        lex[i, j] = max(lex[i, j], 0.95)
    combined = np.maximum(0.6 * sem_norm + 0.4 * lex, 0.9 * sem_norm)
    for (i, j) in hits:  # the acronym itself is strong, explicit evidence
        combined[i, j] = max(combined[i, j], 0.75)
    return MatchMatrices(sem, sem_norm, lex, combined, topic_terms, doc_terms, weights, hits)
