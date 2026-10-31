"""MODULE 2 — Topic extraction, question→topic mapping, frequency and ranking.

Public API
----------
calculate_topic_frequency(questions, topics=None, embedder=None, min_recurrence=2) -> dict
extract_keywords(texts, top_k=10) -> list[dict]
extract_lecture_topics(chunks, top_k=15) -> list[dict]
map_questions_to_topics(questions, topics, embedder=None) -> dict
get_important_topics(frequency_result, top_k=10) -> list[dict]

How topics are found (no unexplained numbers)
---------------------------------------------
Topics come from phrases that recur across questions ("neural network", "three-way
handshake"). Phrases are built from content words only (no stopwords, no exam command
words like *explain*/*describe*, no generic words like *types*/*advantages*). A longer
phrase absorbs the shorter phrases it fully contains. Questions without any recurring
phrase become one-off topics so nothing silently disappears. If ``topics`` (e.g. the
syllabus topics) are supplied, questions are mapped to those instead, using semantic +
term evidence (see matching.py).

Importance score (0-100) and why
--------------------------------
    45 %  how many *different papers* the topic appeared in,
    35 %  how much exam weight it carries (marks if known for most questions, else count),
    20 %  recency (latest year it appeared) — only when years are known.
With a single paper, recurrence cannot be measured: only the weight component is
used and ``confidence`` is "low". Every topic carries a plain-language ``why``.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Sequence

from .embeddings import get_embedder, thresholds_for
from .matching import score_matches
from .similarity import _as_records, _paper_key, _paper_label
from .text_utils import candidate_phrases, content_terms, pretty_topic, split_sentences

COVERAGE_ASSIGN_MIN = 0.35  # combined evidence needed to map a question to a supplied topic


# --------------------------------------------------------------------------- keywords
def extract_keywords(texts: Sequence[str], top_k: int = 10) -> list[dict]:
    """Key phrases of a text collection: frequent, specific, multi-word preferred."""
    n = max(len(texts), 1)
    df: Counter = Counter()
    tf: Counter = Counter()
    surf: dict = defaultdict(Counter)
    for t in texts:
        seen = set()
        for stems, s in candidate_phrases(t):
            tf[stems] += 1
            surf[stems][s] += 1
            seen.add(stems)
        df.update(seen)
    scored = []
    for stems, d in df.items():
        spec = math.log((1 + n) / (1 + d)) + 1.0 if n > 3 else 1.0
        score = tf[stems] * (1 + 0.5 * (len(stems) - 1)) * (0.75 if len(stems) == 1 else 1.0) * spec
        scored.append((score, stems))
    scored.sort(key=lambda x: (-x[0], -len(x[1])))
    out, chosen = [], []
    for score, stems in scored:
        if any(_contains(c, stems) for c in chosen):
            continue
        chosen.append(stems)
        out.append({"keyword": surf[stems].most_common(1)[0][0], "mentions": int(tf[stems]),
                    "texts_containing": int(df[stems])})
        if len(out) >= top_k:
            break
    return out


def _contains(big: tuple, small: tuple) -> bool:
    if len(small) > len(big):
        return False
    return any(big[i : i + len(small)] == small for i in range(len(big) - len(small) + 1))


# --------------------------------------------------------------------------- discovery
def _discover(records: list[dict], min_df: int, max_topics: int) -> list[dict]:
    """Find recurring topic phrases. Returns topic dicts with ``support`` (question
    indices that mention the topic — a question may support several topics)."""
    q_phr: list[dict] = []
    df: dict[tuple, set] = defaultdict(set)
    surf: dict[tuple, Counter] = defaultdict(Counter)
    for i, r in enumerate(records):
        d = {}
        for stems, s in candidate_phrases(r["text"]):
            d.setdefault(stems, s)
            surf[stems][s] += 1
        q_phr.append(d)
        for stems in d:
            df[stems].add(i)

    # unigrams that mostly occur *inside* longer phrases are modifiers of the course
    # ("learning" in "learning rate", "machine learning"...), not topics of their own.
    bigram_parents: dict[str, set] = defaultdict(set)
    for p in df:
        if len(p) == 2:
            for w in p:
                bigram_parents[w].add(p)
    standalone: Counter = Counter()
    for d in q_phr:
        for p in d:
            if len(p) == 1 and not any(p[0] in q and len(q) > 1 for q in d):
                standalone[p] += 1

    def score(stems):
        return len(df[stems]) * (1 + 0.35 * (len(stems) - 1)) * (0.7 if len(stems) == 1 else 1.0)

    def is_modifier(p):
        return (len(p) == 1 and len(bigram_parents[p[0]]) >= 3
                and standalone[p] < 0.5 * len(df[p]))

    cands = [p for p, idx in df.items() if len(idx) >= min_df and not is_modifier(p)]
    cands.sort(key=lambda p: (-score(p), -len(p), p))
    selected: list[dict] = []
    for p in cands:
        host = None
        for s in selected:
            contained = _contains(s["stems"], p)
            inside = len(df[p] & s["support"]) >= 0.8 * len(df[p])
            if (contained and len(df[p] & s["support"]) >= 0.5 * len(df[p])) or inside:
                host = s
                break
        if host is not None:
            if _contains(host["stems"], p):
                host["support"] |= df[p]
            elif len(p) == 1 and len(host["stems"]) == 1 and df[p] == host["support"]:
                first = min(host["support"])
                order = content_terms(records[first]["text"])
                pair = sorted([(host["stems"][0], host["name"]), (p[0], surf[p].most_common(1)[0][0])],
                              key=lambda x: order.index(x[0]) if x[0] in order else 99)
                host["name"] = f"{pair[0][1]} {pair[1][1]}"
                host["stems"] = (pair[0][0], pair[1][0])
                continue
            if not set(p) <= set(host["stems"]):
                host["absorbed"].append(surf[p].most_common(1)[0][0])
            continue
        if len(selected) < max_topics:
            selected.append({"stems": p, "name": surf[p].most_common(1)[0][0],
                             "support": set(df[p]), "absorbed": [], "one_off": False})

    # two unigrams with identical support are one concept ("bias" + "variance")
    merged: list[dict] = []
    for t in selected:
        twin = next((m for m in merged if len(t["stems"]) == 1 and len(m["stems"]) == 1
                     and m["support"] == t["support"]), None)
        if twin is None:
            merged.append(t)
        else:
            first = min(t["support"])
            order = [tok for tok in content_terms(records[first]["text"])]
            pair = sorted([twin, t], key=lambda x: order.index(x["stems"][0]) if x["stems"][0] in order else 99)
            twin["name"] = f"{pair[0]['name']} {pair[1]['name']}"
            twin["stems"] = pair[0]["stems"] + pair[1]["stems"]
    selected = merged

    covered = set().union(*[t["support"] for t in selected]) if selected else set()
    for i, d in enumerate(q_phr):
        if i in covered:
            continue
        best = sorted(d, key=lambda p: (-len(p), p))
        best = [p for p in best if len(p) >= 2] or best
        if not best:
            continue
        name = surf[best[0]].most_common(1)[0][0]
        twin = next((t for t in selected if t["name"] == name), None)
        if twin:
            twin["support"].add(i)
        else:
            selected.append({"stems": best[0], "name": name, "support": {i}, "absorbed": [], "one_off": True})
    selected.sort(key=lambda t: (-len(t["support"]), -len(t["stems"]), t["name"]))
    return selected


# --------------------------------------------------------------------------- mapping
def map_questions_to_topics(questions: Sequence, topics: Sequence, embedder=None) -> dict:
    """Map questions onto *given* topics (e.g. syllabus). Returns assignments with evidence."""
    recs = _as_records(questions)
    emb = get_embedder(embedder)
    names, texts = [], []
    for t in topics:
        if isinstance(t, str):
            names.append(t); texts.append(t)
        else:
            names.append(t.get("name") or t.get("topic"))
            extra = " ".join(str(x) for x in (t.get("subtopics") or [])) + " " + str(t.get("description") or "")
            texts.append(f"{names[-1]} {extra}".strip())
    mm = score_matches(texts, [r["text"] for r in recs], emb)
    assignments, unmapped = {}, []
    for j, r in enumerate(recs):
        if not names:
            break
        i = int(mm.combined[:, j].argmax())
        score = float(mm.combined[i, j])
        # one shared common word ("learning") must not be enough: need half the topic's
        # weighted key terms, or a strong semantic match
        strong = mm.lexical[i, j] >= 0.5 or mm.semantic_norm[i, j] >= 0.5
        if score >= COVERAGE_ASSIGN_MIN and strong:
            matched, _ = mm.terms_found(i, j)
            assignments[r["question_id"]] = {"topic": names[i], "evidence_score": round(score, 2),
                                             "matched_terms": matched}
        else:
            unmapped.append(r["question_id"])
    return {"assignments": assignments, "unmapped_question_ids": unmapped}


# --------------------------------------------------------------------------- frequency
def calculate_topic_frequency(questions: Sequence, topics: Sequence | None = None, embedder=None,
                              min_recurrence: int = 2, max_topics: int = 30) -> dict:
    """Topic frequency + explainable importance ranking. See module docstring."""
    recs = _as_records(questions)
    total_q = len(recs)
    papers = {_paper_key(r) for r in recs}
    P = max(len(papers), 1)
    result = {"total_questions": total_q, "total_papers": len(papers), "topic_source":
              "syllabus_topics" if topics else "discovered_from_questions", "topics": [],
              "notes": []}
    if not recs:
        return result

    absorbed_kw: dict[str, list[str]] = {}
    unmatched: list[dict] = []
    by_topic: dict[str, list[dict]] = defaultdict(list)
    one_off: set = set()
    if topics:
        mapping = map_questions_to_topics(recs, topics, embedder)
        for r in recs:
            a = mapping["assignments"].get(r["question_id"])
            if a:
                by_topic[a["topic"]].append(r)
            else:
                unmatched.append({"question_id": r["question_id"], "text": r["text"],
                                  "document_name": r.get("document_name"), "year": r.get("year")})
        for t in topics:  # listed topics that no question maps to still appear (importance 0)
            by_topic.setdefault(t if isinstance(t, str) else (t.get("name") or t.get("topic")), [])
    else:
        found = _discover(recs, min_recurrence, max_topics)
        for t in found:
            by_topic[t["name"]] = [recs[i] for i in sorted(t["support"])]
            absorbed_kw[t["name"]] = t["absorbed"]
            if t["one_off"]:
                one_off.add(t["name"])
        result["notes"].append("A question that mentions several topics counts toward each of them, so topic shares can add up to more than 100%.")

    marks_known = sum(1 for r in recs if r.get("marks") is not None)
    use_marks = total_q > 0 and marks_known / total_q >= 0.5
    total_marks = sum((r.get("marks") or 0) for r in recs)
    years_all = sorted({r["year"] for r in recs if r.get("year")})
    weights = {}
    for name, qs in by_topic.items():
        weights[name] = (sum((q.get("marks") or 0) for q in qs) / total_marks) if use_marks and total_marks else len(qs) / total_q
    max_w = max(weights.values()) or 1.0

    rows = []
    for name, qs in by_topic.items():
        paper_keys = {_paper_key(q) for q in qs}
        if not qs:
            pass
        years = sorted({q["year"] for q in qs if q.get("year")})
        pc = len(paper_keys)
        paper_share = pc / P
        rel_weight = weights[name] / max_w
        recency = None
        if len(years_all) >= 2 and years:
            recency = (max(years) - years_all[0]) / (years_all[-1] - years_all[0])
        if P == 1:
            score = 100 * rel_weight
            basis = "exam weight only (single paper)"
        elif recency is None:
            score = 100 * (0.5625 * paper_share + 0.4375 * rel_weight)
            basis = "recurrence across papers + exam weight"
        else:
            score = 100 * (0.45 * paper_share + 0.35 * rel_weight + 0.20 * recency)
            basis = "recurrence across papers + exam weight + recency"
        types = Counter(q.get("question_type", "other") for q in qs)
        topic_marks = sum((q.get("marks") or 0) for q in qs)
        kws = list(dict.fromkeys(absorbed_kw.get(name, []) + _cooccurring_keywords(qs, name)))[:6]
        rows.append({
            "topic": pretty_topic(name),
            "question_count": len(qs),
            "paper_count": pc,
            "papers": sorted({_paper_label(q) for q in qs}),
            "years": years,
            "total_marks": topic_marks if any(q.get("marks") is not None for q in qs) else None,
            "share_of_questions_pct": round(100 * len(qs) / total_q, 1),
            "share_of_marks_pct": round(100 * topic_marks / total_marks, 1) if use_marks and total_marks else None,
            "importance_score": round(score, 1),
            "keywords": kws,
            "question_types": dict(types),
            "example_questions": [q["text"] for q in sorted(qs, key=lambda q: -(q.get("year") or 0))[:3]],
            "question_ids": [q["question_id"] for q in qs],
            "recurring": pc >= 2 or (P == 1 and len(qs) >= 2),
            "one_off": name in one_off,
            "_basis": basis,
        })

    rows.sort(key=lambda r: (-r["importance_score"], -r["paper_count"], -r["question_count"], r["topic"]))
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
        r["importance_label"] = ("High" if r["importance_score"] >= 65 else
                                 "Medium" if r["importance_score"] >= 35 else "Low")
        r["confidence"] = "high" if P >= 4 else "medium" if P >= 2 else "low"
        r["why"] = _why(r, P, use_marks, total_q)
        r.pop("_basis")
    result["topics"] = rows
    result["scoring_basis"] = ("exam weight only (single paper — repetition across years cannot be measured)"
                               if P == 1 else
                               "recurrence across papers (45%) + exam weight (35%) + recency (20%)"
                               if len(years_all) >= 2 else
                               "recurrence across papers (56%) + exam weight (44%); no years found for recency")
    if P == 1:
        result["notes"].append("Only one paper analysed: topic importance reflects that paper only. Upload more past papers for reliable recurrence.")
    elif P < 4:
        result["notes"].append(f"Only {P} papers analysed; treat rankings as indicative.")
    if topics:
        result["unmatched_questions"] = unmatched
        if unmatched:
            result["notes"].append(f"{len(unmatched)} question(s) did not match any listed syllabus topic (see unmatched_questions) "
                                   "— the syllabus list may be missing a topic, or the wording differs too much for the current embedder.")
    return result


def _cooccurring_keywords(qs: list[dict], topic_name: str, k: int = 5) -> list[str]:
    own = set(content_terms(topic_name))
    c: Counter = Counter()
    surf: dict = {}
    for q in qs:
        for stems, s in candidate_phrases(q["text"], max_n=2):
            if len(stems) == 1 and stems[0] in own:
                continue
            if set(stems) <= own:
                continue
            c[stems] += 1
            surf.setdefault(stems, s)
    min_count = 2 if len(qs) >= 2 else 1
    ranked = sorted(((k, v) for k, v in c.items() if v >= min_count), key=lambda kv: (-kv[1], -len(kv[0])))
    out = []
    for stems, _ in ranked:
        if any(set(stems) & set(content_terms(o)) for o in out):
            continue
        out.append(surf[stems])
        if len(out) >= k:
            break
    return out


def _why(r: dict, P: int, use_marks: bool, total_q: int) -> str:
    parts = []
    if P > 1:
        yrs = f" ({', '.join(map(str, r['years']))})" if r["years"] else ""
        parts.append(f"Appeared in {r['paper_count']} of {P} papers{yrs}")
    else:
        parts.append("Appears in the uploaded paper")
    qs = f"{r['question_count']} question{'s' if r['question_count'] != 1 else ''}"
    if r["total_marks"] and use_marks and r["share_of_marks_pct"] is not None:
        qs += f" worth {r['total_marks']:g} marks ({r['share_of_marks_pct']}% of all marks)"
    parts.append(qs)
    if r["question_types"]:
        top = max(r["question_types"], key=r["question_types"].get)
        if top != "other":
            parts.append(f"usually asked as '{top.replace('_', ' ')}' questions")
    if r["one_off"]:
        parts.append("asked only once so far")
    return "; ".join(parts) + "."


def get_important_topics(frequency_result: dict, top_k: int = 10, include_one_off: bool = False) -> list[dict]:
    """Slim, frontend-ready list of the most important topics."""
    out = []
    for t in frequency_result.get("topics", []):
        if t["one_off"] and not include_one_off:
            continue
        out.append({k: t[k] for k in ("rank", "topic", "importance_label", "importance_score",
                                        "paper_count", "question_count", "years", "keywords", "why", "confidence")})
        if len(out) >= top_k:
            break
    return out


# --------------------------------------------------------------------------- lecture notes
def extract_lecture_topics(chunks: Sequence[dict], top_k: int = 15, min_chunks: int = 2) -> list[dict]:
    """Important topics in lecture notes: key phrases recurring across pages/chunks,
    reported with the documents and pages they appear on."""
    recs = [{"text": c["text"], "question_id": f"c{i}", "chunk": c} for i, c in enumerate(chunks) if (c.get("text") or "").strip()]
    if not recs:
        return []
    min_df = min(min_chunks, len(recs))
    info = _discover(recs, min_df=min_df, max_topics=max(top_k * 2, 20))
    names = [t["name"] for t in info if not t["one_off"]]
    stems_by_name = {n: tuple(content_terms(n)) for n in names}
    out = []
    for n in names:
        st = stems_by_name[n]
        pages = defaultdict(set)
        mentions = 0
        for r in recs:
            terms = content_terms(r["text"])
            hits = sum(1 for i in range(len(terms) - len(st) + 1) if tuple(terms[i:i + len(st)]) == st) if st else 0
            if hits:
                mentions += hits
                c = r["chunk"]
                pages[c.get("document_name") or c.get("document_id")].add(c.get("page_number"))
        if not pages:
            continue
        out.append({"topic": pretty_topic(n), "mentions": mentions,
                    "documents": [{"document": d, "pages": sorted(p for p in ps if p is not None)} for d, ps in pages.items()]})
    out.sort(key=lambda x: -x["mentions"])
    return out[:top_k]
