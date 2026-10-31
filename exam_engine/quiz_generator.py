"""MODULE 4 — Grounded multiple-choice quiz generation (+ grading).

Public API
----------
generate_quiz(context, topic, n_questions=5, llm=None, seed=0) -> dict
grade_quiz(quiz, answers) -> dict

Nothing unsupported is generated
--------------------------------
* **Extractive mode (default, no LLM needed).** Questions are built only from sentences
  that exist in the supplied context: definition sentences ("X is …") become
  *"Which statement describes X?"*, and key-term sentences become fill-in-the-blank
  questions. Distractors are real terms/definitions from the *same* material.
  If the material cannot support four distinct options, that question is skipped —
  the quiz is allowed to come back shorter than requested.
* **LLM mode (``llm=callable(prompt)->str``, e.g. Member 1's local model).** The model
  must return JSON including an ``evidence`` quote. Every question is validated:
  four distinct options, correct answer among them, explanation present, and the
  evidence quote must be found in the context. Failing questions are dropped and
  counted in ``generation_report``.

Question record::

    {"question_id","topic","question","options":[{"key":"A","text":...} x4],
     "correct_option":"B","correct_answer":"...","explanation":"...",
     "difficulty":"basic|intermediate|advanced",
     "source":{"document_id","document_name","page_number","evidence"},
     "generation_method":"extractive|llm"}
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from typing import Callable, Sequence

from .text_utils import (GENERIC_TERMS, candidate_phrases, content_terms, jaccard,
                         pretty_topic, split_sentences, tokenize)

_DEF_RE = re.compile(
    r"^(?:(?:A|An|The)\s+)?(?P<term>[A-Za-z][A-Za-z0-9\-/ ]{1,48}?)\s+"
    r"(?:is|are|refers to|is defined as|are defined as|means|can be defined as)\s+(?P<defn>.{20,220}?)[.]?$"
)
KEYS = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- helpers
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:24] or "topic"


def _qid(topic: str, text: str) -> str:
    return f"{_slug(topic)}-{hashlib.md5(text.encode()).hexdigest()[:8]}"


def _normalise_context(context) -> list[dict]:
    if isinstance(context, str):
        return [{"document_id": None, "document_name": None, "page_number": None, "text": context}]
    if isinstance(context, dict):
        return [context]
    return [c if isinstance(c, dict) else {"text": str(c)} for c in context]


def _sentences(chunks: Sequence[dict]) -> list[dict]:
    out = []
    for c in chunks:
        for s in split_sentences(c.get("text") or ""):
            n = len(s.split())
            if 6 <= n <= 45:
                out.append({"text": s, "chunk": c})
    return out


def _relevant(sents: list[dict], topic: str) -> list[dict]:
    t_terms = set(content_terms(topic))
    if not t_terms:
        return sents
    scored = [(len(t_terms & set(content_terms(s["text"]))) / len(t_terms), s) for s in sents]
    # keep sentences from chunks that mention the topic, even if the sentence itself doesn't
    chunk_hit = {id(s["chunk"]) for sc, s in scored if sc > 0}
    keep = [s for sc, s in scored if sc > 0 or id(s["chunk"]) in chunk_hit]
    return keep or sents


def _source(s: dict) -> dict:
    c = s["chunk"]
    return {"document_id": c.get("document_id"), "document_name": c.get("document_name"),
            "page_number": c.get("page_number"), "evidence": s["text"]}


def _finish(topic, question, correct, distractors, explanation, difficulty, src, method, rng) -> dict:
    opts = [correct] + list(distractors[:3])
    rng.shuffle(opts)
    return {
        "question_id": _qid(topic, question),
        "topic": topic,
        "question": question,
        "options": [{"key": k, "text": t} for k, t in zip(KEYS, opts)],
        "correct_option": KEYS[opts.index(correct)],
        "correct_answer": correct,
        "explanation": explanation,
        "difficulty": difficulty,
        "source": src,
        "generation_method": method,
    }


# --------------------------------------------------------------------------- extractive
def _definition_candidates(sents):
    out = []
    for s in sents:
        m = _DEF_RE.match(s["text"].strip())
        if not m:
            continue
        term = m.group("term").strip()
        if len(term.split()) > 5 or term.lower() in {"it", "this", "that", "there", "which"}:
            continue
        if not content_terms(term) or all(w in GENERIC_TERMS for w in tokenize(term)):
            continue
        out.append({"term": term, "defn": m.group("defn").strip().rstrip("."), "sent": s})
    return out


_NOUN_SUFFIXES = ("tion", "sion", "ity", "ment", "ness", "ence", "ance", "ism", "ology", "ization",
                  "ics", "ogram", "algorithm", "matrix", "vector", "entropy", "gradient", "kernel",
                  "variance", "bias", "margin", "neuron", "layer", "tree", "cluster", "node")


def _noun_like(word: str) -> bool:
    """Cheap stand-in for a POS tagger: single words that look like nouns."""
    w = word.lower()
    return len(w) >= 7 and w.endswith(_NOUN_SUFFIXES) or w.endswith(("tion", "sion", "ity", "ment"))


def _concept_pool(sents) -> dict:
    """Domain concepts = phrases that recur in >= 2 different sentences of the material.
    Returns {stems: (surface, sentence_count)}."""
    count: Counter = Counter()
    surf: dict = {}
    for s in sents:
        seen = set()
        for stems, sf in candidate_phrases(s["text"], max_n=2):
            if stems in seen:
                continue
            seen.add(stems)
            surf.setdefault(stems, sf)
            count[stems] += 1
    pool = {st: (surf[st], c) for st, c in count.items()
            if c >= 2 and len(surf[st]) >= 5 and (len(st) >= 2 or _noun_like(surf[st]))}
    # a unigram that only ever appears inside a recurring bigram is not a concept by itself
    for st in [p for p in pool if len(p) == 1]:
        if any(st[0] in big and len(big) == 2 and pool[big][1] >= pool[st][1] for big in pool):
            del pool[st]
    return pool


def _extractive(sents_topic, sents_all, topic, n, rng) -> list[dict]:
    qs: list[dict] = []
    used: set = set()

    defs = _definition_candidates(sents_topic)
    all_defs = _definition_candidates(sents_all)
    for d in defs:
        others = [o for o in all_defs if o["term"].lower() != d["term"].lower()
                  and jaccard(content_terms(o["defn"]), content_terms(d["defn"])) < 0.4]
        distractors = []
        for o in sorted(others, key=lambda o: abs(len(o["defn"]) - len(d["defn"]))):
            if o["defn"] not in distractors:
                distractors.append(o["defn"])
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            continue
        q = f"According to the course material, which statement best describes {d['term']}?"
        qs.append(_finish(topic, q, d["defn"][0].upper() + d["defn"][1:], [x[0].upper() + x[1:] for x in distractors],
                          f"The notes state: \"{d['sent']['text']}\"", "basic", _source(d["sent"]), "extractive", rng))
        used.add(d["sent"]["text"])

    pool = _concept_pool(sents_all)
    t_terms = set(content_terms(topic))
    for s in sents_topic:
        if s["text"] in used or len(pool) < 4:
            continue
        here = []
        for stems, _sf in candidate_phrases(s["text"], max_n=2):
            if stems in pool and not set(stems) <= t_terms:  # blanking the topic itself is trivial
                here.append(stems)
        if not here:
            continue
        here.sort(key=lambda st: (-len(st), pool[st][1], st))
        ans_stems = here[0]
        answer = pool[ans_stems][0]
        pat = re.compile(r"\b" + re.escape(answer) + r"\w*\b", re.IGNORECASE)
        m = pat.search(s["text"])
        if not m:
            continue
        shown = m.group(0)
        blanked = s["text"][: m.start()] + "_____" + s["text"][m.end():]
        if len([w for w in blanked.split() if "_____" not in w]) < 5:
            continue
        sent_terms = set(content_terms(s["text"]))
        cands = [pool[st][0] for st in pool if st != ans_stems and not (set(st) & sent_terms)
                 and not (set(st) & set(ans_stems))]
        cands.sort(key=lambda t: (abs(len(t) - len(answer)), t))
        distractors = list(dict.fromkeys(cands))[:3]
        if len(distractors) < 3:
            continue
        q = f"Fill in the blank using the course material: \"{blanked}\""
        qs.append(_finish(topic, q, shown, distractors, f"The notes state: \"{s['text']}\"",
                          "intermediate", _source(s), "extractive", rng))
        used.add(s["text"])

    # interleave: definitions first, then spread across pages
    seen_pages, picked, rest = set(), [], []
    for q in qs:
        pg = (q["source"]["document_id"], q["source"]["page_number"])
        (picked if pg not in seen_pages else rest).append(q)
        seen_pages.add(pg)
    ordered = picked + rest
    uniq, texts = [], set()
    for q in ordered:
        if q["question"] not in texts:
            uniq.append(q)
            texts.add(q["question"])
    return uniq[:n]


# --------------------------------------------------------------------------- LLM mode
def build_prompt(chunks: Sequence[dict], topic: str, n: int) -> str:
    ctx = "\n\n".join(
        f"[{c.get('document_name') or c.get('document_id') or 'source'} | page {c.get('page_number')}]\n{c.get('text', '')}"
        for c in chunks)
    return (
        "You write exam practice questions strictly from the SOURCE below. Do not use outside knowledge.\n"
        f"Topic: {topic}\nWrite {n} multiple-choice questions. Reply with ONLY a JSON list; each item:\n"
        '{"question": str, "options": [4 distinct strings], "correct_index": 0-3, "explanation": str, '
        '"evidence": "an exact sentence copied from the SOURCE that proves the answer", "difficulty": "basic|intermediate|advanced"}\n'
        "If the SOURCE cannot support a question, write fewer questions.\n\nSOURCE:\n" + ctx
    )


def _parse_json_list(raw: str) -> list:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        data = data.get("questions", [])
    return data if isinstance(data, list) else []


def _find_evidence(evidence: str, sents: list[dict]) -> dict | None:
    ev = set(content_terms(evidence))
    if len(ev) < 3:
        return None
    best, best_score = None, 0.0
    for s in sents:
        st = set(content_terms(s["text"]))
        score = len(ev & st) / len(ev)
        if score > best_score:
            best, best_score = s, score
    return best if best_score >= 0.8 else None


def validate_llm_items(items: list, sents_all: list[dict], topic: str, rng) -> tuple[list[dict], list[str]]:
    good, rejected = [], []
    for it in items:
        try:
            opts = [str(o).strip() for o in it["options"]]
            idx = int(it["correct_index"])
            q, expl, ev = str(it["question"]).strip(), str(it["explanation"]).strip(), str(it["evidence"]).strip()
        except (KeyError, TypeError, ValueError):
            rejected.append("malformed item")
            continue
        if len(opts) != 4 or len({o.lower() for o in opts}) != 4 or not 0 <= idx <= 3 or not all(opts):
            rejected.append("needs exactly 4 distinct options with a valid correct_index")
            continue
        if not q or not expl:
            rejected.append("missing question or explanation")
            continue
        src = _find_evidence(ev, sents_all)
        if src is None:
            rejected.append("evidence not found in the supplied material (unsupported)")
            continue
        correct = opts[idx]
        distract = [o for i, o in enumerate(opts) if i != idx]
        diff = it.get("difficulty") if it.get("difficulty") in ("basic", "intermediate", "advanced") else "intermediate"
        good.append(_finish(topic, q, correct, distract, expl, diff, _source(src), "llm", rng))
    return good, rejected


# --------------------------------------------------------------------------- public
def generate_quiz(context, topic: str, n_questions: int = 5, llm: Callable[[str], str] | None = None,
                  seed: int = 0) -> dict:
    """Generate grounded MCQs about ``topic`` from ``context`` (str, chunk, or list of chunks)."""
    chunks = _normalise_context(context)
    sents_all = _sentences(chunks)
    sents_topic = _relevant(sents_all, topic)
    rng = random.Random(f"{seed}:{topic}")
    report = {"requested": n_questions, "generated": 0, "from_llm": 0, "from_extractive": 0,
              "llm_rejected": 0, "llm_rejection_reasons": [], "notes": []}
    questions: list[dict] = []

    if llm is not None:
        try:
            items = _parse_json_list(llm(build_prompt(chunks, topic, n_questions)))
        except Exception as exc:  # local model unavailable / timed out
            items = []
            report["notes"].append(f"Local LLM call failed ({type(exc).__name__}); used extractive generation.")
        good, rejected = validate_llm_items(items, sents_all, topic, rng)
        questions.extend(good[:n_questions])
        report["from_llm"] = len(questions)
        report["llm_rejected"] = len(rejected)
        report["llm_rejection_reasons"] = sorted(set(rejected))

    if len(questions) < n_questions:
        have = {q["question"] for q in questions}
        extra = [q for q in _extractive(sents_topic, sents_all, topic, n_questions * 2, rng) if q["question"] not in have]
        questions.extend(extra[: n_questions - len(questions)])
        report["from_extractive"] = len(questions) - report["from_llm"]

    report["generated"] = len(questions)
    if len(questions) < n_questions:
        report["notes"].append(
            f"Only {len(questions)} well-supported question(s) could be generated from the supplied material; "
            "unsupported questions were deliberately not created.")
    if not sents_all:
        report["notes"].append("No usable sentences were found in the supplied context.")
    return {"topic": topic, "questions": questions, "generation_report": report}


def grade_quiz(quiz: dict, answers: dict) -> dict:
    """Grade one attempt. ``answers`` = {question_id: "A".."D"} (unanswered = incorrect)."""
    results = []
    for q in quiz.get("questions", []):
        sel = (answers.get(q["question_id"]) or "").strip().upper() or None
        results.append({
            "question_id": q["question_id"], "topic": q["topic"], "difficulty": q.get("difficulty"),
            "selected_option": sel, "correct_option": q["correct_option"],
            "is_correct": sel == q["correct_option"], "explanation": q["explanation"],
            "source": q["source"],
        })
    n = len(results)
    c = sum(r["is_correct"] for r in results)
    return {"topic": quiz.get("topic"), "total": n, "correct": c, "incorrect": n - c,
            "accuracy": round(c / n, 3) if n else None, "results": results}
