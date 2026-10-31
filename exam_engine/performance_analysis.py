"""MODULE 5 — Weak-topic identification from quiz attempts.

Public API
----------
analyze_weak_topics(quiz_results, min_attempts=3, weak_below=0.6, strong_from=0.8,
                    prerequisites=None, topic_importance=None) -> dict

Input (any of): a flat list of answer records, ``grade_quiz`` outputs, or a list of those::

    {"topic": "Networking", "is_correct": false, "question_id": "q1",   # required: topic, is_correct
     "difficulty": "basic|intermediate|advanced", "timestamp": "2026-10-30T10:00:00",  # optional
     "source": {"document_name": "...", "page_number": 4}, "question": "..."}     # optional

Rules (all visible in the output, nothing hidden):
* accuracy = correct / attempted per topic;
* ``weak`` needs at least ``min_attempts`` answers and accuracy < ``weak_below`` — with fewer
  answers the topic is ``not_enough_data`` rather than guessed at;
* the priority score uses a *smoothed* accuracy ((correct+1)/(attempted+2)), so one wrong answer
  does not look like a catastrophe, and rises with repeated misses of the same question,
  and with the topic's exam importance if ``topic_importance`` is supplied;
* recommendations look at *which* questions were failed: failing ``basic`` questions
  means "revise fundamentals first"; passing basics but failing harder ones means "practise
  application"; ``prerequisites`` ({"TCP congestion control": ["TCP basics"]}) put
  prerequisite topics ahead of the topics that depend on them.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterable

DIFFICULTIES = ("basic", "intermediate", "advanced")


def _flatten(quiz_results) -> list[dict]:
    recs: list[dict] = []

    def add(x):
        if isinstance(x, dict) and "results" in x and isinstance(x["results"], list):
            for r in x["results"]:
                add(r)
        elif isinstance(x, dict) and "topic" in x and "is_correct" in x:
            recs.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                add(y)

    add(quiz_results)
    return recs


def _ts(r, i):
    t = r.get("timestamp")
    if t is None:
        return (0, i)
    try:
        return (1, datetime.fromisoformat(str(t)).timestamp(), i) if isinstance(t, str) else (1, float(t), i)
    except ValueError:
        return (0, i)


def analyze_weak_topics(quiz_results, min_attempts: int = 3, weak_below: float = 0.6, strong_from: float = 0.8,
                        prerequisites: dict | None = None, topic_importance: dict | None = None) -> dict:
    recs = _flatten(quiz_results)
    recs = [r for _, r in sorted(((_ts(r, i), r) for i, r in enumerate(recs)), key=lambda x: x[0])]
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_topic[str(r["topic"])].append(r)
    importance = {k.lower(): v for k, v in (topic_importance or {}).items()}

    topics = []
    for name, rs in by_topic.items():
        attempted = len(rs)
        correct = sum(bool(r["is_correct"]) for r in rs)
        incorrect = attempted - correct
        acc = correct / attempted
        smoothed = (correct + 1) / (attempted + 2)

        by_diff = {}
        for d in DIFFICULTIES:
            sub = [r for r in rs if r.get("difficulty") == d]
            if sub:
                c = sum(bool(r["is_correct"]) for r in sub)
                by_diff[d] = {"attempted": len(sub), "correct": c, "accuracy": round(c / len(sub), 3)}

        wrong_counts: dict[str, int] = defaultdict(int)
        for r in rs:
            if not r["is_correct"] and r.get("question_id"):
                wrong_counts[r["question_id"]] += 1
        repeated_wrong = {q: n for q, n in wrong_counts.items() if n >= 2}

        trend = None
        if attempted >= 6:
            half = attempted // 2
            early = sum(bool(r["is_correct"]) for r in rs[:half]) / half
            late = sum(bool(r["is_correct"]) for r in rs[half:]) / (attempted - half)
            trend = "improving" if late - early >= 0.2 else "declining" if early - late >= 0.2 else "steady"

        if attempted < min_attempts:
            status = "not_enough_data"
        elif acc < weak_below:
            status = "weak"
        elif acc < strong_from:
            status = "developing"
        else:
            status = "strong"

        imp = importance.get(name.lower())
        weakness = 1 - smoothed
        evidence = min(1.0, attempted / 6)
        priority = 100 * weakness * (0.5 + 0.5 * evidence)
        priority += min(15, 5 * len(repeated_wrong))
        if imp is not None:
            priority += 20 * float(imp) * weakness
        if status == "strong":
            priority *= 0.4
        priority = round(min(priority, 100), 1)

        pointers, seen = [], set()
        for r in rs:
            if not r["is_correct"] and isinstance(r.get("source"), dict):
                key = (r["source"].get("document_name") or r["source"].get("document_id"), r["source"].get("page_number"))
                if key not in seen and key[0]:
                    seen.add(key)
                    pointers.append({"document": key[0], "page": key[1]})

        row = {
            "topic": name, "attempted": attempted, "correct": correct, "incorrect": incorrect,
            "accuracy": round(acc, 3), "smoothed_accuracy": round(smoothed, 3),
            "status": status, "trend": trend, "accuracy_by_difficulty": by_diff,
            "repeatedly_missed_questions": [{"question_id": q, "times_wrong": n} for q, n in sorted(repeated_wrong.items(), key=lambda x: -x[1])],
            "revision_priority": priority,
            "priority_label": "High" if priority >= 55 else "Medium" if priority >= 30 else "Low",
            "review_pointers": pointers[:3],
        }
        row["recommendation"] = _recommend(row)
        topics.append(row)

    topics.sort(key=lambda r: (-r["revision_priority"], r["topic"]))
    order = _order_with_prerequisites([t for t in topics if t["status"] in ("weak", "developing", "not_enough_data") and t["incorrect"] > 0],
                                       prerequisites or {})
    for t in topics:
        t["revise_after"] = [p for p in (prerequisites or {}).get(t["topic"], []) if p in order and order.index(p) < (order.index(t["topic"]) if t["topic"] in order else 0)]
    total = len(recs)
    corr = sum(bool(r["is_correct"]) for r in recs)
    return {
        "overall": {"attempted": total, "correct": corr, "incorrect": total - corr,
                    "accuracy": round(corr / total, 3) if total else None},
        "weak_topics": [t["topic"] for t in topics if t["status"] == "weak"],
        "strong_topics": [t["topic"] for t in topics if t["status"] == "strong"],
        "topics": topics,
        "revision_order": order,
        "settings": {"min_attempts": min_attempts, "weak_below": weak_below, "strong_from": strong_from},
    }


def _recommend(row: dict) -> str:
    n, c, name, st = row["attempted"], row["correct"], row["topic"], row["status"]
    if st == "not_enough_data":
        if row["incorrect"] == n:
            return f"You got all {n} {name} question(s) wrong so far — try a few more to confirm, and skim the basics of {name} first."
        return f"Only {n} answer(s) on {name} so far; take a longer quiz before judging it."
    if st == "strong":
        return f"{name} looks solid ({c}/{n} correct). A quick refresh before the exam is enough."
    bd = row["accuracy_by_difficulty"]
    basic = bd.get("basic")
    harder = [bd[d] for d in ("intermediate", "advanced") if d in bd]
    parts = []
    if basic and basic["attempted"] >= 2 and basic["accuracy"] < 0.6:
        parts.append(f"You are missing basic {name} questions ({basic['correct']}/{basic['attempted']} correct): revise {name} fundamentals before attempting advanced questions")
    elif basic and basic["accuracy"] >= 0.6 and harder and sum(h["correct"] for h in harder) / max(1, sum(h["attempted"] for h in harder)) < 0.6:
        parts.append(f"The basics of {name} look fine, but harder questions are going wrong: practise worked, application-style questions")
    else:
        parts.append(f"{c}/{n} correct on {name} ({round(100 * row['accuracy'])}%): revise this topic")
    if row["repeatedly_missed_questions"]:
        parts.append(f"{len(row['repeatedly_missed_questions'])} question(s) were missed more than once — go through their explanations")
    if row["review_pointers"]:
        ptr = row["review_pointers"][0]
        parts.append(f"start with {ptr['document']}" + (f", page {ptr['page']}" if ptr.get("page") else ""))
    if row["trend"] == "improving":
        parts.append("your recent answers are improving — keep going")
    elif row["trend"] == "declining":
        parts.append("recent answers are worse than earlier ones — take a break and revisit it fresh")
    return "; ".join(parts) + "."


def _order_with_prerequisites(rows: list[dict], prereq: dict) -> list[str]:
    """Priority order, but any prerequisite topic that is itself being revised comes first."""
    names = [r["topic"] for r in rows]
    out: list[str] = []

    def visit(n, stack=()):
        if n in out or n in stack:
            return
        for p in prereq.get(n, []):
            if p in names:
                visit(p, stack + (n,))
        out.append(n)

    for n in names:
        visit(n)
    return out
