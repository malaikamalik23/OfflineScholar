"""MODULE 6 — Adaptive revision plan.

Public API
----------
generate_revision_plan(analysis_results, days=7, minutes_per_day=90, embedder=None) -> dict

``analysis_results`` combines whatever is available (every key is optional)::

    {
      "topic_frequency":   <calculate_topic_frequency() output>,   # what gets examined
      "syllabus_coverage": <analyze_syllabus_coverage() output>,   # what the notes lack
      "weak_topics":       <analyze_weak_topics() output>,         # how the student performs
      "progress":          {"Topic": {"minutes_done": 30}}         # revision already done
    }

Priority (0-100) per topic = weighted mix of the signals that EXIST for that topic:

    exam importance 0.35 · weakness 0.30 · coverage gap 0.20 · interaction 0.15

*interaction* = importance x max(weakness, gap): it makes "frequently examined AND
(weak OR poorly covered)" topics rise above topics that are only one of those.
Weights are renormalised over the available signals — a missing signal is never
treated as "all fine" or "all bad". Topic names from the three sources are aligned by
normalised wording first, then by embedding similarity.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np

from .embeddings import cosine_matrix, embed_all, get_embedder, thresholds_for
from .text_utils import content_terms

W_IMPORTANCE, W_WEAK, W_GAP, W_INTERACT = 0.35, 0.30, 0.20, 0.15
UNSEEN_IMPORTANCE = 0.15  # prior for a topic that matches nothing in the analysed past papers
COVERAGE_GAP = {"Covered": 0.0, "Partially covered": 0.5, "Not covered": 1.0}


def _key(name: str) -> str:
    return " ".join(sorted(set(content_terms(name))))


def _align(names: Sequence[str], embedder) -> dict[str, str]:
    """Map every raw topic name to a canonical name (first spelling seen)."""
    canon: "OrderedDict[str, str]" = OrderedDict()
    mapping: dict[str, str] = {}
    pending = []
    for n in names:
        k = _key(n)
        if k in canon:
            mapping[n] = canon[k]
        else:
            pending.append(n)
            canon[k] = n
            mapping[n] = n
    if len(pending) > 1 and embedder.kind == "dense":
        (vecs,) = embed_all(embedder, pending)
        sim = cosine_matrix(vecs)
        thr = thresholds_for(embedder)["question_group"] + 0.08
        reps: list[int] = []
        for i, n in enumerate(pending):
            hit = next((r for r in reps if sim[i, r] >= thr), None)
            if hit is None:
                reps.append(i)
            else:
                mapping[n] = pending[hit]
    else:
        # lexical: merge when one topic's key terms contain the other's (e.g. "Overfitting" ⊂ "Overfitting and regularization")
        reps = []
        for i, n in enumerate(pending):
            ti = set(content_terms(n))
            hit = None
            for r in reps:
                tr = set(content_terms(pending[r]))
                if ti and tr and (ti <= tr or tr <= ti) and min(len(ti), len(tr)) >= 1:
                    hit = r
                    break
            if hit is None:
                reps.append(i)
            else:
                mapping[n] = pending[hit]
    return mapping


def generate_revision_plan(analysis_results: dict, days: int = 7, minutes_per_day: int = 90, embedder=None) -> dict:
    freq = (analysis_results or {}).get("topic_frequency") or {}
    cov = (analysis_results or {}).get("syllabus_coverage") or {}
    weak = (analysis_results or {}).get("weak_topics") or {}
    progress = (analysis_results or {}).get("progress") or {}

    signals: dict[str, dict] = {}
    raw_names: list[str] = []

    def put(name, key, val, detail):
        raw_names.append(name)
        signals.setdefault(name, {})[key] = (val, detail)

    ftopics = [t for t in freq.get("topics", []) if not t.get("one_off")] or freq.get("topics", [])
    max_imp = max([t["importance_score"] for t in ftopics], default=0) or 1
    for t in ftopics:
        put(t["topic"], "importance", t["importance_score"] / max_imp if max_imp else 0,
            {"label": t["importance_label"], "why": t.get("why", ""), "papers": t.get("paper_count")})
    for t in cov.get("topics", []):
        put(t["topic"], "gap", COVERAGE_GAP[t["status"]],
            {"status": t["status"], "document": t.get("document"), "pages": t.get("pages"), "explanation": t.get("explanation", "")})
    for t in weak.get("topics", []):
        if t["status"] == "not_enough_data" and t["incorrect"] == 0:
            continue
        put(t["topic"], "weakness", 1 - t["smoothed_accuracy"],
            {"status": t["status"], "accuracy": t["accuracy"], "attempted": t["attempted"], "recommendation": t["recommendation"],
             "review_pointers": t.get("review_pointers", [])})

    emb = get_embedder(embedder, prefer_dense=embedder is not None)
    mapping = _align(list(dict.fromkeys(raw_names)), emb)
    merged: dict[str, dict] = {}
    for raw, sig in signals.items():
        c = mapping.get(raw, raw)
        m = merged.setdefault(c, {"aliases": set()})
        m["aliases"].add(raw)
        for k, v in sig.items():
            if k not in m or v[0] > m[k][0]:
                m[k] = v

    present = [k for k in ("importance", "weakness", "gap") if any(k in m for m in merged.values())]
    items = []
    for name, m in merged.items():
        imp = m.get("importance", (None, None))[0]
        imp_unseen = False
        if imp is None and "importance" in present:
            # past papers WERE analysed and this topic did not show up: low importance, not "unknown"
            imp, imp_unseen = UNSEEN_IMPORTANCE, True
        if not imp_unseen and "importance" in m and (m["importance"][1] or {}).get("papers") == 0:
            imp_unseen = True  # listed in the syllabus, but no analysed past-paper question maps to it
        wk = m.get("weakness", (None, None))[0]
        gap = m.get("gap", (None, None))[0]
        parts, weights = [], []
        for val, w in ((imp, W_IMPORTANCE), (wk, W_WEAK), (gap, W_GAP)):
            if val is not None:
                parts.append(val * w); weights.append(w)
        need = [x for x in (wk, gap) if x is not None]
        if imp is not None and need:
            parts.append(imp * max(need) * W_INTERACT); weights.append(W_INTERACT)
        score = 100 * sum(parts) / sum(weights) if weights else 0.0
        prog = progress.get(name) or next((progress[a] for a in m["aliases"] if a in progress), {})
        done = float(prog.get("minutes_done", 0) or 0)
        items.append({"topic": name, "aliases": sorted(m["aliases"] - {name}), "priority_score": round(score, 1),
                      "signals": {"exam_importance": None if imp is None else round(imp, 2),
                                  "weakness": None if wk is None else round(wk, 2),
                                  "coverage_gap": None if gap is None else round(gap, 2)},
                      "details": {k: v[1] for k, v in m.items() if k != "aliases"},
                      "not_seen_in_past_papers": imp_unseen,
                      "minutes_done": done})

    items.sort(key=lambda x: (-x["priority_score"], x["topic"]))
    total_budget = max(0, int(days) * int(minutes_per_day))
    top_scores = sum(max(i["priority_score"], 1) for i in items[: max(1, min(len(items), total_budget // 20))]) or 1
    remaining = total_budget
    for rank, it in enumerate(items, 1):
        it["rank"] = rank
        it["priority_label"] = "High" if it["priority_score"] >= 55 else "Medium" if it["priority_score"] >= 30 else "Low"
        want = max(20, int(round(total_budget * max(it["priority_score"], 1) / top_scores / 5) * 5))
        want = min(want, 120)
        left = max(0, want - int(it["minutes_done"]))
        it["suggested_minutes"] = min(left, remaining) if it["priority_label"] != "Low" or remaining > 0 else 0
        remaining -= it["suggested_minutes"]
        it["reasons"] = _reasons(it)
        it["actions"] = _actions(it)
    items = [i for i in items if i["suggested_minutes"] > 0 or i["priority_label"] != "Low"]

    schedule = _schedule(items, days, minutes_per_day)
    available = [k for k in ("topic_frequency", "syllabus_coverage", "weak_topics") if (analysis_results or {}).get(k)]
    missing = [k for k in ("topic_frequency", "syllabus_coverage", "weak_topics") if k not in available]
    notes = []
    if missing:
        notes.append("Plan built without: " + ", ".join(missing) + ". Priorities reflect only the signals available.")
    if "weakness" not in present:
        notes.append("No quiz data yet — take a quiz so weak topics can be targeted.")
    return {
        "priority_topics": items,
        "schedule": schedule,
        "summary": {"topics_in_plan": len(items), "days": days, "minutes_per_day": minutes_per_day,
                    "signals_used": available, "headline": _headline(items)},
        "notes": notes,
    }


def _reasons(it: dict) -> list[str]:
    r, d = [], it["details"]
    if it.get("not_seen_in_past_papers"):
        r.append("Not matched to any topic in the analysed past papers (low exam priority unless your syllabus says otherwise)")
    if "importance" in d and not it.get("not_seen_in_past_papers"):
        r.append(f"Frequently examined: {d['importance'].get('why', '').rstrip('.')}" if d["importance"].get("why") else "Frequently examined")
    if "gap" in d and d["gap"]["status"] != "Covered":
        r.append(f"Syllabus gap: {d['gap']['status'].lower()} in your notes")
    if "weakness" in d and d["weakness"]["status"] in ("weak", "developing", "not_enough_data"):
        r.append(f"Quiz performance: {d['weakness']['accuracy'] * 100:.0f}% over {d['weakness']['attempted']} answer(s)")
    return r


def _actions(it: dict) -> list[str]:
    d, acts = it["details"], []
    if "gap" in d and d["gap"]["status"] == "Not covered":
        acts.append("Find or write notes for this topic (textbook / slides) — your lecture notes do not cover it")
    elif "gap" in d and d["gap"]["status"] == "Partially covered":
        acts.append("Fill the missing parts of this topic: " + (d["gap"]["explanation"] or "review the textbook"))
    if "weakness" in d and d["weakness"]["status"] in ("weak", "developing", "not_enough_data"):
        acts.append(d["weakness"]["recommendation"])
    if "importance" in d and not acts and not it.get("not_seen_in_past_papers"):
        acts.append("Practise past-paper questions on this topic")
    elif "importance" in d and not it.get("not_seen_in_past_papers") and d["importance"].get("label") == "High":
        acts.append("Then solve the past-paper questions on this topic under exam conditions")
    return acts


def _schedule(items: list[dict], days: int, per_day: int) -> list[dict]:
    plan = [{"day": d + 1, "sessions": [], "minutes": 0} for d in range(max(0, int(days)))]
    if not plan:
        return plan
    queue = [(it["topic"], it["suggested_minutes"]) for it in items if it["suggested_minutes"] > 0]
    di = 0
    for topic, mins in queue:
        while mins > 0 and di < len(plan):
            room = per_day - plan[di]["minutes"]
            if room < 10:
                di += 1
                continue
            take = min(mins, room)
            plan[di]["sessions"].append({"topic": topic, "minutes": take})
            plan[di]["minutes"] += take
            mins -= take
    return plan


def _headline(items: list[dict]) -> str:
    top = [i["topic"] for i in items[:3] if i["priority_label"] != "Low"]
    if not top:
        return "No urgent revision topics found with the data available."
    return "Start with: " + ", ".join(top) + "."
