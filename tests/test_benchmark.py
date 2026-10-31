"""Grading helpers of scripts/benchmark_models.py (the benchmark itself needs real models and PDFs)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("benchmark_models", ROOT / "scripts" / "benchmark_models.py")
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)


def test_facts_need_every_group_and_ignore_case():
    assert bm.facts_ok("Use the Avalanche method: pay the HIGHEST INTEREST debt first, snowball pays the smallest balance.",
                       ["highest interest", "smallest balance"])
    assert not bm.facts_ok("Pay the highest interest debt first.", ["highest interest", "smallest balance"])


def test_facts_match_text_with_pdf_ligatures():
    # PyMuPDF keeps typographic ligatures ("ﬁ" is one character); a reader sees "fi".
    assert bm.facts_ok("The classiﬁer is misclassiﬁed when …", ["misclassif"])


def test_source_hit_needs_matching_file_and_page():
    expected = [["ML_Lecture_04.pdf", 2], ["ML_Lecture_04.pdf", 3]]
    assert bm.source_hit([("ML_Lecture_01.pdf", 2), ("ML_Lecture_04.pdf", 3)], expected)
    assert not bm.source_hit([("ML_Lecture_04.pdf", 5), ("ML_Lecture_05.pdf", 2)], expected)


def test_summary_counts_refusals_and_hallucinations():
    rows = [
        {"expect": "answer", "error": None, "grounded": True, "facts_ok": True, "citation_ok": True, "seconds": 4.0, "llm_called": True},
        {"expect": "answer", "error": None, "grounded": False, "facts_ok": False, "citation_ok": False, "seconds": 3.0, "llm_called": True},
        {"expect": "refuse", "error": None, "grounded": False, "seconds": 0.1, "llm_called": False},
        {"expect": "refuse", "error": None, "grounded": True, "seconds": 6.0, "llm_called": True},
        {"expect": "answer", "error": "Ollama down", "seconds": 0.0, "llm_called": True},
    ]
    s = bm.summarise("m", rows)
    assert (s["answerable"], s["correct"], s["cited_right_page"], s["over_refused"]) == (2, 1, 1, 1)
    assert (s["unanswerable"], s["refused"], s["hallucinated"], s["errors"]) == (2, 1, 1, 1)
    assert s["median_s"] == 4.0  # only questions where the LLM was actually called


def test_question_file_is_consistent():
    spec = json.loads((ROOT / "scripts" / "benchmark_questions.json").read_text(encoding="utf-8"))
    subjects = {s["name"] for s in spec["subjects"]}
    ids = [q["id"] for q in spec["questions"]]
    assert len(ids) == len(set(ids))
    for q in spec["questions"]:
        assert q["subject"] in subjects and q["expect"] in {"answer", "refuse"}
        if q["expect"] == "answer":
            assert q["must_match"] and q["sources"]
