"""End-to-end and unit tests for exam_engine. Run:  pytest -q   (from offline_scholar/)"""
import json
import re

import numpy as np
import pytest

from exam_engine import (ExamEngine, analyze_syllabus_coverage, analyze_weak_topics, calculate_topic_frequency,
                         extract_questions, extract_questions_from_chunks, generate_quiz, generate_revision_plan,
                         group_similar_questions, grade_quiz, parse_syllabus_text)
from tests.exam_fixtures import LECTURES, PAPERS, SYLLABUS_TOPICS


# ---------------------------------------------------------------- a stand-in "dense" embedder
class ConceptEmbedder:
    """Deterministic fake of a neural embedder: synonyms share a concept dimension.
    Lets us test the semantic code path offline (the real MiniLM model is not bundled)."""

    kind = "dense"
    CONCEPTS = [
        {"cpu", "processor", "central", "processing"},
        {"evaluation", "metric", "precision", "recall", "accuracy", "confusion", "matrix"},
        {"car", "automobile", "vehicle"},
        {"network", "networks", "neural"},
    ]

    def prepare(self, texts):  # nothing to fit
        pass

    def encode(self, texts):
        out = np.zeros((len(texts), len(self.CONCEPTS) + 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                hit = False
                for c, words in enumerate(self.CONCEPTS):
                    if w in words:
                        out[i, c] += 1
                        hit = True
                if not hit and len(w) > 3:
                    out[i, len(self.CONCEPTS) + hash(w) % 64] += 0.3
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1
        return out / n


# ================================================================ Module 1: extraction
def test_extracts_numbered_questions_marks_sections_and_subparts():
    text = """UNIVERSITY EXAM 2023
Time Allowed: 3 hours   Maximum Marks: 60
Instructions: attempt all questions
Section A (2 marks each)
1. Define the OSI model.
2) What is a deadlock in an operating system?
Section B
Q3. Explain the working of a neural network. [10 marks]
Q4. (a) Differentiate between TCP and UDP. (5)
    (b) Explain the three-way handshake used by TCP. (5)
"""
    qs = extract_questions(text, document_id="d", document_name="Paper 2023.pdf", page_number=1)
    by = {q["number"]: q for q in qs}
    assert set(by) == {"1", "2", "3", "4a", "4b"}
    assert by["1"]["marks"] == 2 and by["1"]["section"] == "Section A"      # per-section default marks
    assert by["3"]["marks"] == 10 and "[" not in by["3"]["text"]             # explicit marks stripped from text
    assert by["4a"]["question_type"] == "compare"
    assert all(q["year"] == 2023 and q["document_id"] == "d" and q["page_number"] == 1 for q in qs)
    assert not any("Time Allowed" in q["text"] or "Instructions" in q["text"] for q in qs)


def test_instruction_filter_does_not_eat_real_questions():
    qs = extract_questions("1. Explain the instruction cycle of a CPU.\n2. Attempt all questions.", document_id="d")
    assert [q["text"] for q in qs] == ["Explain the instruction cycle of a CPU."]


def test_wrapped_lines_and_marks_on_next_line_and_two_word_commands():
    qs = extract_questions("1. Describe SQL joins with\n   examples.\n   (8)\n2. Define regularization.", document_id="d")
    assert qs[0]["text"] == "Describe SQL joins with examples." and qs[0]["marks"] == 8
    assert qs[1]["text"] == "Define regularization."


def test_page_spanning_and_page_numbers_from_chunks():
    qs = extract_questions_from_chunks(PAPERS)
    assert len(qs) >= 28
    q = next(q for q in qs if q["document_id"] == "paper_2022" and "K-means" in q["text"])
    assert q["page_number"] == 2 and q["year"] == 2022


def test_unnumbered_fallback_requires_question_like_sentences():
    qs = extract_questions("This paper is about ML. Explain overfitting in detail. Thanks for reading.", document_id="d")
    assert [q["text"] for q in qs] == ["Explain overfitting in detail."]
    assert qs[0]["confidence"] <= 0.6


# ================================================================ Module 1: similarity
def test_brief_example_is_grouped_even_though_wording_differs():
    qa = {"text": "Explain the working of a neural network.", "document_id": "p1", "year": 2022}
    qb = {"text": "Describe how artificial neural networks function.", "document_id": "p2", "year": 2023}
    qc = {"text": "What is a B-tree index?", "document_id": "p2", "year": 2023}
    res = group_similar_questions([qa, qb, qc])
    assert len(res["groups"]) == 1
    g = res["groups"][0]
    assert g["repeated"] and g["paper_count"] == 2 and g["match_type"] == "reworded"
    assert {"neural", "network"} <= set(g["shared_key_terms"])
    assert len(res["unique_questions"]) == 1


def test_same_paper_similarity_is_not_a_cross_exam_repeat():
    res = group_similar_questions([
        {"text": "What is a confusion matrix?", "document_id": "p1"},
        {"text": "What is a confusion matrix?", "document_id": "p1"}])
    assert res["groups"][0]["match_type"] == "exact" and res["groups"][0]["repeated"] is False


def test_dense_embedder_groups_paraphrases_with_no_shared_keywords():
    qs = [{"text": "What is a CPU?", "document_id": "a"}, {"text": "Define central processing unit.", "document_id": "b"},
          {"text": "Explain SQL joins.", "document_id": "b"}]
    assert group_similar_questions(qs)["groups"] == []                     # lexical fallback can't know
    res = group_similar_questions(qs, embedder=ConceptEmbedder())
    assert res["embedder"] == "dense" and len(res["groups"]) == 1 and res["groups"][0]["repeated"]


def test_average_linkage_does_not_chain_unrelated_questions():
    qs = [{"text": "Explain neural network training", "document_id": "a"},
          {"text": "Explain neural network training with backpropagation", "document_id": "b"},
          {"text": "Explain training with backpropagation of errors in compilers", "document_id": "c"}]
    res = group_similar_questions(qs, threshold=0.6)
    assert all(g["size"] <= 2 for g in res["groups"])


# ================================================================ Module 2: topics
def test_topic_frequency_counts_are_derived_from_the_data_not_hardcoded():
    qs = extract_questions_from_chunks(PAPERS)
    res = calculate_topic_frequency(qs)
    top = {t["topic"]: t for t in res["topics"]}
    nn = top["Neural Network"]
    assert nn["paper_count"] == 4 and nn["rank"] == 1 and nn["importance_label"] == "High"
    assert "4 of 4 papers" in nn["why"]
    assert res["topics"] == sorted(res["topics"], key=lambda t: -t["importance_score"])

    # a different subject proves nothing is tied to the fixture
    net = [{"text": "Explain the TCP three-way handshake.", "document_id": "n1", "year": 2023, "marks": 10},
           {"text": "Describe TCP congestion control.", "document_id": "n2", "year": 2024, "marks": 10},
           {"text": "Discuss TCP flow control and congestion control.", "document_id": "n3", "year": 2025, "marks": 10},
           {"text": "What is a subnet mask?", "document_id": "n3", "year": 2025, "marks": 3}]
    names = [t["topic"].lower() for t in calculate_topic_frequency(net)["topics"]]
    assert any("congestion" in n for n in names)


def test_single_paper_is_honest_about_what_it_cannot_know():
    qs = extract_questions(PAPERS[0]["text"], document_id="only", document_name="Only.pdf")
    res = calculate_topic_frequency(qs)
    assert res["total_papers"] == 1 and all(t["confidence"] == "low" for t in res["topics"])
    assert any("one paper" in n.lower() for n in res["notes"])


def test_syllabus_topics_map_questions_and_unmatched_are_reported_not_ranked():
    qs = extract_questions_from_chunks(PAPERS)
    res = calculate_topic_frequency(qs, topics=SYLLABUS_TOPICS)
    names = [t["topic"] for t in res["topics"]]
    assert "Neural networks" in names and "Not matched" not in " ".join(names)
    assert res["unmatched_questions"]                       # e.g. confusion matrix: not on the list
    rl = next(t for t in res["topics"] if t["topic"] == "Reinforcement learning")
    assert rl["question_count"] == 0                        # no false match from the word "learning"


# ================================================================ Module 3: syllabus coverage
def test_coverage_statuses_pages_and_explanations():
    res = analyze_syllabus_coverage(SYLLABUS_TOPICS, LECTURES)
    row = {t["topic"]: t for t in res["topics"]}
    nn = row["Neural networks"]
    assert nn["status"] == "Covered" and nn["document"].startswith("Lecture 3") and set(nn["pages"]) <= {4, 5} and nn["pages"]
    assert row["Reinforcement learning"]["status"] == "Not covered" and row["Reinforcement learning"]["pages"] == []
    assert "No lecture content" in row["Reinforcement learning"]["explanation"]
    # SVM only appears as the acronym "SVM" in the notes: must not be called missing
    assert row["Support vector machines"]["status"] in ("Covered", "Partially covered")
    # "trade-off" vs "tradeoff"
    assert row["Bias-variance tradeoff"]["status"] == "Covered"
    s = res["summary"]
    assert s["covered"] + s["partially_covered"] + s["not_covered"] == len(SYLLABUS_TOPICS)


def test_dense_semantic_match_counts_without_any_shared_keyword():
    lecture = [{"document_id": "l", "document_name": "Arch.pdf", "page_number": 7,
                "text": "The processor fetches, decodes and executes instructions one after another."},
               {"document_id": "l", "document_name": "Arch.pdf", "page_number": 9, "text": "Paging splits memory into frames."}]
    res = analyze_syllabus_coverage(["Central processing unit"], lecture, embedder=ConceptEmbedder())
    r = res["topics"][0]
    assert r["status"] != "Not covered" and r["pages"] == [7] and r["term_coverage"] < 0.5
    lexical = analyze_syllabus_coverage(["Central processing unit"], lecture)["topics"][0]
    assert lexical["status"] == "Not covered"              # documents the fallback's known limitation


def test_subtopics_give_partial_credit_and_name_the_missing_part():
    res = analyze_syllabus_coverage([{"name": "Evaluation metrics", "subtopics": ["confusion matrix", "precision and recall", "ROC curves"]}], LECTURES)
    r = res["topics"][0]
    assert r["status"] == "Partially covered" and r["pages"] == [1] and "ROC curves" in r["explanation"]
    assert [s["status"] for s in r["subtopics"]][-1] == "Not covered"


def test_parse_syllabus_text_and_empty_inputs():
    topics = parse_syllabus_text("Unit 1: Intro to AI\n- Search: BFS, DFS; A*\n3. Learning\n")
    assert topics[0]["name"] == "Intro to AI" or topics[0]["name"].startswith("Intro")
    assert any(t["subtopics"] for t in topics)
    assert analyze_syllabus_coverage([], LECTURES)["summary"]["total_topics"] == 0
    assert analyze_syllabus_coverage(["X topic"], [])["topics"][0]["status"] == "Not covered"


# ================================================================ Module 4: quizzes
def _check_grounded(quiz, chunks):
    corpus = " ".join(c["text"] for c in chunks)
    for q in quiz["questions"]:
        assert len(q["options"]) == 4 and len({o["text"] for o in q["options"]}) == 4
        assert q["correct_option"] in "ABCD" and q["options"]["ABCD".index(q["correct_option"])]["text"] == q["correct_answer"]
        assert q["explanation"] and q["topic"] and q["source"]["page_number"] is not None
        assert q["source"]["evidence"] in corpus                       # evidence is a real sentence of the notes
        assert q["correct_answer"].lower() in q["source"]["evidence"].lower() or q["generation_method"] == "llm"


def test_extractive_quiz_is_grounded_and_may_be_short():
    quiz = generate_quiz(LECTURES, "Neural networks", n_questions=3)
    assert 1 <= len(quiz["questions"]) <= 3
    _check_grounded(quiz, LECTURES)
    thin = generate_quiz("Cats sleep a lot.", "Cats", 5)
    assert thin["questions"] == [] and thin["generation_report"]["notes"]


def test_llm_output_is_validated_against_the_context():
    good_sentence = "The learning rate controls the size of each update; too large a learning rate makes training diverge."

    def fake_llm(prompt):
        assert "SOURCE" in prompt
        return json.dumps([
            {"question": "What happens if the learning rate is too large?", "options": ["Training diverges", "Training is instant", "Weights vanish", "Loss is zero"],
             "correct_index": 0, "explanation": "A too large learning rate makes training diverge.", "evidence": good_sentence, "difficulty": "basic"},
            {"question": "Who invented backpropagation?", "options": ["A", "B", "C", "D"], "correct_index": 1,
             "explanation": "Made up.", "evidence": "Backpropagation was invented by a famous committee in 1901."},
            {"question": "Broken", "options": ["only", "two"], "correct_index": 0, "explanation": "x", "evidence": good_sentence}])

    quiz = generate_quiz(LECTURES, "Gradient descent", 3, llm=fake_llm)
    llm_qs = [q for q in quiz["questions"] if q["generation_method"] == "llm"]
    assert len(llm_qs) == 1 and llm_qs[0]["source"]["page_number"] == 5 and llm_qs[0]["difficulty"] == "basic"
    rep = quiz["generation_report"]
    assert rep["llm_rejected"] == 2 and any("unsupported" in r for r in rep["llm_rejection_reasons"])


def test_llm_failure_falls_back_to_extractive():
    def broken(_):
        raise TimeoutError
    quiz = generate_quiz(LECTURES, "Neural networks", 2, llm=broken)
    assert quiz["questions"] and any("Local LLM call failed" in n for n in quiz["generation_report"]["notes"])


def test_grade_quiz_and_unanswered_counts_as_wrong():
    quiz = generate_quiz(LECTURES, "Neural networks", 2)
    qs = quiz["questions"]
    res = grade_quiz(quiz, {qs[0]["question_id"]: qs[0]["correct_option"]})
    assert res["correct"] == 1 and res["incorrect"] == len(qs) - 1
    assert res["results"][0]["topic"] == "Neural networks"


# ================================================================ Module 5: weak topics
def _rec(topic, ok, diff=None, qid=None, **kw):
    return {"topic": topic, "is_correct": ok, "difficulty": diff, "question_id": qid, **kw}


def test_networking_example_recommends_fundamentals_first():
    recs = [_rec("Networking", False, "basic", "n1"), _rec("Networking", False, "basic", "n2"), _rec("Networking", False, "basic", "n1"),
            _rec("Networking", True, "advanced", "n3"), _rec("Databases", True, "basic", "d1"), _rec("Databases", True, "basic", "d2"),
            _rec("Databases", True, "intermediate", "d3")]
    res = analyze_weak_topics(recs)
    net = next(t for t in res["topics"] if t["topic"] == "Networking")
    assert net["status"] == "weak" and (net["correct"], net["incorrect"]) == (1, 3) and net["accuracy"] == 0.25
    assert "fundamentals before attempting advanced" in net["recommendation"]
    assert net["repeatedly_missed_questions"] == [{"question_id": "n1", "times_wrong": 2}]
    assert res["weak_topics"] == ["Networking"] and res["strong_topics"] == ["Databases"]
    assert res["topics"][0]["topic"] == "Networking"


def test_small_samples_are_not_called_weak():
    res = analyze_weak_topics([_rec("Graphs", False, qid="g1")])
    t = res["topics"][0]
    assert t["status"] == "not_enough_data" and res["weak_topics"] == []


def test_basics_fine_but_advanced_failing_gets_application_advice():
    recs = [_rec("Trees", True, "basic"), _rec("Trees", True, "basic"), _rec("Trees", False, "advanced"), _rec("Trees", False, "advanced"), _rec("Trees", False, "intermediate")]
    assert "application-style" in analyze_weak_topics(recs)["topics"][0]["recommendation"]


def test_prerequisites_are_ordered_first_and_graded_quizzes_are_accepted():
    quiz = generate_quiz(LECTURES, "Neural networks", 2)
    graded = grade_quiz(quiz, {})
    recs = [_rec("TCP congestion control", False)] * 3 + [_rec("TCP basics", False)] * 3
    res = analyze_weak_topics([graded, recs], prerequisites={"TCP congestion control": ["TCP basics"]})
    assert res["revision_order"].index("TCP basics") < res["revision_order"].index("TCP congestion control")
    assert any(t["topic"] == "Neural networks" for t in res["topics"])


def test_trend_detection_and_empty_input():
    recs = [_rec("A", False)] * 3 + [_rec("A", True)] * 3
    assert analyze_weak_topics(recs)["topics"][0]["trend"] == "improving"
    assert analyze_weak_topics([])["overall"]["accuracy"] is None


# ================================================================ Module 6: revision
def _analysis():
    eng = ExamEngine()
    pp = eng.analyze_past_papers(PAPERS, syllabus_topics=SYLLABUS_TOPICS)
    cov = eng.syllabus_coverage(SYLLABUS_TOPICS, LECTURES)
    recs = [_rec("Neural networks", False)] * 4 + [_rec("Neural networks", True)] + [_rec("Decision trees", True)] * 3
    return pp["topic_frequency"], cov, eng.weak_topics(recs)


def test_frequent_and_weak_topic_outranks_frequent_but_fine_and_rare_but_uncovered():
    freq, cov, weak = _analysis()
    plan = generate_revision_plan({"topic_frequency": freq, "syllabus_coverage": cov, "weak_topics": weak}, days=5, minutes_per_day=90)
    order = [t["topic"] for t in plan["priority_topics"]]
    assert order[0] == "Neural networks"
    top = plan["priority_topics"][0]
    assert top["signals"]["weakness"] > 0.5 and top["signals"]["exam_importance"] == 1.0 and top["reasons"] and top["actions"]
    by = {t["topic"]: t for t in plan["priority_topics"]}
    rl = by["Reinforcement learning"]
    assert rl["not_seen_in_past_papers"] and rl["priority_score"] < top["priority_score"]
    assert by["Decision trees"]["priority_score"] < by["Gradient descent"]["priority_score"] + 40   # sanity: finite & ordered


def test_schedule_respects_budget_and_progress_reduces_minutes():
    freq, cov, weak = _analysis()
    base = generate_revision_plan({"topic_frequency": freq, "syllabus_coverage": cov, "weak_topics": weak}, days=3, minutes_per_day=60)
    assert all(d["minutes"] <= 60 for d in base["schedule"]) and len(base["schedule"]) == 3
    assert sum(d["minutes"] for d in base["schedule"]) <= 180
    done = generate_revision_plan({"topic_frequency": freq, "syllabus_coverage": cov, "weak_topics": weak,
                                   "progress": {"Neural networks": {"minutes_done": 40}}}, days=3, minutes_per_day=60)
    a = next(t for t in base["priority_topics"] if t["topic"] == "Neural networks")["suggested_minutes"]
    b = next(t for t in done["priority_topics"] if t["topic"] == "Neural networks")["suggested_minutes"]
    assert b < a


def test_plan_with_missing_signals_says_so_and_never_crashes():
    freq, cov, weak = _analysis()
    only_cov = generate_revision_plan({"syllabus_coverage": cov})
    assert any("without" in n.lower() for n in only_cov["notes"]) and only_cov["priority_topics"]
    assert generate_revision_plan({})["priority_topics"] == []
    only_weak = generate_revision_plan({"weak_topics": weak})
    assert any("topic_frequency" in n for n in only_weak["notes"])


def test_alignment_merges_differently_worded_names_with_dense_embedder():
    cov = {"topics": [{"topic": "Central processing unit", "status": "Not covered", "document": None, "pages": [], "explanation": ""}]}
    weak = analyze_weak_topics([_rec("CPU", False)] * 3, topic_importance=None)
    plan = generate_revision_plan({"syllabus_coverage": cov, "weak_topics": weak}, embedder=ConceptEmbedder())
    assert len(plan["priority_topics"]) == 1 and plan["priority_topics"][0]["aliases"]


# ================================================================ facade / contracts
def test_every_engine_output_is_json_serialisable():
    eng = ExamEngine()
    pp = eng.analyze_past_papers(PAPERS, syllabus_topics=SYLLABUS_TOPICS)
    cov = eng.syllabus_coverage(SYLLABUS_TOPICS, LECTURES)
    quiz = eng.make_quiz(LECTURES, "Neural networks", 2)
    graded = eng.grade(quiz, {})
    weak = eng.weak_topics([graded] * 2)
    plan = eng.revision_plan({"topic_frequency": pp["topic_frequency"], "syllabus_coverage": cov, "weak_topics": weak})
    for obj in (pp, cov, quiz, graded, weak, plan, eng.lecture_topics(LECTURES)):
        json.dumps(obj)
    assert {"questions", "similar_question_groups", "repeated_questions", "topic_frequency", "important_topics"} <= set(pp)
    assert any(g["repeated"] for g in pp["similar_question_groups"]["groups"])


def test_lecture_topics_report_documents_and_pages():
    topics = ExamEngine().lecture_topics(LECTURES, top_k=10)
    assert topics and all("documents" in t and t["documents"][0]["pages"] for t in topics)


# ---- regressions found on real PDFs (MIT OCW 6.867 exams, lectures and a web-printed syllabus) ----
def test_course_number_is_not_a_question_number():
    qs = extract_questions("6.867 Machine learning Final exam (Fall 2003)\n1. (3 points) Define overfitting.\n")
    assert [q["number"] for q in qs] == ["1"]


def test_ligatures_are_normalised_and_admin_lines_ignored():
    qs = extract_questions("1. Your name and MIT ID:\n2. (3 points) Explain why classiﬁers overﬁt noisy data.\n")
    assert len(qs) == 1 and "classifiers overfit" in qs[0]["text"]


def test_running_footers_are_removed_from_pages():
    from exam_engine.text_utils import strip_running_lines

    pages = [{"document_id": 1, "page_number": i, "text": f"{i}. Define precision for model number {'abcd'[i-1]}x.\nCite as: Some Course, Fall 2006. Page {i}"}
             for i in range(1, 5)]
    out = strip_running_lines(pages)
    assert all("Cite as" not in p["text"] and "Define precision" in p["text"] for p in out)
    two = strip_running_lines(pages[:2])                       # too few pages to call anything a footer
    assert all("Cite as" in p["text"] for p in two)


def test_web_printed_syllabus_keeps_only_topics():
    from exam_engine.bridge import syllabus_topics_from

    text = ("menu\nSyllabus\nGrading\nBishop. Neural Networks. ISBN: 9780198538646\nCalendar\nLEC #\nTOPICS\nKEY DATES\n"
            "Introduction, linear classification, perceptron update rule\nProblem set 1 out\n"
            "Boosting, margin, and complexity\nHMMs (cont.)\nTerms and Conditions\n© 2001–2026 MIT")
    names = [t["name"] for t in syllabus_topics_from([{"text": text}])]
    assert names == ["linear classification", "perceptron update rule", "Boosting", "margin", "complexity", "HMMs"]
