# exam_engine — Offline Scholar, Member 2 (examination intelligence)

Analyses past papers, finds repeated questions and important topics, checks lecture notes against the
syllabus, makes grounded practice quizzes, finds weak topics and builds a revision plan.
**All local. No paid APIs. No frontend, no extra backend, no extra vector DB.**

```
exam_engine/
  question_extraction.py   Module 1  extract_questions(text)
  similarity.py            Module 1  group_similar_questions(questions)
  topic_analysis.py        Module 2  calculate_topic_frequency(questions)
  syllabus_coverage.py     Module 3  analyze_syllabus_coverage(syllabus_topics, lecture_chunks)
  quiz_generator.py        Module 4  generate_quiz(context, topic) · grade_quiz(quiz, answers)
  performance_analysis.py  Module 5  analyze_weak_topics(quiz_results)
  revision_engine.py       Module 6  generate_revision_plan(analysis_results)
  service.py               ExamEngine facade (what Member 1's backend calls)
  api_router.py            OPTIONAL APIRouter for Member 1 to mount (no app/DB of its own)
  embeddings.py, matching.py, text_utils.py   shared helpers
docs/EXAM_ENGINE_CONTRACTS.md     every input/output shape (for Members 1 and 3)
docs/exam_examples/*.json       real outputs from the synthetic test course
docs/EXAM_ENGINE_REFERENCE_REVIEW.md   what the five reference repos really contain + licenses
tests/                     33 tests (pytest -q)
```

## Quick start

```bash
pip install -r requirements.txt
pytest -q                          # 33 tests
python scripts/generate_exam_examples.py
```

```python
from exam_engine import ExamEngine
engine = ExamEngine(embedder=shared_embedding_service, llm=local_llm_callable)   # both optional

past = engine.analyze_past_papers(page_records, syllabus_topics=topics)  # questions, repeats, topic ranking
cov  = engine.syllabus_coverage(topics, lecture_page_records)
quiz = engine.make_quiz(retrieved_chunks, "Neural networks", n_questions=5)
res  = engine.grade(quiz, {"<question_id>": "B"})
weak = engine.weak_topics([res, ...])                                      # all stored attempts
plan = engine.revision_plan({"topic_frequency": past["topic_frequency"],
                             "syllabus_coverage": cov, "weak_topics": weak}, days=7, minutes_per_day=90)
```

The seven reusable functions required by the brief are importable directly from `exam_engine`.

## Integration with Member 1

* **Input** is the agreed `{document_id, document_name, page_number, text}` record. Nothing else is read.
* **Embeddings:** pass Member 1's shared service as `embedder` — any object with `encode(list[str])` (or `embed`), or a plain callable.
  If none is given the engine looks for a *locally cached* sentence-transformers model (never downloads), then falls back to TF-IDF.
* **LLM:** optional `llm(prompt) -> str` for quiz generation. Without it, quizzes are extractive.
* **Database:** the engine stores nothing. Member 1 persists results/attempts in SQLite and feeds attempts back to `weak_topics`.
  `api_router.build_router(engine, get_chunks=...)` lets Member 1 expose all endpoints with one `include_router` call
  (`get_chunks(document_ids)` is their SQLite lookup; endpoints also accept chunks inline).
* **Member 3** reads the JSON in `docs/EXAM_ENGINE_CONTRACTS.md`. Every ranked item carries a plain-language `why` / `explanation` / `recommendation`.

## How it decides (explainable by design)

* **Questions:** rule-based extraction (numbering styles, sub-parts, marks, sections, instruction filtering, wrapped lines), with a low-confidence fallback for unnumbered papers.
* **Repeated questions:** embedding cosine + *average-linkage* clustering (no chaining). "Repeated" means ≥ 2 **different** papers. Each group lists shared key terms.
* **Topics:** recurring content phrases (command words like *explain* and generic words like *types* are ignored); or your syllabus topics.
  Importance = recurrence across papers 45 % + exam weight 35 % + recency 20 %, always with a sentence saying why.
  With a single paper the engine says recurrence can't be measured and marks confidence low.
* **Syllabus coverage:** semantic similarity + idf-weighted term coverage + multi-page context + acronym matching (SVM ↔ support vector machine).
  A strong semantic match counts on its own; "Not covered" names the closest material.
* **Quizzes:** only from sentences that exist in the supplied context; LLM output must quote evidence found in that context or it is dropped.
* **Weak topics:** needs ≥ 3 answers before calling a topic weak; smoothed accuracy; reads *which difficulty* was failed to say "revise fundamentals first" vs "practise application".
* **Revision plan:** mixes exam importance, weakness, coverage gap and their interaction; missing signals are renormalised, never assumed.

## Known limitations (please read)

1. **The dense (neural) path was not run with a real model.** The build sandbox could not reach the model hub, so semantic behaviour was tested
   with a deterministic stand-in embedder (`tests/test_exam_engine.py::ConceptEmbedder`) that proves the plumbing, not the quality.
   Run the tests and `generate_examples.py` once on a machine with the real model and review the output before trusting the thresholds.
2. **TF-IDF fallback is lexical.** It handles stemming, "neural network" ↔ "artificial neural networks", acronyms and "trade-off" ↔ "tradeoff", but not true synonyms
   (e.g. "Evaluation metrics" vs a page on *confusion matrix* is reported Not covered unless the syllabus lists subtopics). `result["embedder"]` says which path ran.
3. **Thresholds are starting points** (`embeddings.THRESHOLDS`, `syllabus_coverage.COVERED_MIN/PARTIAL_MIN`). Calibrate on a handful of real papers/notes
   you have labelled; the dense values are typical for MiniLM-class models, not measured here.
4. **Extractive quizzes are conservative** — mostly definition and fill-in-the-blank, and short on thin material. Use the LLM path for richer questions; it is still validated.
5. Extraction is regex-based. Tables, two-column layouts and scanned/OCR noise depend on Member 1's PDF extraction quality.
6. All demo data is **synthetic** (`tests/fixtures.py`). No real exam statistics are included or implied.

## Calibration (10 minutes)

Take 20 question pairs you label "same" / "different" and 15 syllabus topics you label covered / partial / missing. Print `similarity` / `evidence_score`
for them and move `question_group`, `COVERED_MIN`, `PARTIAL_MIN` to the values that separate your labels.
