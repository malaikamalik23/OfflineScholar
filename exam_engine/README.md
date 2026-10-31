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
  api.py                   Backend hooks: analyze_subject · recommend_revision · generate_quiz_for_subject
  bridge.py                Adapters to the backend (DB readers, shared embedder/LLM); lazy imports
  embeddings.py, matching.py, text_utils.py   shared helpers
docs/EXAM_ENGINE_CONTRACTS.md     every input/output shape (for Members 1 and 3)
docs/exam_examples/*.json       real outputs from the synthetic test course
docs/EXAM_ENGINE_REFERENCE_REVIEW.md   what the five reference repos really contain + licenses
scripts/analyze_folder.py  run the engine on a folder of PDFs and print a report
scripts/compare_embedders.py  TF-IDF fallback vs the real embedding model
tests/                     engine unit tests + integration tests through the real upload API
```

## Quick start

```bash
pip install -r requirements.txt
pytest -q                          # whole suite (engine + backend)
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

## Backend integration (how Member 1's API uses the engine)

The backend already routes `GET /api/analysis/{subject_id}` and `GET /api/revision/{subject_id}` to
`exam_engine.api` (it imports it lazily and answers 501 if the package is missing). Nothing else has to be wired.

| Hook (`exam_engine/api.py`) | Called by | Returns |
|---|---|---|
| `analyze_subject(subject_id, db)` | `GET /api/analysis/{id}` | past-paper analysis, syllabus coverage, weak topics, `notes` |
| `recommend_revision(subject_id, db, days=7, minutes_per_day=90)` | `GET /api/revision/{id}` | prioritised topics, day-by-day schedule; also rewrites `revision_recommendations` rows |
| `generate_quiz_for_subject(subject_id, topic, db, n_questions=5)` | *(no route yet — add one)* | grounded MCQs with source document + page |

How the document roles are read (all from the existing tables, nothing new is stored):

* `Document.doc_type == "past_paper"` → questions. Past papers are **re-read from the stored PDF** (`Document.stored_path`)
  because stored chunks have no line breaks, so question numbers can't be told from numbers inside sentences.
  If the file is gone the engine falls back to the chunks and says so in `notes`.
* `doc_type == "syllabus"` → topic list. Web-printed syllabi are cleaned (only what follows a *Topics* / *Calendar* heading,
  no URLs/ISBNs/dates, comma lists split into topics).
* `doc_type == "lecture"` → compared with the syllabus **page by page** (also re-read from the PDF; running footers are stripped),
  and used as quiz material.
* `quiz_answers` joined to `quiz_attempts` → weak topics (`topic`, `correct`, `question_ref`, time).

Rules the hooks follow:

* **A missing input never raises.** No syllabus, no papers or no quiz results gives `null` for that section and a
  sentence in `notes` saying what to upload or do (so the frontend can show an empty state).
* `matching` in the response is `"dense"` (real sentence-transformers model) or `"lexical"` (TF-IDF fallback). The backend's
  test-only hashing embedder is replaced by TF-IDF.
* Quiz writing uses the local Ollama model if it answers; otherwise extractive questions. `question_ref` of a quiz answer should be the
  question's `question_id`, and `topic` the quiz topic, so weak-topic analysis lines up with the syllabus names.
* Results are recomputed on every call (only PDF text is cached). Fine for course-sized material; cache if it ever gets slow.

Other facts for Members 1 and 3:

* **Input** is the agreed `{document_id, document_name, page_number, text}` record.
* **Embeddings / LLM** are the shared services (`bridge.build_engine`); standalone, pass `embedder=` / `llm=` yourself.
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

1. **Tested on one real course so far** (MIT OCW 6.867: 4 exams, 6 lectures, a syllabus). With the real MiniLM model the coverage
   thresholds were tightened once (`PARTIAL_MIN` 0.35 → 0.55) because unrelated topics scored 0.38–0.53. Generic phrases
   such as "Current problems in machine learning" can still come out *Covered*. Check results on your own material.
2. **TF-IDF fallback is lexical.** It handles stemming, "neural network" ↔ "artificial neural networks", acronyms and "trade-off" ↔ "tradeoff",
   but not true synonyms. `result["embedder"]` / `matching` says which path ran.
3. **Thresholds are provisional** (`embeddings.THRESHOLDS`, `syllabus_coverage.COVERED_MIN/PARTIAL_MIN`).
4. **Extractive quizzes are modest.** They are definition and fill-in-the-blank items, filtered so fragments, formulas and sentences that depend
   on the previous one are skipped (fewer questions rather than bad ones). Some blanks still have two reasonable answers.
   The Ollama path gives more natural questions and is validated against the notes; it has **not been tested on real lectures yet**.
5. Extraction is regex-based. Tables, two-column layouts and scanned pages (OCR) depend on Member 1's PDF extraction.
   Repeated-question detection was only checked by eye on the real exams.
6. The synthetic course in `tests/exam_fixtures.py` is demo data. No real exam statistics are included or implied.

## Calibration (10 minutes)

Run `python scripts/analyze_folder.py <folder of PDFs>` on a course you know (file names decide roles: *Midterm/Final/Paper/Exam* =
past papers, *Syllabus* = syllabus, anything else = lecture notes), and `python scripts/compare_embedders.py` to see the TF-IDF fallback next to
the real model. Label 20 question pairs "same" / "different" and 15 syllabus topics covered / partial / missing, then move `question_group`,
`COVERED_MIN`, `PARTIAL_MIN` to the values that separate your labels.
