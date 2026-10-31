# Evaluation

Status on 2026-10-31: **retrieval measured on real documents with the real embedding model; the 3-model LLM
comparison has not been completed yet.** A run was started and stopped before any model finished, so this file
reports no answer-quality or speed numbers for `qwen2.5:3b`, `llama3.2:3b` or `phi3:mini`.

## How to run

```bash
python scripts/benchmark_models.py --retrieval-only                       # embeddings + FAISS only, no LLM
python scripts/benchmark_models.py --models qwen2.5:3b llama3.2:3b phi3:mini
```

The script ingests the PDFs into a throwaway folder (`data/benchmark/_run`, never the app's real data) through the
real pipeline, then asks every question in `scripts/benchmark_questions.json` through `ai_engine.rag.answer_question`,
the function behind `POST /api/chat/ask`. Results go to `data/benchmark/results/` (JSON + Markdown). Requires the
embedding model already downloaded and Ollama running; it never uses the network.

## Test material (not in git)

Put the PDFs in `data/benchmark/ml/` and `data/benchmark/other/`.

| Subject | Documents | Pages | Chunks |
|---|---|---|---|
| Machine Learning 6.867 | MIT OpenCourseWare 6.867 Fall 2006: lectures 1–6, syllabus, midterms 2002–2004, final 2003 | 83 | 198 |
| Short Courses | Four 25-page sample courses: Python, personal finance, digital marketing, project management | 100 | 183 |

All 15 PDFs have a text layer (no OCR needed). Ingestion took about 1 s per document on CPU, after a one-off
~10 s model load.

## Questions

31 questions: 22 the documents answer (12 ML, 10 short courses), each with the page(s) that contain the answer and the
facts a correct answer must state; 9 the documents do not answer. The unanswerable set includes traps: general
knowledge, a topic that only exists in the *other* subject, a topic the syllabus names but never explains (hidden
Markov models), and questions next to a covered topic (Python's walrus operator, Agile Manifesto year).

## Results: retrieval (`all-MiniLM-L6-v2`, top_k = 4, OS_MIN_SCORE = 0.25)

| Measure | Result |
|---|---|
| Right page in the top 4, answerable questions | **21 / 22 (95%)** |
| Nothing retrieved for unanswerable questions (LLM not called) | **4 / 9 (44%)** |

- The one miss: *What are support vectors?* retrieved Lecture 3 page 6 (which mentions support vectors) instead of the
  defining page 3.
- The similarity threshold only stops clearly unrelated questions (capital of Australia, World Cup, boiling point,
  support vector machines asked in the short-courses subject). The other 5 unanswerable questions retrieve passages
  scoring 0.33–0.57, for example the walrus operator against the Python modules page. For those, refusing depends
  entirely on the LLM following the `NOT_FOUND` instruction, which is exactly what the unfinished model comparison
  is meant to measure.
- Some correct answers score low (0.32 for the SEO title-length question), so raising `OS_MIN_SCORE` to block more
  unanswerable questions would also start dropping answerable ones.

## Observed, not yet measured

- PyMuPDF keeps typographic ligatures from the MIT PDFs (`classiﬁer` with a single `ﬁ` character). Whether this
  lowers retrieval scores was not measured. Normalising extracted text with Unicode NFKC is a cheap fix to test.

## Known limits

From the brief and code review, not measured here: a 3B model can misread numbers from OCR'd slides and the
answer-support check cannot catch that; the model sometimes over-refuses; no auth; no schema migrations;
English-only OCR by default.
