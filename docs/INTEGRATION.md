# Integration contract (Member 1 ⇄ Member 2)

## Reusable functions

```python
from ai_engine.interfaces import (
    extract_document, get_document_chunks, generate_embeddings, retrieve_context)
```

| Function | Returns |
|---|---|
| `extract_document(file_path)` | `ExtractedDocument(file_name, page_count, pages=[PageText(page_number, text, ocr)], warnings)`; raises `ExtractionError(code, message)` |
| `get_document_chunks(document_id)` | `list[ChunkRecord]` in reading order; raises `DocumentNotFoundError` |
| `generate_embeddings(texts)` | `numpy.float32 (n, dim)`, L2-normalised (cosine = dot product) |
| `retrieve_context(question, subject_id=None, top_k=4)` | `list[RetrievedChunk]` best first; `[]` if nothing relevant |

`ChunkRecord` fields: `chunk_id, document_id, document_name, subject_id, doc_type, page_number (1-based), chunk_index, text`.
`RetrievedChunk` adds `score` (cosine similarity). Both have `.to_dict()`.

These open their own DB sessions, so they can be called from anywhere (no FastAPI needed).
Past papers are uploaded with `doc_type="past_paper"` and syllabi with `doc_type="syllabus"`; filter on `ChunkRecord.doc_type`.

## Exam-engine hooks (called by the API)

`exam_engine/api.py` must expose:

```python
def analyze_subject(subject_id: int, db: Session) -> dict      # -> GET /api/analysis/{subject_id}
def recommend_revision(subject_id: int, db: Session) -> dict   # -> GET /api/revision/{subject_id}
```

Return JSON-serialisable dicts; the shape is yours — the API passes it through unchanged.
Inside, use `backend.database.session_scope()` only if you need a separate session.

## Shared SQLite schema (`backend/models.py`)

`subjects(id, name)` · `documents(id, subject_id, filename, doc_type, status, page_count, chunk_count, …)` ·
`chunks(id, document_id, page_number, chunk_index, text)` — `chunks.id` is also the FAISS vector id ·
`quiz_attempts(id, subject_id, total, correct, score)` · `quiz_answers(attempt_id, topic, question_ref, correct)` ·
`topic_performance(subject_id, topic, attempts, correct)` — running per-topic totals ·
`revision_recommendations(subject_id, topic, priority, reason)` — **write your recommendations here**.

Need a new column/table? Add it to `backend/models.py` in a PR and tell Member 1 (tables are created with `create_all`; there are no migrations yet, so delete `data/offline_scholar.db` after schema changes in development).
