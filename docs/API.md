# API reference

Base URL `http://localhost:8000/api` · interactive docs at `/docs` · errors are always
`{"error": {"code": "...", "message": "..."}}` (validation errors use FastAPI's standard 422 body).

| Method & path | Purpose |
|---|---|
| `POST /documents/upload` | multipart: `file` (PDF), optional `subject_id` **or** `subject_name` (default "General"), `doc_type` = `lecture` \| `past_paper` \| `syllabus`. Returns **202** + document; processing continues in the background |
| `GET /documents[?subject_id=]` | list documents |
| `GET /documents/{id}` · `GET /documents/{id}/status` | details · processing status (`pending` → `processing` → `ready` \| `failed`, with `error`, `warnings`, `page_count`, `chunk_count`) |
| `DELETE /documents/{id}` | removes file, chunks and vectors (204) |
| `GET/POST /subjects` · `DELETE /subjects/{id}` | manage subjects |
| `POST /chat/ask` | grounded Q&A (below) |
| `POST /quiz/generate` | `{subject_id, topic, n_questions (1-20, default 5)}` → multiple-choice questions written by `exam_engine` from the subject's lecture notes, each with `source` document + page. Uses the local LLM (≈10–20 s per question on CPU), falls back to extractive questions; may return `questions: []` with a note |
| `POST /quiz/submit` | store a graded quiz, update per-topic performance |
| `GET /analysis/{subject_id}` · `GET /revision/{subject_id}` | provided by `exam_engine` (501 until installed) |
| `GET /health` | database, Ollama, model, embeddings (`available` is false when offline and the model isn't saved locally), OCR status; `status` is `ok` only if database, LLM and embeddings are all usable, else `degraded` |

Error codes: `unsupported_file_type`, `invalid_pdf`, `empty_file`, `file_too_large` (413), `duplicate_document` (409),
`document_not_found`, `subject_not_found` (404), `document_processing` (409), `llm_unavailable` (503),
`embedding_model_unavailable` (503: `OS_OFFLINE=1` but the embedding model was never saved to `data/models`),
`exam_engine_unavailable` (501).

## Examples

```bash
# Upload (then poll status until "ready")
curl -F "file=@Lecture 01.pdf" -F subject_name=Biology -F doc_type=lecture http://localhost:8000/api/documents/upload
curl http://localhost:8000/api/documents/1/status

# Ask
curl -X POST http://localhost:8000/api/chat/ask -H 'content-type: application/json' \
  -d '{"question": "Where does photosynthesis take place?", "subject_id": 1, "top_k": 4}'
```

```json
{
  "answer": "Photosynthesis takes place in the chloroplasts [1].",
  "grounded": true,
  "confidence": 0.61,
  "sources": [
    {"document_name": "Lecture 01.pdf", "page_number": 1,
     "excerpt": "Photosynthesis is the process by which green plants…", "score": 0.61, "document_id": 1}
  ]
}
```

When the material doesn't cover the question: `{"answer": "I couldn't find this in your uploaded materials…", "grounded": false, "sources": [], "confidence": 0.0}`
(the LLM is not even called when no passage passes the similarity threshold).

```bash
# Quiz result (client grades each question)
curl -X POST http://localhost:8000/api/quiz/submit -H 'content-type: application/json' \
  -d '{"subject_id": 1, "answers": [{"topic": "Mitosis", "correct": true}, {"topic": "Meiosis", "correct": false}]}'
```

## Frontend notes

CORS allows `localhost:3000/5173` by default (`OS_CORS_ORIGINS` to change). Upload is asynchronous: poll
`/documents/{id}/status` every ~1 s. `grounded` is true only if relevant passages were retrieved and the model
answered from them without abstaining; `sources` lists the passages the model cited (or all retrieved passages
if it cited none). Treat `confidence` as retrieval relevance, not a correctness probability.
