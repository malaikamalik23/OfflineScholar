# Offline Scholar

Offline-first study assistant: upload lecture notes, past papers and syllabi (PDF), ask questions, and get
answers **grounded in your own documents** with document names and page numbers. Everything runs locally —
no cloud or paid APIs, no internet needed after the one-time model download. MIT licensed.

```
frontend/      UI (separate)          exam_engine/  Member 2: analysis & revision
backend/       FastAPI, SQLite, routes   ai_engine/   extraction, chunking, embeddings, FAISS, Ollama RAG
data/          uploads, FAISS indexes, SQLite (git-ignored)      tests/  docs/  scripts/
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional OCR for scanned PDFs: install Tesseract (apt install tesseract-ocr / brew install tesseract)

# one-time, needs internet: embedding model + LLM (install Ollama first: https://ollama.com)
ollama serve &                      # skip if Ollama already runs as a service
python scripts/download_models.py   # MiniLM embeddings (~90 MB) + qwen2.5:3b (~2 GB)

cp .env.example .env                # set OS_OFFLINE=1 to forbid network use from now on
uvicorn backend.main:app --port 8000
```

Open http://localhost:8000/docs. Check http://localhost:8000/api/health — `status` should be `ok`.

## How it works

PDF → PyMuPDF text per page (Tesseract OCR for image-only pages) → sentence-aware chunks that never cross pages
→ MiniLM embeddings → FAISS (one index per subject; vector id = `chunks.id` in SQLite) → question embedding →
top-k passages (below `OS_MIN_SCORE` they are discarded) → prompt to the Ollama model that must answer only from
numbered passages and cite them, or say `NOT_FOUND`. If nothing relevant is retrieved the LLM isn't called.

## Configuration (`OS_*` env vars, see `.env.example`)

`OS_OLLAMA_MODEL` (default `qwen2.5:3b`; try `llama3.2:3b` or `phi3:mini`), `OS_EMBEDDING_MODEL`, `OS_MIN_SCORE`,
`OS_CHUNK_SIZE`, `OS_OCR_ENABLED`, `OS_MAX_UPLOAD_MB`, `OS_DATA_DIR`, `OS_OFFLINE`.

## Tests

```bash
pip install -r requirements-dev.txt && pytest
```

The suite uses a deterministic hashing embedder and a fake LLM so it runs without downloads; it exercises real
PDFs, OCR, chunking, FAISS, SQLite and every endpoint. Verifying answer *quality* needs the real models (see above).

## Docs

[API reference & examples](docs/API.md) · [Integration contract for exam_engine](docs/INTEGRATION.md)

## Limitations

PDF only; no auth (single-user, local); no schema migrations yet; answer quality depends on the small local model;
OCR handles English by default (`OS_OCR_LANGUAGE` + Tesseract language packs for others).
