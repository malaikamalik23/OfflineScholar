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
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# OCR for scanned PDFs and slide images: install Tesseract
#   Linux: apt install tesseract-ocr   macOS: brew install tesseract
#   Windows: winget install UB-Mannheim.TesseractOCR   (found automatically even though it is not on PATH)

# one-time, needs internet: install Ollama (https://ollama.com; Windows: winget install Ollama.Ollama),
# make sure it is running (Windows/macOS: the Ollama app; Linux: `ollama serve`), then:
python scripts/download_models.py   # MiniLM embeddings (~90 MB) into data/models + qwen2.5:3b (~1.9 GB)

cp .env.example .env                # set OS_OFFLINE=1 to forbid network use from now on
uvicorn backend.main:app --port 8000
```

Open http://127.0.0.1:8000/docs. Check http://127.0.0.1:8000/api/health — `status` should be `ok`.

## Desktop app

The UI in `frontend/` (React + Vite, from the Figma Make design) talks to the API above. To run it as a desktop app:

```bash
pip install -r requirements-desktop.txt        # adds pywebview (native window via Edge WebView2)
cd frontend && npm install && npm run build && cd ..
python scripts/download_models.py              # once, with internet (skip if already done)
```

Then double-click **`Offline Scholar.bat`** (Windows) or run `python scripts/desktop.py`. It starts the backend on
127.0.0.1 with `OS_OFFLINE=1` and opens the app in its own window (`--browser` opens your browser instead). Ollama
must be running. FastAPI also serves the built UI at http://localhost:8000 when you run uvicorn directly.

For UI development: run the backend on :8000, then `npm run dev` in `frontend/` (http://localhost:5173 proxies `/api`).

## How it works

PDF → PyMuPDF text per page (Tesseract OCR for image-only pages, and for slides whose text layer is just a title
while the content is a picture) → sentence-aware chunks that never cross pages → MiniLM embeddings → FAISS (one
index per subject; vector id = `chunks.id` in SQLite) → question embedding → top-k passages (below `OS_MIN_SCORE`
they are discarded) → prompt to the Ollama model that must answer only from numbered passages and cite them, or say
`NOT_FOUND`. If nothing relevant is retrieved the LLM isn't called; if the answer's words are mostly not in the
passages it cites (`OS_MIN_ANSWER_SUPPORT`), it is treated as not found instead of being shown as grounded.

Measured on a real lecture with the default models on a laptop CPU: see [docs/EVALUATION.md](docs/EVALUATION.md).

## Configuration (`OS_*` env vars, see `.env.example`)

`OS_OLLAMA_MODEL` (default `qwen2.5:3b`, see [docs/EVALUATION.md](docs/EVALUATION.md)), `OS_OLLAMA_URL` (default
`http://127.0.0.1:11434`; avoid `localhost` on Windows, it adds ~2 s per request), `OS_LLM_MAX_TOKENS` (answer length
cap, 400), `OS_EMBEDDING_MODEL`, `OS_MIN_SCORE` (0.25), `OS_MIN_ANSWER_SUPPORT` (0.55), `OS_CHUNK_SIZE`,
`OS_OCR_ENABLED`, `OS_TESSERACT_CMD` (only if Tesseract is installed somewhere unusual), `OS_MAX_UPLOAD_MB`,
`OS_DATA_DIR`, `OS_OFFLINE`.

## Checking that it really runs offline

After `scripts/download_models.py` has run once: set `OS_OFFLINE=1`, turn Wi-Fi off / unplug the network, start
the server, then upload a PDF and ask a question. `/api/health` should say `"status": "ok"` and
`"offline_mode": true`. If the embedding model was never downloaded, health shows `embeddings.available: false`
and asking returns `embedding_model_unavailable` instead of trying to reach the internet. (Ollama itself may check
for app updates in the background; that is the Ollama app, not Offline Scholar.)

## Tests

```bash
pip install -r requirements-dev.txt && pytest
```

The suite uses a deterministic hashing embedder and a fake LLM so it runs without downloads (that is also what
GitHub Actions runs); it exercises real PDFs, OCR, chunking, FAISS, SQLite and every endpoint. Answer *quality*
needs the real models: start the server and run `python scripts/evaluate.py --pdf <lecture.pdf> --questions
<questions.json>` (format in the script's docstring).

## Docs

[API reference & examples](docs/API.md) · [Integration contract for exam_engine](docs/INTEGRATION.md) ·
[Evaluation with real models](docs/EVALUATION.md)

## Limitations

PDF only; no auth (single-user, local); no schema migrations yet; answer quality depends on the small local model;
OCR handles English by default (`OS_OCR_LANGUAGE` + Tesseract language packs for others).
