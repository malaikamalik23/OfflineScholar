"""One-time setup (needs internet). After this, Offline Scholar runs fully offline.

    python scripts/download_models.py

1. Saves the Sentence Transformers embedding model into data/models/.
2. Pulls the Ollama LLM (requires the `ollama` CLI and a running `ollama serve`).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings  # noqa: E402


def main() -> int:
    s = get_settings()
    target = s.models_dir / s.embedding_model.split("/")[-1]
    print(f"[1/2] Downloading embedding model {s.embedding_model} -> {target}")
    from sentence_transformers import SentenceTransformer

    s.models_dir.mkdir(parents=True, exist_ok=True)
    SentenceTransformer(s.embedding_model, device="cpu").save(str(target))

    print(f"[2/2] Pulling Ollama model {s.ollama_model}")
    if shutil.which("ollama") is None:
        print("  `ollama` not found. Install from https://ollama.com, then run:  ollama pull " + s.ollama_model)
        return 1
    rc = subprocess.call(["ollama", "pull", s.ollama_model])
    if rc == 0:
        print("\nDone. Optionally set OS_OFFLINE=1 in .env to forbid any network use from now on.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
