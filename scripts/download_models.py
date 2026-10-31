"""One-time setup (needs internet). After this, Offline Scholar runs fully offline.

    python scripts/download_models.py [--force]

1. Saves the Sentence Transformers embedding model into data/models/ (skipped if already there).
2. Pulls the Ollama LLM through Ollama's HTTP API (Ollama must be running; the `ollama` CLI does not
   need to be on PATH, which matters on Windows right after installing it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from backend.config import get_settings  # noqa: E402


def download_embedding_model(force: bool) -> None:
    s = get_settings()
    target = s.models_dir / s.embedding_model.split("/")[-1]
    if (target / "modules.json").is_file() and not force:
        print(f"[1/2] Embedding model already present: {target}")
        return
    print(f"[1/2] Downloading embedding model {s.embedding_model} -> {target}")
    from sentence_transformers import SentenceTransformer

    s.models_dir.mkdir(parents=True, exist_ok=True)
    SentenceTransformer(s.embedding_model, device="cpu").save(str(target))


def pull_ollama_model() -> int:
    s = get_settings()
    print(f"[2/2] Pulling Ollama model {s.ollama_model} from {s.ollama_url}")
    last = ""
    try:
        with httpx.stream("POST", f"{s.ollama_url.rstrip('/')}/api/pull",
                          json={"model": s.ollama_model}, timeout=httpx.Timeout(30.0, read=None)) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    print(f"  Ollama error: {msg['error']}")
                    return 1
                status = msg.get("status", "")
                if msg.get("total"):
                    status += f" {100 * msg.get('completed', 0) // msg['total']}%"
                if status != last:
                    print(f"  {status}", end="\r" if "%" in status else "\n", flush=True)
                    last = status
    except httpx.ConnectError:
        print(f"  Cannot reach Ollama at {s.ollama_url}. Install it from https://ollama.com and make sure "
              "it is running (Windows/macOS: start the Ollama app; Linux: `ollama serve`).")
        return 1
    print()
    return 0 if last == "success" else 1


def main() -> int:
    download_embedding_model(force="--force" in sys.argv)
    rc = pull_ollama_model()
    if rc == 0:
        print("\nDone. Optionally set OS_OFFLINE=1 in .env to forbid any network use from now on.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
