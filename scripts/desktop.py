"""Offline Scholar desktop app: starts the local backend and opens the UI in its own window.

    python scripts/desktop.py              # or double-click "Offline Scholar.bat"
    python scripts/desktop.py --browser    # open in the default browser instead of a window

Needs, once: `pip install -r requirements-desktop.txt`, `python scripts/download_models.py`
(embedding model + Ollama model) and `npm run build` in frontend/. Ollama must be running.
Everything stays on this computer; offline mode is on by default.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # .env and the default data/ folder are relative to the repo
os.environ.setdefault("OS_OFFLINE", "1")  # no network after setup; set OS_OFFLINE=0 to allow downloads

import httpx  # noqa: E402
import uvicorn  # noqa: E402

TITLE = "Offline Scholar"


def free_port(preferred: int) -> int:
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def preflight() -> list[str]:
    """Problems worth telling the user about before the window opens. None of them stop the app."""
    problems = []
    if not (ROOT / "frontend" / "dist" / "index.html").is_file():
        problems.append("The UI isn't built yet: run `npm install` and `npm run build` in the frontend folder.")
    from backend.config import get_settings

    s = get_settings()
    if not (s.models_dir / s.embedding_model.split("/")[-1] / "modules.json").is_file():
        problems.append("The search model isn't downloaded: run `python scripts/download_models.py` once with internet.")
    try:
        httpx.get(f"{s.ollama_url.rstrip('/')}/api/tags", timeout=2, trust_env=False).raise_for_status()
    except Exception:  # noqa: BLE001
        problems.append("Ollama isn't running: start the Ollama app (answers and quizzes need it).")
    return problems


def start_server(port: int) -> uvicorn.Server:
    from backend.main import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 120  # the first start loads the embedding model
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=5, trust_env=False).status_code == 200:
                return server
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise SystemExit("The backend did not start within 2 minutes. Run `uvicorn backend.main:app` to see the error.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--browser", action="store_true", help="open in the default browser instead of a window")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    for p in preflight():
        print("!", p)
    port = free_port(args.port)
    print(f"Starting {TITLE} on http://127.0.0.1:{port} …")
    server = start_server(port)
    url = f"http://127.0.0.1:{port}/#dashboard"

    try:
        import webview  # pywebview: a native window using Edge WebView2 on Windows
    except ImportError:
        webview = None
    if args.browser or webview is None:
        if webview is None:
            print("pywebview isn't installed (pip install -r requirements-desktop.txt); opening your browser instead.")
        webbrowser.open(url)
        print("Running. Close this window or press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        webview.create_window(TITLE, url, width=1280, height=840, min_size=(900, 600))
        webview.start()  # blocks until the window is closed
    server.should_exit = True


if __name__ == "__main__":
    main()
