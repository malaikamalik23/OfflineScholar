from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_engine.llm import LLMUnavailableError  # noqa: E402
from ai_engine.runtime import build_services  # noqa: E402
from backend.config import Settings  # noqa: E402
from backend.main import create_app  # noqa: E402


class FakeLLM:
    """Test double for Ollama: answers from the first context passage and cites it."""

    model = "fake"

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.reply: str | None = None
        self.down = False

    def chat(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.down:
            raise LLMUnavailableError("Cannot reach Ollama")
        if self.reply is not None:
            return self.reply
        first = user.split("\n\n")[1].split("\n", 1)[1]  # text of passage [1]
        return f"{first.split('. ')[0].strip('.')}. [1]"

    def status(self) -> dict:
        return {"reachable": not self.down, "model": "fake", "model_installed": not self.down}


def make_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    doc.save(path)
    return path


def make_scanned_pdf(path: Path, text: str) -> Path:
    """Render text to an image and embed only the image: no text layer, like a scan."""
    src = pymupdf.open()
    src.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 400), text, fontsize=20)
    png = src[0].get_pixmap(dpi=150).tobytes("png")
    out = pymupdf.open()
    out.new_page().insert_image(pymupdf.Rect(0, 0, 595, 842), stream=png)
    out.save(path)
    return path


BIOLOGY = [
    "Photosynthesis is the process by which green plants convert light energy into chemical energy. "
    "It takes place in the chloroplasts, where chlorophyll absorbs sunlight. "
    "The light reactions produce ATP and NADPH, which power the Calvin cycle.",
    "Mitosis is the division of a eukaryotic cell nucleus that produces two identical daughter cells. "
    "The stages of mitosis are prophase, metaphase, anaphase and telophase.",
]
HISTORY = [
    "The Treaty of Versailles was signed in 1919 and ended the state of war between Germany and the Allied Powers. "
    "It required Germany to pay reparations and limited the size of its army.",
]


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", embedding_backend="hashing", min_score=0.2, _env_file=None)


@pytest.fixture
def client(settings, llm):
    app = create_app(settings, build_services(settings, llm=llm))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def pdfs(tmp_path) -> dict[str, Path]:
    return {
        "biology": make_pdf(tmp_path / "Biology 01.pdf", BIOLOGY),
        "history": make_pdf(tmp_path / "History 01.pdf", HISTORY),
    }


def upload(client: TestClient, path: Path, **form) -> dict:
    with path.open("rb") as f:
        r = client.post("/api/documents/upload", files={"file": (path.name, f, "application/pdf")}, data=form)
    assert r.status_code == 202, r.text
    return r.json()
