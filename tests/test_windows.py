"""Regression tests for Windows-specific problems (they simulate Windows, so they run on any OS)."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ai_engine import extraction
from ai_engine.runtime import build_services
from backend.main import create_app

from .conftest import upload


def test_tesseract_found_in_default_windows_folder_when_not_on_path(tmp_path, monkeypatch):
    exe = tmp_path / "Tesseract-OCR" / "tesseract.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    monkeypatch.setattr(extraction.shutil, "which", lambda _cmd: None)
    monkeypatch.setattr(extraction.sys, "platform", "win32")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    assert extraction.find_tesseract() == str(exe)
    assert extraction.tesseract_available()


def test_tesseract_cmd_setting_is_honoured(tmp_path, monkeypatch):
    exe = tmp_path / "custom" / "tesseract.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    monkeypatch.setattr(extraction.shutil, "which", lambda _cmd: None)
    assert extraction.find_tesseract(str(exe)) == str(exe)
    assert extraction.find_tesseract(str(tmp_path / "missing.exe")) is None


def _lock_uploads(monkeypatch, uploads_dir: Path):
    """Make every PDF in uploads_dir undeletable, like a file open in a Windows PDF viewer."""
    real_unlink = Path.unlink

    def locked(self, *a, **kw):
        if self.parent == uploads_dir:
            raise PermissionError(32, "The process cannot access the file because it is being used")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", locked)
    return real_unlink


def test_delete_succeeds_when_pdf_is_locked(client, pdfs, settings, monkeypatch, llm):
    """The delete must still succeed (not 500); the leftover file is removed at the next startup."""
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    assert len(list(settings.uploads_dir.glob("*.pdf"))) == 1  # upload landed in the app's data dir
    real_unlink = _lock_uploads(monkeypatch, settings.uploads_dir)
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert client.get(f"/api/documents/{doc['id']}").status_code == 404
    assert len(list(settings.uploads_dir.glob("*.pdf"))) == 1  # still there, locked

    monkeypatch.setattr(Path, "unlink", real_unlink)  # "viewer closed"
    with TestClient(create_app(settings, build_services(settings, llm=llm))):
        pass
    assert list(settings.uploads_dir.glob("*.pdf")) == []


def test_delete_subject_with_locked_pdf(client, pdfs, settings, monkeypatch):
    doc = upload(client, pdfs["biology"], subject_name="Biology")
    _lock_uploads(monkeypatch, settings.uploads_dir)
    assert client.delete(f"/api/subjects/{doc['subject_id']}").status_code == 204
    assert client.get("/api/subjects").json() == []


def test_upload_uses_app_settings_not_global_default(client, pdfs, settings):
    """Regression: uploads were written to the global default data dir instead of the app's."""
    upload(client, pdfs["biology"], subject_name="Biology")
    assert len(list(settings.uploads_dir.glob("*.pdf"))) == 1


def test_startup_sweep_keeps_referenced_uploads(client, pdfs, settings, llm):
    upload(client, pdfs["biology"], subject_name="Biology")
    with TestClient(create_app(settings, build_services(settings, llm=llm))) as c:
        assert c.get("/api/documents").json()[0]["status"] == "ready"
    assert len(list(settings.uploads_dir.glob("*.pdf"))) == 1
