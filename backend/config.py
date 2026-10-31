"""Application configuration. All values can be overridden with OS_* env vars or a .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OS_", env_file=".env", extra="ignore")

    # Storage
    data_dir: Path = ROOT / "data"
    database_url: str | None = None  # defaults to sqlite in data_dir

    # Local LLM (Ollama)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout: float = 180.0
    llm_temperature: float = 0.1
    llm_num_ctx: int = 4096

    # Embeddings. "hashing" is a dependency-free fallback intended for tests/CI only.
    embedding_backend: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    offline: bool = False  # True -> forbid HuggingFace network access

    # Chunking / retrieval
    chunk_size: int = 900
    chunk_overlap: int = 150
    default_top_k: int = 4
    min_score: float = 0.25  # cosine similarity below which a passage is considered irrelevant

    # OCR (needs the `tesseract` binary; silently skipped if missing)
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_dpi: int = 200

    # API
    max_upload_mb: int = 50
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'offline_scholar.db').as_posix()}"

    @property
    def resolved_embedding_model(self) -> str:
        """Prefer a copy saved by scripts/download_models.py so no network is needed."""
        local = self.models_dir / self.embedding_model.split("/")[-1]
        return str(local) if local.is_dir() else self.embedding_model

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
