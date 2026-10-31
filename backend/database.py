"""SQLite engine/session management."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings
from .models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def configure_database(settings: Settings) -> Engine:
    """(Re)create the engine and tables. Called once at app startup."""
    global _engine, _SessionLocal
    settings.ensure_dirs()
    if _engine is not None:
        _engine.dispose()
    _engine = create_engine(
        settings.db_url, connect_args={"check_same_thread": False, "timeout": 30}
    )

    @event.listens_for(_engine, "connect")
    def _pragmas(dbapi_conn, _):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    Base.metadata.create_all(_engine)
    return _engine


def _factory() -> sessionmaker[Session]:
    if _SessionLocal is None:
        raise RuntimeError("Database not configured; call configure_database() first")
    return _SessionLocal


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    db = _factory()()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """For code outside request handlers (ingestion, Member 2's modules)."""
    db = _factory()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
