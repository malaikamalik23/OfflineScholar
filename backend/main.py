"""FastAPI application. Run:  uvicorn backend.main:app --reload"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_engine.ingest import ensure_index_consistent
from ai_engine.runtime import Services, build_services, get_services, set_services
from backend.config import Settings, get_settings
from backend.database import configure_database, session_scope
from backend.errors import register_error_handlers
from backend.routes import analysis, chat, documents, health, quiz, subjects
from backend.storage import remove_orphan_uploads

log = logging.getLogger("offline_scholar")


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        configure_database(settings)
        set_services(services or build_services(settings))
        try:
            rebuilt = ensure_index_consistent(get_services())
            if rebuilt:
                log.warning("Rebuilt vector indexes for subjects: %s", rebuilt)
        except Exception:  # noqa: BLE001 - never block startup on index repair
            log.exception("Index consistency check failed")
        with session_scope() as db:  # files a previous delete couldn't remove (locked on Windows)
            removed = remove_orphan_uploads(db, settings.uploads_dir)
            if removed:
                log.info("Removed %d orphaned upload(s)", removed)
        yield

    app = FastAPI(
        title="Offline Scholar API",
        version="0.1.0",
        description="Offline-first study assistant: upload PDFs, ask questions, get cited answers.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
    )
    register_error_handlers(app)

    api = APIRouter(prefix="/api")
    for module in (documents, subjects, chat, quiz, analysis, health):
        api.include_router(module.router)
    app.include_router(api)
    return app


app = create_app()
