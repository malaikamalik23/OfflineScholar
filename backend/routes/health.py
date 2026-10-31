from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from ai_engine.extraction import tesseract_available
from ai_engine.runtime import get_services
from backend.database import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health(db: Session = Depends(get_db)):
    """Cheap readiness check (does not load the embedding model)."""
    svc = get_services()
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    llm = svc.llm.status()
    llm_ok = bool(llm.get("reachable") and llm.get("model_installed"))
    emb_ok = bool(getattr(svc.embedder, "available", True))
    return {
        "status": "ok" if db_ok and llm_ok and emb_ok else "degraded",
        "database": {"ok": db_ok},
        "llm": llm,
        "embeddings": {"model": svc.embedder.name, "loaded": bool(getattr(svc.embedder, "loaded", False)),
                       "available": emb_ok},
        "ocr": {"enabled": svc.settings.ocr_enabled,
                "tesseract_installed": tesseract_available(svc.settings.tesseract_cmd)},
        "offline_mode": svc.settings.offline,
    }
