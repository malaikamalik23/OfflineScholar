"""Bridge to Member 2's exam_engine. We only call it; we never reimplement its algorithms.

Expected contract (see docs/INTEGRATION.md): the package `exam_engine.api` exposes

    analyze_subject(subject_id: int, db: sqlalchemy.orm.Session) -> dict      # JSON-serialisable
    recommend_revision(subject_id: int, db: sqlalchemy.orm.Session) -> dict   # JSON-serialisable
"""
from __future__ import annotations

import importlib
from typing import Any

from sqlalchemy.orm import Session

from .errors import AppError


def _call(func_name: str, subject_id: int, db: Session) -> Any:
    try:
        module = importlib.import_module("exam_engine.api")
    except ModuleNotFoundError as exc:
        if exc.name in ("exam_engine", "exam_engine.api"):
            raise AppError(501, "exam_engine_unavailable",
                           "The exam engine (Member 2) is not installed yet.") from exc
        raise
    func = getattr(module, func_name, None)
    if func is None:
        raise AppError(501, "exam_engine_unavailable", f"exam_engine.api.{func_name} is not implemented yet.")
    return func(subject_id, db)


def get_analysis(subject_id: int, db: Session) -> Any:
    return _call("analyze_subject", subject_id, db)


def get_revision(subject_id: int, db: Session) -> Any:
    return _call("recommend_revision", subject_id, db)
