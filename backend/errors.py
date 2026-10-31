"""Uniform error responses: {"error": {"code": "...", "message": "..."}}."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai_engine.embeddings import EmbeddingModelUnavailableError
from ai_engine.llm import LLMUnavailableError


class AppError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code, self.code, self.message = status_code, code, message


def _body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return JSONResponse(_body(exc.code, exc.message), status_code=exc.status_code)

    @app.exception_handler(LLMUnavailableError)
    async def _llm_error(_: Request, exc: LLMUnavailableError):
        return JSONResponse(_body("llm_unavailable", str(exc)), status_code=503)

    @app.exception_handler(EmbeddingModelUnavailableError)
    async def _embedding_error(_: Request, exc: EmbeddingModelUnavailableError):
        return JSONResponse(_body("embedding_model_unavailable", str(exc)), status_code=503)
