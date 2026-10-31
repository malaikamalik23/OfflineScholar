"""Pydantic request/response models (the API contract)."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocType(str, Enum):
    lecture = "lecture"
    past_paper = "past_paper"
    syllabus = "syllabus"


class SubjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be blank")
        return v


class SubjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    subject_id: int
    filename: str
    doc_type: str
    status: str  # pending | processing | ready | failed
    error: str | None = None
    warnings: list[str] = []
    page_count: int = 0
    chunk_count: int = 0
    ocr_pages: int = 0
    size_bytes: int
    created_at: datetime
    processed_at: datetime | None = None


class DocumentStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    status: str
    error: str | None = None
    warnings: list[str] = []
    page_count: int = 0
    chunk_count: int = 0


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    subject_id: int | None = None  # None -> search all subjects
    top_k: int = Field(default=4, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question is too short")
        return v


class SourceOut(BaseModel):
    document_name: str
    page_number: int
    excerpt: str
    score: float | None = None
    document_id: int | None = None


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[SourceOut]
    confidence: float = Field(description="Best retrieval cosine similarity (0 if nothing relevant)")


class QuizAnswerIn(BaseModel):
    topic: str = Field(min_length=1, max_length=200)
    correct: bool
    question_ref: str | None = Field(default=None, max_length=200)


class QuizGenerate(BaseModel):
    subject_id: int
    topic: str = Field(min_length=2, max_length=200)
    n_questions: int = Field(default=5, ge=1, le=20)

    @field_validator("topic")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("topic is too short")
        return v


class QuizSubmit(BaseModel):
    subject_id: int
    answers: list[QuizAnswerIn] = Field(min_length=1, max_length=500)


class TopicStat(BaseModel):
    topic: str
    attempts: int
    correct: int
    accuracy: float


class QuizResult(BaseModel):
    attempt_id: int
    total: int
    correct: int
    score: float
    topics: list[TopicStat]
