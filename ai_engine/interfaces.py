"""Stable public interface for other modules (Member 2's exam_engine, scripts, notebooks).

    from ai_engine.interfaces import (
        extract_document, get_document_chunks, generate_embeddings, retrieve_context)
"""
from .extraction import ExtractedDocument, ExtractionError, PageText, extract_document
from .retrieval import (
    ChunkRecord,
    DocumentNotFoundError,
    RetrievedChunk,
    generate_embeddings,
    get_document_chunks,
    retrieve_context,
)

__all__ = [
    "extract_document", "get_document_chunks", "generate_embeddings", "retrieve_context",
    "ExtractedDocument", "ExtractionError", "PageText", "ChunkRecord", "RetrievedChunk",
    "DocumentNotFoundError",
]
