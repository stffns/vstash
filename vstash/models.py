"""
models.py — Typed result models for vstash operations.

All structured data flowing between modules uses Pydantic BaseModel
instead of raw dicts. This ensures validation, type safety, and
clear contracts between layers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResult(BaseModel):
    """Result of ingesting a single document."""

    status: str = Field(description="'ok', 'empty', or 'error'")
    source: str = Field(description="Original file path or URL")
    doc_id: str | None = Field(default=None, description="Hash-based document ID")
    title: str | None = Field(default=None, description="Derived document title")
    chunks: int = Field(default=0, description="Number of chunks generated")
    chars: int = Field(default=0, description="Total character count of source text")
    elapsed_s: float = Field(default=0.0, description="Ingestion time in seconds")
    error: str | None = Field(default=None, description="Error message if status is 'error'")


class SearchResult(BaseModel):
    """A single search result from hybrid RRF search."""

    text: str = Field(description="Chunk text content")
    title: str = Field(description="Source document title")
    path: str = Field(description="Source document path")
    chunk: int = Field(description="Chunk sequence number within document")
    score: float = Field(description="RRF score (higher = more relevant)")


class DocumentInfo(BaseModel):
    """Metadata about an ingested document."""

    path: str = Field(description="Absolute file path or URL")
    title: str = Field(description="Document title")
    source_type: str = Field(description="Type: pdf, docx, code, url, etc.")
    chunk_count: int = Field(description="Number of stored chunks")
    char_count: int = Field(description="Total character count")
    added_at: str = Field(description="ISO timestamp of ingestion")


class StoreStats(BaseModel):
    """Aggregate statistics about the Vex memory store."""

    documents: int = Field(description="Total document count")
    chunks: int = Field(description="Total chunk count")
    db_size_mb: float = Field(description="Database file size in MB")
    db_path: str = Field(description="Absolute path to database file")
