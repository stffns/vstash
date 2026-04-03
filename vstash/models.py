"""
models.py — Typed result models for vstash operations.

All structured data flowing between modules uses Pydantic BaseModel
instead of raw dicts. This ensures validation, type safety, and
clear contracts between layers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IngestResult(BaseModel):
    """Result of ingesting a single document."""

    status: Literal["ok", "empty", "skipped", "error"] = Field(description="Ingestion outcome")
    source: str = Field(description="Original file path or URL")
    doc_id: str | None = Field(default=None, description="Hash-based document ID")
    title: str | None = Field(default=None, description="Derived document title")
    chunks: int = Field(default=0, description="Number of chunks generated")
    chars: int = Field(default=0, description="Total character count of source text")
    elapsed_s: float = Field(default=0.0, description="Ingestion time in seconds")
    error: str | None = Field(default=None, description="Error message if status is 'error'")


class ExplainInfo(BaseModel):
    """Diagnostic breakdown of why a chunk ranked where it did."""

    vec_rank: int | None = Field(default=None, description="Rank in vector search (0-based)")
    vec_distance: float | None = Field(default=None, description="Cosine distance from query")
    fts_rank: int | None = Field(default=None, description="Rank in FTS5 keyword search (0-based)")
    rrf_vec: float = Field(default=0.0, description="RRF contribution from vector search")
    rrf_fts: float = Field(default=0.0, description="RRF contribution from FTS5 search")
    rrf_total: float = Field(default=0.0, description="Combined RRF score before scoring")
    freq_score: float | None = Field(default=None, description="Normalized frequency score [0,1]")
    decay_days: float | None = Field(default=None, description="Days since last access")
    gamma: float | None = Field(default=None, description="Scoring maturity gate (0=off, 1=full)")
    effective_beta: float | None = Field(default=None, description="Beta * gamma applied")
    mmr_penalty: float = Field(
        default=0.0, description="MMR applied penalty: (1-lambda)*max_similarity"
    )
    fts_terms: list[str] = Field(default_factory=list, description="FTS query terms searched")


class SearchResult(BaseModel):
    """A single search result from hybrid RRF search."""

    chunk_id: int = Field(
        description="Database row ID of the chunk (valid for current index state; may change on re-ingest)"
    )
    text: str = Field(description="Chunk text content")
    title: str = Field(description="Source document title")
    path: str = Field(description="Source document path")
    chunk: int = Field(description="Chunk sequence number within document")
    score: float = Field(description="RRF score (higher = more relevant)")
    explain: ExplainInfo | None = Field(
        default=None, description="Diagnostic breakdown (when explain=True)"
    )


class DocumentInfo(BaseModel):
    """Metadata about an ingested document."""

    path: str = Field(description="Absolute file path or URL")
    title: str = Field(description="Document title")
    source_type: str = Field(description="Type: pdf, docx, code, url, etc.")
    collection: str = Field(default="default", description="Named collection")
    project: str | None = Field(default=None, description="Project tag from frontmatter")
    layer: str | None = Field(default=None, description="Layer tag from frontmatter")
    tags: str | None = Field(default=None, description="Comma-separated tags from frontmatter")
    chunk_count: int = Field(description="Number of stored chunks")
    char_count: int = Field(description="Total character count")
    added_at: str = Field(description="ISO timestamp of ingestion")


class ChunkInfo(BaseModel):
    """A single chunk retrieved by ID."""

    chunk_id: int = Field(description="Database row ID of the chunk")
    doc_id: str = Field(description="Parent document hash ID")
    chunk: int = Field(description="Chunk sequence number within document")
    text: str = Field(description="Chunk text content")
    title: str = Field(description="Source document title")
    path: str = Field(description="Source document path")
    collection: str = Field(default="default", description="Document collection")


class StoreStats(BaseModel):
    """Aggregate statistics about the vstash memory store."""

    documents: int = Field(description="Total document count")
    chunks: int = Field(description="Total chunk count")
    collections: int = Field(default=0, description="Number of distinct collections")
    db_size_mb: float = Field(description="Database file size in MB")
    db_path: str = Field(description="Absolute path to database file")
