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
    rrf_total: float = Field(default=0.0, description="Combined RRF score")
    mmr_penalty: float = Field(
        default=0.0, description="MMR applied penalty: (1-lambda)*max_similarity"
    )
    rrf_vec_weight: float | None = Field(
        default=None, description="Adaptive RRF weight applied to vector component"
    )
    rrf_fts_weight: float | None = Field(
        default=None, description="Adaptive RRF weight applied to FTS component"
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


# ------------------------------------------------------------------ #
# Miss analysis (#108)                                                #
# ------------------------------------------------------------------ #


class StageVerdict(BaseModel):
    """Outcome of a single chunk at one stage of the search pipeline.

    Used by miss analysis to explain *why* an expected document did
    not appear in the top-k of a query.
    """

    stage: Literal[
        "vector_search",
        "distance_cutoff",
        "fts_search",
        "rrf_fusion",
        "recency_boost",
        "mmr_dedup",
        "top_k_cutoff",
    ] = Field(description="Pipeline stage being inspected")
    passed: bool = Field(description="Whether the chunk survived this stage")
    rank: int | None = Field(
        default=None, description="0-indexed rank of the chunk at this stage, if applicable"
    )
    score: float | None = Field(
        default=None, description="Numeric score relevant to this stage (distance, RRF, etc.)"
    )
    detail: str = Field(description="Human-readable explanation of what happened")
    counterfactual: str | None = Field(
        default=None,
        description="What would have changed the verdict ('would have passed if X')",
    )


class MissAnalysisActualResult(BaseModel):
    """A chunk that DID appear in the top-k, included for context."""

    rank: int
    chunk_id: int
    path: str
    title: str
    score: float


class MissAnalysis(BaseModel):
    """Diagnosis of why an expected document did not appear in search results.

    See ``VstashStore.miss_analysis()`` for the full pipeline trace.
    Returned by ``Memory.miss_analysis()`` and the CLI ``--miss`` flag.
    """

    query: str = Field(description="The search query that was run")
    expected_path: str | None = Field(
        default=None, description="Path of the expected document (if specified by path)"
    )
    expected_chunk_id: int | None = Field(
        default=None,
        description="ID of the specific chunk evaluated (best match within the expected doc)",
    )
    top_k_requested: int = Field(description="top_k value used for the search")
    appeared_in_results: bool = Field(
        description="True if the expected document IS in the top-k (no miss to analyze)"
    )
    final_rank: int | None = Field(
        default=None,
        description="Final rank of the expected chunk in top-k results, or None if it did not make the cut",
    )
    dropped_at: str | None = Field(
        default=None,
        description="Name of the first pipeline stage where the expected chunk was eliminated",
    )
    stage_verdicts: list[StageVerdict] = Field(
        default_factory=list,
        description="Per-stage trace describing what happened to the expected chunk",
    )
    actual_top_k: list[MissAnalysisActualResult] = Field(
        default_factory=list,
        description="The chunks that DID appear in top-k, for comparison",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Actionable, rule-based suggestions to improve the query",
    )
