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
    """A single search result from hybrid RRF search.

    **Score semantics (#135).**  ``score`` is the Reciprocal Rank
    Fusion score with the canonical constant ``k=60``: for one
    candidate, ``vec_weight * 1/(60 + rank_vec) + fts_weight *
    1/(60 + rank_fts)``.  Concretely:

    - **Range:** typically ``[0, ~0.033]``.  ``1/(60 + 0) ≈ 0.0167``
      is the maximum per component, and recency boost / post-processing
      can push the combined value slightly higher.
    - **Within a single result list, higher means more relevant.**
    - **NOT comparable across queries.**  Two queries can produce
      different score scales because the candidate pool, adaptive RRF
      weights, and post-processing all vary by query.  Do not threshold
      the raw score to gate "is this match good enough?" — use
      ``vstash explain`` or ``vstash relevance`` for that.
    - Min/max within one list may be used for *display* normalization
      (a progress bar, a heatmap) but never as a confidence signal.

    The ``chunk_id`` field is a database row ID and is only valid
    for the current index state — it may change on re-ingest.
    """

    chunk_id: int = Field(
        description="Database row ID of the chunk (valid for current index state; may change on re-ingest)"
    )
    text: str = Field(description="Chunk text content")
    title: str = Field(description="Source document title")
    path: str = Field(description="Source document path")
    chunk: int = Field(description="Chunk sequence number within document")
    score: float = Field(
        description=(
            "RRF score (higher = more relevant within one query). "
            "NOT comparable across queries — see class docstring."
        )
    )
    explain: ExplainInfo | None = Field(
        default=None, description="Diagnostic breakdown (when explain=True)"
    )
    added_at: str | None = Field(default=None, description="ISO timestamp of document ingestion")


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


class IntegrityCheck(BaseModel):
    """Result of one integrity check run by ``VstashStore.integrity_check()``.

    Each check is a small, focused invariant — orphan chunks, FTS index
    out of sync, missing vec_chunks rows, etc.  Together they form a
    battery that ``vstash check`` runs against the database after a
    crash or suspected corruption.
    """

    name: str = Field(description="Short identifier for the check")
    description: str = Field(description="Human-readable explanation of what the check verifies")
    passed: bool = Field(description="True if the invariant holds")
    affected_count: int = Field(
        default=0, description="Number of rows / documents that violate the invariant"
    )
    detail: str = Field(default="", description="Free-text detail (sample IDs, error message)")
    repairable: bool = Field(
        default=False,
        description="True if ``integrity_repair()`` knows how to fix this check",
    )


class IntegrityRepair(BaseModel):
    """Result of one repair action attempted by ``VstashStore.integrity_repair()``."""

    name: str = Field(description="Short identifier of the check that was repaired")
    success: bool = Field(description="True if the repair completed without error")
    affected_count: int = Field(default=0, description="Rows / documents touched by the repair")
    detail: str = Field(default="", description="Free-text detail (action taken, error message)")


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
    target_resolution: Literal["explicit_id", "only_chunk", "best_of_n"] = Field(
        default="only_chunk",
        description=(
            "How expected_chunk_id was chosen. "
            "'explicit_id' = caller provided it directly. "
            "'only_chunk' = the expected document has exactly one chunk. "
            "'best_of_n' = the document has multiple chunks; we picked the one "
            "closest to the query embedding (may bias trace toward a best case)."
        ),
    )
    total_chunks_in_doc: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of chunks in the expected document. "
            "When > 1 with resolution='best_of_n', the trace reflects only "
            "the single best-matching chunk, not the document as a whole."
        ),
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
