"""vstash — local document memory with instant semantic search."""

__version__ = "0.36.0"

from .memory import Memory
from .models import (
    AskResult,
    ChunkInfo,
    DocumentInfo,
    ExplainInfo,
    IngestResult,
    SearchResult,
    StoreStats,
)

__all__ = [
    "Memory",
    "AskResult",
    "ChunkInfo",
    "DocumentInfo",
    "ExplainInfo",
    "IngestResult",
    "SearchResult",
    "StoreStats",
]
