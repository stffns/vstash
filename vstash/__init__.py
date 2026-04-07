"""vstash — local document memory with instant semantic search."""

__version__ = "0.22.0"

from .memory import Memory
from .models import ChunkInfo, DocumentInfo, ExplainInfo, IngestResult, SearchResult, StoreStats

__all__ = [
    "Memory",
    "ChunkInfo",
    "DocumentInfo",
    "ExplainInfo",
    "IngestResult",
    "SearchResult",
    "StoreStats",
]
