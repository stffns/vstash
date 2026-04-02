"""vstash — local document memory with instant semantic search."""

__version__ = "0.13.0"

from .memory import Memory
from .models import ChunkInfo, DocumentInfo, IngestResult, SearchResult, StoreStats

__all__ = [
    "Memory",
    "ChunkInfo",
    "DocumentInfo",
    "IngestResult",
    "SearchResult",
    "StoreStats",
]
