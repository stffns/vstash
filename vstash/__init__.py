"""vstash — local document memory with instant semantic search."""

__version__ = "0.4.0"

from .memory import Memory
from .models import DocumentInfo, IngestResult, SearchResult, StoreStats

__all__ = [
    "Memory",
    "DocumentInfo",
    "IngestResult",
    "SearchResult",
    "StoreStats",
]
