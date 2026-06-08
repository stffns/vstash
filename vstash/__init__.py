"""vstash -- local document memory with instant semantic search."""

__version__ = "0.38.0"

from .errors import (
    BackendError,
    LimitError,
    SchemaVersionError,
    VstashError,
)
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
    "AskResult",
    "BackendError",
    "ChunkInfo",
    "DocumentInfo",
    "ExplainInfo",
    "IngestResult",
    "LimitError",
    "Memory",
    "SchemaVersionError",
    "SearchResult",
    "StoreStats",
    "VstashError",
]
