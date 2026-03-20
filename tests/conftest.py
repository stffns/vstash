"""Shared test fixtures for Vex test suite."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from vstash.config import VstashConfig
from vstash.embed import get_embedding_dim
from vstash.store import VstashStore


def _sqlite_vec_available() -> bool:
    """Check if sqlite-vec extension loading is available."""
    try:
        import sqlite_vec
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute("SELECT vec_version()")
        conn.close()
        return True
    except Exception:
        return False


requires_sqlite_vec = pytest.mark.skipif(
    not _sqlite_vec_available(),
    reason="sqlite-vec extension or enable_load_extension not available",
)


@pytest.fixture
def sample_config() -> VstashConfig:
    """Create a VstashConfig with testing defaults."""
    return VstashConfig()


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    """Provide a temporary database path for testing."""
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def sample_store(tmp_db_path: str, sample_config: VstashConfig) -> VstashStore:
    """Create a VstashStore with a temporary database.

    Yields the store and closes it after the test.
    """
    if not _sqlite_vec_available():
        pytest.skip("sqlite-vec not available")
    dim = get_embedding_dim(sample_config.embeddings.model)
    store = VstashStore(tmp_db_path, embedding_dim=dim)
    yield store  # type: ignore[misc]
    store.close()


@pytest.fixture
def populated_store(sample_store: VstashStore) -> VstashStore:
    """Create a VstashStore with some pre-loaded documents.

    Adds two documents with simple embeddings for search testing.
    """
    dim = sample_store.embedding_dim

    # Document 1: about Python programming
    chunks_1 = [
        "Python is a high-level programming language known for its simplicity.",
        "Python supports multiple paradigms including object-oriented and functional.",
    ]
    embeddings_1 = [[0.1] * dim, [0.2] * dim]
    sample_store.add_document(
        path="/test/python_guide.md",
        title="Python Guide",
        chunks=chunks_1,
        embeddings=embeddings_1,
        source_type="markdown",
    )

    # Document 2: about machine learning
    chunks_2 = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are a key component of deep learning.",
        "Training data quality affects model performance significantly.",
    ]
    embeddings_2 = [[0.3] * dim, [0.4] * dim, [0.5] * dim]
    sample_store.add_document(
        path="/test/ml_intro.pdf",
        title="ML Introduction",
        chunks=chunks_2,
        embeddings=embeddings_2,
        source_type="pdf",
    )

    return sample_store
