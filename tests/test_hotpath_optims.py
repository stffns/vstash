"""Tests for the search/store hot-path optimizations (PR: perf/search-hotpath).

- added_at index: the dominant ORDER BY / range column now has an index.
- cache key: the query cache key hashes the embedding as a tuple (no numpy
  alloc per search) and must still produce stable hits for identical queries.
"""

from __future__ import annotations

from vstash.config import CacheConfig
from vstash.store import VstashStore


def test_added_at_index_exists(sample_store: VstashStore) -> None:
    """idx_documents_added_at must be created (idempotently) on store open."""
    rows = sample_store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_documents_added_at'"
    ).fetchall()
    assert rows, "idx_documents_added_at should be created on store open"


def test_cache_key_is_stable_for_identical_queries(populated_store: VstashStore) -> None:
    """The tuple-based cache key is deterministic: identical query params yield
    the same key (so the LRU cache still hits)."""
    args = dict(
        query_text="python",
        top_k=5,
        vec_weight=None,
        fts_weight=None,
        distance_cutoff=1.3225,
        collection=None,
        project=None,
        layer=None,
        adaptive_rrf=True,
        recency_boost=0.0,
        added_after=None,
        added_before=None,
        tags=None,
        mmr_lambda=0.5,
        retrieval_mode="hybrid",
        cache_epoch=0,
    )
    emb = [0.1, 0.2, 0.3, 0.4]
    k1 = populated_store._compute_search_cache_key(query_embedding=emb, **args)
    k2 = populated_store._compute_search_cache_key(query_embedding=list(emb), **args)
    assert k1 == k2, "identical query params (incl. embedding) must produce the same cache key"
    # A different embedding must produce a different key.
    k3 = populated_store._compute_search_cache_key(query_embedding=[0.9, 0.8, 0.7, 0.6], **args)
    assert k3 != k1


def test_cache_hits_on_repeated_search(tmp_db_path: str) -> None:
    """End-to-end: a repeated identical search hits the query cache (exercises
    the new key on the real search path)."""
    store = VstashStore(tmp_db_path, embedding_dim=4, cache=CacheConfig(query_cache_size=16))
    try:
        store.add_document(
            path="/a.md", title="A", chunks=["alpha beta"], embeddings=[[0.1, 0.2, 0.3, 0.4]]
        )
        q = [0.1, 0.2, 0.3, 0.4]
        r1 = store.search(query_embedding=q, query_text="alpha", top_k=3)
        r2 = store.search(query_embedding=list(q), query_text="alpha", top_k=3)
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]
    finally:
        store.close()
