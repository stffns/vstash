"""Tests for query result LRU cache in VstashStore."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from vstash.config import CacheConfig
from vstash.store import VstashStore

from .conftest import requires_sqlite_vec


@requires_sqlite_vec
class TestQueryCache:
    """Query LRU cache: hit, miss, invalidation, opt-out."""

    @pytest.fixture
    def cached_store(self, tmp_db_path: str) -> VstashStore:
        dim = 384
        store = VstashStore(
            tmp_db_path,
            embedding_dim=dim,
            cache=CacheConfig(query_cache_size=8),
        )
        chunks = [
            "Python is a high-level programming language.",
            "Machine learning uses neural networks.",
        ]
        embeddings = [[0.1] * dim, [0.3] * dim]
        store.add_document(
            path="/test/doc.md",
            title="Test Doc",
            chunks=chunks,
            embeddings=embeddings,
        )
        yield store
        store.close()

    def test_cache_hit_returns_same_results(self, cached_store: VstashStore):
        q_emb = [0.1] * cached_store.embedding_dim
        r1 = cached_store.search(q_emb, "python", top_k=2)
        r2 = cached_store.search(q_emb, "python", top_k=2)
        assert len(r1) > 0
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]

    def test_cache_hit_returns_copy(self, cached_store: VstashStore):
        q_emb = [0.1] * cached_store.embedding_dim
        r1 = cached_store.search(q_emb, "python", top_k=2)
        r2 = cached_store.search(q_emb, "python", top_k=2)
        assert r1 is not r2

    def test_cache_miss_on_different_query(self, cached_store: VstashStore):
        dim = cached_store.embedding_dim
        r1 = cached_store.search([0.1] * dim, "python", top_k=2)
        r2 = cached_store.search([0.3] * dim, "neural", top_k=2)
        assert [r.chunk_id for r in r1] != [r.chunk_id for r in r2]

    def test_cache_invalidated_on_add(self, cached_store: VstashStore):
        dim = cached_store.embedding_dim
        q_emb = [0.1] * dim
        cached_store.search(q_emb, "python", top_k=5)
        epoch_before = cached_store._cache_epoch

        cached_store.add_document(
            path="/test/new.md",
            title="New",
            chunks=["Python decorators are powerful."],
            embeddings=[[0.1] * dim],
        )

        assert cached_store._cache_epoch == epoch_before + 1
        assert len(cached_store._query_cache) == 0

    def test_cache_invalidated_on_delete(self, cached_store: VstashStore):
        q_emb = [0.1] * cached_store.embedding_dim
        cached_store.search(q_emb, "python", top_k=2)
        assert len(cached_store._query_cache) == 1

        cached_store.delete_document("/test/doc.md")
        assert len(cached_store._query_cache) == 0

    def test_cache_disabled_when_size_zero(self, tmp_db_path: str):
        dim = 384
        store = VstashStore(
            tmp_db_path,
            embedding_dim=dim,
            cache=CacheConfig(query_cache_size=0),
        )
        store.add_document(
            path="/test/doc.md",
            title="Test",
            chunks=["hello world"],
            embeddings=[[0.1] * dim],
        )
        store.search([0.1] * dim, "hello", top_k=2)
        assert len(store._query_cache) == 0
        store.close()

    def test_cache_disabled_by_default(self, tmp_db_path: str):
        dim = 384
        store = VstashStore(tmp_db_path, embedding_dim=dim)
        assert store._cache_config.query_cache_size == 0
        store.close()

    def test_cache_skipped_with_explain(self, cached_store: VstashStore):
        q_emb = [0.1] * cached_store.embedding_dim
        cached_store.search(q_emb, "python", top_k=2, explain=True)
        assert len(cached_store._query_cache) == 0

    def test_lru_eviction(self, tmp_db_path: str):
        dim = 384
        store = VstashStore(
            tmp_db_path,
            embedding_dim=dim,
            cache=CacheConfig(query_cache_size=2),
        )
        store.add_document(
            path="/test/doc.md",
            title="Test",
            chunks=["hello world"],
            embeddings=[[0.1] * dim],
        )
        store.search([0.1] * dim, "a", top_k=1)
        store.search([0.2] * dim, "b", top_k=1)
        store.search([0.3] * dim, "c", top_k=1)
        assert len(store._query_cache) == 2
        store.close()

    def test_different_params_different_cache_entries(self, cached_store: VstashStore):
        q_emb = [0.1] * cached_store.embedding_dim
        cached_store.search(q_emb, "python", top_k=2)
        cached_store.search(q_emb, "python", top_k=5)
        assert len(cached_store._query_cache) == 2

    def test_cache_respects_collection_filter(self, cached_store: VstashStore):
        q_emb = [0.1] * cached_store.embedding_dim
        r1 = cached_store.search(q_emb, "python", top_k=2)
        r2 = cached_store.search(q_emb, "python", top_k=2, collection="nonexistent")
        ids_1 = [r.chunk_id for r in r1]
        ids_2 = [r.chunk_id for r in r2]
        assert ids_1 != ids_2 or (len(r1) > 0 and len(r2) == 0)

    def test_integrity_repair_invalidates_cache(self, cached_store: VstashStore):
        q_emb = [0.1] * cached_store.embedding_dim
        cached_store.search(q_emb, "python", top_k=2)
        assert len(cached_store._query_cache) == 1

        cached_store.integrity_repair()
        assert len(cached_store._query_cache) == 0

    def test_concurrent_cache_access(self, cached_store: VstashStore):
        q_emb = [0.1] * cached_store.embedding_dim
        # Warm cache with a cold search first so all threads hit the cache
        r0 = cached_store.search(q_emb, "python", top_k=2)
        assert len(r0) > 0

        def do_search(_i: int):
            return cached_store.search(q_emb, "python", top_k=2)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(do_search, i) for i in range(20)]
            results = [f.result() for f in futures]

        assert all(len(r) > 0 for r in results)
        ids = [r[0].chunk_id for r in results]
        assert len(set(ids)) == 1

    def test_exception_does_not_cache_partial_result(self, tmp_db_path: str):
        dim = 384
        store = VstashStore(
            tmp_db_path,
            embedding_dim=dim,
            cache=CacheConfig(query_cache_size=8),
        )
        store.add_document(
            path="/test/doc.md",
            title="Test",
            chunks=["hello world"],
            embeddings=[[0.1] * dim],
        )
        # Corrupt chunks table to force an error mid-search
        store._conn.execute("DROP TABLE chunks")
        try:
            store.search([0.1] * dim, "hello", top_k=2)
        except Exception:
            pass
        assert len(store._query_cache) == 0
        store.close()
