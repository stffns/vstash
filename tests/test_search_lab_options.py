"""Opt-in retrieval options from the Spanish-retrieval lab.

``fts_tokenizer`` (store), ``lexical_gate`` and ``min_vec_keep`` (search).
All three default to the published pipeline; these tests pin the opt-in
behaviour and that the defaults are untouched.
"""

from __future__ import annotations

import math

import pytest

from tests.conftest import requires_sqlite_vec
from vstash.config import CacheConfig
from vstash.store import VstashStore

pytestmark = requires_sqlite_vec

DIM = 8


def _unit(angle: float) -> list[float]:
    v = [0.0] * DIM
    v[0], v[1] = math.cos(angle), math.sin(angle)
    return v


def _store(tmp_db_path: str, **kw) -> VstashStore:
    return VstashStore(tmp_db_path, embedding_dim=DIM, **kw)


def _add(store: VstashStore, path: str, text: str, emb: list[float]) -> None:
    store.add_document(path=path, title=path, chunks=[text], embeddings=[emb])


# ------------------------------------------------------------------ #
# fts_tokenizer                                                        #
# ------------------------------------------------------------------ #


class TestFtsTokenizer:
    def test_unknown_tokenizer_is_rejected(self, tmp_db_path: str) -> None:
        with pytest.raises(ValueError, match="fts_tokenizer"):
            _store(tmp_db_path, fts_tokenizer="porter'); DROP TABLE chunks; --")

    def test_default_keeps_inverted_marks_and_accents_as_token_chars(
        self, tmp_db_path: str
    ) -> None:
        """Documents the historical behaviour the option exists to fix."""
        store = _store(tmp_db_path)
        try:
            assert store._stem_terms(["¿Quién", "REUNIÓN"]) == ["reuniÓn", "¿quién"]
        finally:
            store.close()

    @pytest.mark.parametrize(
        "tokenizer", ["porter unicode61 remove_diacritics 2", "unicode61 remove_diacritics 2"]
    )
    def test_unicode61_folds_case_accents_and_punctuation(
        self, tmp_db_path: str, tokenizer: str
    ) -> None:
        store = _store(tmp_db_path, fts_tokenizer=tokenizer)
        try:
            _add(store, "/a", "La reunión con Ángel fue el martes", _unit(0.0))
            _add(store, "/b", "Presupuesto de la obra", _unit(1.0))
            for query in ("¿Reunion?", "REUNIÓN", "angel", "¡Ángel!"):
                hits = store.search(_unit(0.5), query, top_k=5, retrieval_mode="fts_only")
                assert [h.path for h in hits] == ["/a"], query
            # The stemming helper must agree with the index, or adaptive
            # RRF would look every term up under the wrong key.
            assert store._stem_terms(["¿Quién"]) == ["quien"]
        finally:
            store.close()

    def test_default_misses_what_unicode61_finds(self, tmp_db_path: str) -> None:
        store = _store(tmp_db_path)
        try:
            _add(store, "/a", "La reunión con Ángel fue el martes", _unit(0.0))
            assert store.search(_unit(0.5), "¿Reunión?", top_k=5, retrieval_mode="fts_only") == []
        finally:
            store.close()


class TestFtsTokenizerOnExistingIndex:
    U61 = "porter unicode61 remove_diacritics 2"

    def test_reopen_adopts_the_tokenizer_the_index_was_built_with(self, tmp_db_path: str) -> None:
        store = _store(tmp_db_path, fts_tokenizer=self.U61)
        _add(store, "/a", "La reunión con Ángel", _unit(0.0))
        store.close()
        # No argument: an existing index keeps its own tokenizer.
        store = _store(tmp_db_path)
        try:
            assert store._fts_tokenizer == self.U61
            assert store._stem_terms(["¿Reunión"]) == ["reunion"]
        finally:
            store.close()

    def test_mismatched_request_keeps_the_index_and_warns(
        self, tmp_db_path: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = _store(tmp_db_path)  # historical "porter ascii"
        _add(store, "/a", "La reunión con Ángel", _unit(0.0))
        store.close()
        with caplog.at_level("WARNING"):
            store = _store(tmp_db_path, fts_tokenizer=self.U61)
        try:
            assert "rebuild_fts" in caplog.text
            assert store._fts_tokenizer == "porter ascii"
            assert store._stem_terms(["¿Quién"]) == ["¿quién"]  # still agrees with the index
        finally:
            store.close()

    def test_rebuild_fts_switches_without_touching_vectors(self, tmp_db_path: str) -> None:
        store = _store(tmp_db_path, cache=CacheConfig(query_cache_size=8))
        try:
            _add(store, "/a", "La reunión con Ángel fue el martes", _unit(0.0))
            _add(store, "/b", "Presupuesto de la obra", _unit(1.0))
            kw = {"top_k": 5, "retrieval_mode": "fts_only"}
            assert store.search(_unit(0.5), "¿Reunion?", **kw) == []
            vectors = store._conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]

            store.rebuild_fts(self.U61)

            # Same call as before the rebuild: the query cache must not answer it.
            assert [h.path for h in store.search(_unit(0.5), "¿Reunion?", **kw)] == ["/a"]
            assert store._stem_terms(["¿Quién"]) == ["quien"]
            assert store._conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0] == vectors
            # Deletes still keep the rebuilt index in sync (trigger survives).
            store.delete_document("/a")
            assert store.search(_unit(0.5), "reunion", **kw) == []
            assert [c.passed for c in store.integrity_check()] == [True] * 5
        finally:
            store.close()
        reopened = _store(tmp_db_path)
        try:
            assert reopened._fts_tokenizer == self.U61
        finally:
            reopened.close()

    def test_rebuild_fts_rejects_unknown_tokenizer(self, tmp_db_path: str) -> None:
        store = _store(tmp_db_path)
        try:
            with pytest.raises(ValueError, match="tokenizer"):
                store.rebuild_fts("klingon")
        finally:
            store.close()


# ------------------------------------------------------------------ #
# lexical_gate                                                         #
# ------------------------------------------------------------------ #


@pytest.fixture
def gate_store(tmp_db_path: str):
    """40 notes: every one has the common word, one has five rare terms."""
    store = _store(tmp_db_path)
    for i in range(40):
        text = f"informe semanal numero{i}"
        if i == 7:
            text += " zafiro quetzal obsidiana malaquita turquesa"
        _add(store, f"/n{i}", text, _unit(0.03 * i))
    yield store
    store.close()


def _fts_weight(store: VstashStore, query: str, **kw) -> float:
    hits = store.search(_unit(0.0), query, top_k=5, explain=True, distance_cutoff=25.0, **kw)
    assert hits and hits[0].explain is not None
    return hits[0].explain.rrf_fts_weight


class TestLexicalGate:
    def test_weak_keyword_evidence_lowers_fts_weight(self, gate_store: VstashStore) -> None:
        # "informe" is in every note: the best BM25 hit carries ~no evidence.
        off = _fts_weight(gate_store, "informe")
        on = _fts_weight(gate_store, "informe", lexical_gate=True)
        assert on == pytest.approx(0.1)
        assert off >= on

    def test_strong_keyword_evidence_keeps_the_weight(self, gate_store: VstashStore) -> None:
        query = "zafiro quetzal obsidiana malaquita turquesa"
        off = _fts_weight(gate_store, query)
        on = _fts_weight(gate_store, query, lexical_gate=True)
        assert on == pytest.approx(off)
        assert off > 0.3

    def test_short_query_matched_in_full_keeps_the_weight(self, gate_store: VstashStore) -> None:
        """One rare word is little evidence in absolute terms, but it is the
        whole query: coverage keeps the gate open (the NFCorpus-style case)."""
        off = _fts_weight(gate_store, "zafiro")
        on = _fts_weight(gate_store, "zafiro", lexical_gate=True)
        assert on == pytest.approx(off)
        assert off > 0.3

    def test_long_query_matching_one_common_word_is_gated(self, gate_store: VstashStore) -> None:
        # Six terms asked, the best hit matches only the ubiquitous one.
        query = "informe cobalto niquel wolframio estroncio paladio"
        assert _fts_weight(gate_store, query) > 0.3
        assert _fts_weight(gate_store, query, lexical_gate=True) == pytest.approx(0.1)

    def test_weights_keep_their_sum(self, gate_store: VstashStore) -> None:
        hits = gate_store.search(_unit(0.0), "informe", top_k=5, explain=True, lexical_gate=True)
        info = hits[0].explain
        assert info is not None
        assert info.rrf_vec_weight + info.rrf_fts_weight == pytest.approx(1.0)

    def test_default_is_off(self, gate_store: VstashStore) -> None:
        a = gate_store.search(_unit(0.0), "informe", top_k=10)
        b = gate_store.search(_unit(0.0), "informe", top_k=10, lexical_gate=False)
        assert [r.chunk_id for r in a] == [r.chunk_id for r in b]

    def test_only_hybrid_mode_is_gated(self, gate_store: VstashStore) -> None:
        a = gate_store.search(_unit(0.0), "informe", top_k=10, retrieval_mode="fts_only")
        b = gate_store.search(
            _unit(0.0), "informe", top_k=10, retrieval_mode="fts_only", lexical_gate=True
        )
        assert [(r.chunk_id, r.score) for r in a] == [(r.chunk_id, r.score) for r in b]

    def test_gate_is_part_of_the_cache_key(self, tmp_db_path: str) -> None:
        store = _store(tmp_db_path, cache=CacheConfig(query_cache_size=8))
        try:
            # Vector order and keyword order disagree, so the weight matters.
            for i in range(40):
                text = "informe " * (i + 1) + f"numero{i}"
                _add(store, f"/n{i}", text, _unit(0.03 * i))
            kw = {"top_k": 10, "distance_cutoff": 25.0, "vec_weight": 0.5, "fts_weight": 0.5}
            plain = store.search(_unit(0.0), "informe", **kw)
            gated = store.search(_unit(0.0), "informe", lexical_gate=True, **kw)
            assert [r.score for r in plain] != [r.score for r in gated]
        finally:
            store.close()


# ------------------------------------------------------------------ #
# min_vec_keep                                                         #
# ------------------------------------------------------------------ #


class TestMinVecKeep:
    def test_floor_keeps_the_closest_candidates(self, tmp_db_path: str) -> None:
        store = _store(tmp_db_path)
        try:
            for i in range(6):
                _add(store, f"/n{i}", f"nota {i}", _unit(0.05 + 0.3 * i))
            query = _unit(0.0)
            cut = store.search(query, "zzz", top_k=5, retrieval_mode="vec_only")
            assert [r.path for r in cut] == ["/n0"]  # the ratio cutoff drops the rest
            kept = store.search(query, "zzz", top_k=5, retrieval_mode="vec_only", min_vec_keep=3)
            assert [r.path for r in kept] == ["/n0", "/n1", "/n2"]
        finally:
            store.close()

    def test_floor_never_removes_what_the_cutoff_kept(self, tmp_db_path: str) -> None:
        store = _store(tmp_db_path)
        try:
            for i in range(6):
                _add(store, f"/n{i}", f"nota {i}", _unit(0.5 + 0.01 * i))
            query = _unit(0.0)
            base = store.search(query, "zzz", top_k=6, retrieval_mode="vec_only")
            floored = store.search(query, "zzz", top_k=6, retrieval_mode="vec_only", min_vec_keep=2)
            assert len(base) == 6
            assert [r.path for r in floored] == [r.path for r in base]
        finally:
            store.close()


# ------------------------------------------------------------------ #
# drop_stopwords                                                       #
# ------------------------------------------------------------------ #


class TestDropStopwords:
    def test_helpers_fold_case_accents_and_punctuation(self) -> None:
        from vstash.store._stopwords import drop_stopwords, is_stopword

        assert is_stopword("¿Qué") and is_stopword("The") and is_stopword("más,")
        assert not is_stopword("reunión")
        assert drop_stopwords(["¿Quién", "abogó", "por", "las", "garantías?"]) == [
            "abogó",
            "garantías?",
        ]
        # A query with nothing but function words is left as it is.
        assert drop_stopwords(["de", "la"]) == ["de", "la"]

    def test_stopwords_leave_the_keyword_query_only(self, tmp_db_path: str) -> None:
        store = _store(tmp_db_path)
        try:
            _add(store, "/a", "acta de la reunion", _unit(0.0))
            _add(store, "/b", "el coste de la obra", _unit(0.3))
            kw = {"top_k": 5, "retrieval_mode": "fts_only", "explain": True}
            plain = store.search(_unit(0.0), "la reunion de", **kw)
            assert {h.path for h in plain} == {"/a", "/b"}  # "/b" matches only "de", "la"
            assert plain[0].explain.fts_terms == ["la", "reunion", "de"]
            dropped = store.search(_unit(0.0), "la reunion de", drop_stopwords=True, **kw)
            assert [h.path for h in dropped] == ["/a"]
            assert dropped[0].explain.fts_terms == ["reunion"]
        finally:
            store.close()

    def test_default_is_off_and_part_of_the_cache_key(self, tmp_db_path: str) -> None:
        store = _store(tmp_db_path, cache=CacheConfig(query_cache_size=8))
        try:
            _add(store, "/a", "acta de la reunion", _unit(0.0))
            _add(store, "/b", "el coste de la obra", _unit(0.3))
            kw = {"top_k": 5, "retrieval_mode": "fts_only"}
            assert len(store.search(_unit(0.0), "la reunion de", **kw)) == 2
            assert len(store.search(_unit(0.0), "la reunion de", drop_stopwords=True, **kw)) == 1
            assert len(store.search(_unit(0.0), "la reunion de", **kw)) == 2
        finally:
            store.close()


# ------------------------------------------------------------------ #
# Adapters: services layer and SDK forward the options                  #
# ------------------------------------------------------------------ #


class TestAdaptersForwardLabOptions:
    def test_memory_search_forwards_to_the_store(self, tmp_path, monkeypatch) -> None:
        from vstash import Memory

        (tmp_path / "a.md").write_text("# Acta\n\nLa reunion de arquitectura fue el martes.")
        with Memory(db=tmp_path / "mem.db") as mem:
            mem.add(tmp_path / "a.md")
            seen: dict = {}
            real = mem._store.search

            def spy(*args, **kwargs):
                seen.update(kwargs)
                return real(*args, **kwargs)

            monkeypatch.setattr(mem._store, "search", spy)
            hits = mem.search(
                "la reunion de arquitectura",
                top_k=3,
                lexical_gate=True,
                drop_stopwords=True,
                min_vec_keep=3,
                distance_cutoff=2.0,
            )
            assert hits
            assert seen["lexical_gate"] is True
            assert seen["drop_stopwords"] is True
            assert seen["min_vec_keep"] == 3
            assert seen["distance_cutoff"] == 2.0

            seen.clear()
            mem.search("la reunion de arquitectura", top_k=3)
            assert seen["lexical_gate"] is False
            assert seen["drop_stopwords"] is False
            assert seen["min_vec_keep"] == 0
            assert "distance_cutoff" not in seen  # store default untouched
