"""Tests for the ingest_batch helper (issue #266).

The helper is the coalescing route shared by ``ingest_directory`` and
the ``watch.py`` worker. Contracts under test:

- N sources are routed through a single ``add_documents_batch`` call
  inside one ``batch_mode(defer_fts=True)`` context (transaction +
  deferred FTS + single snapvec vstack).
- Per-file error isolation: one bad source does not kill the rest,
  the bad one shows up as ``status="error"`` in its slot of the
  output list and the remaining sources are ingested successfully.
- Empty input is a no-op and returns an empty list.
"""

from __future__ import annotations

from pathlib import Path


from tests.conftest import requires_sqlite_vec
from vstash.config import VstashConfig
from vstash.ingest import ingest_batch
from vstash.store import VstashStore


pytestmark = requires_sqlite_vec


_PADDING = (
    "This document has enough prose to survive the minimum-chunk "
    "filter after markdown stripping and whitespace normalisation. "
    "Without this padding the chunker reports an empty document and "
    "the batch test sees status='empty' for every file, which masks "
    "the coalescing contract we actually want to exercise."
)


def _write(tmp_path: Path, name: str, body: str) -> str:
    p = tmp_path / name
    p.write_text(f"{body}\n\n{_PADDING}\n")
    return str(p)


class TestIngestBatchCoalescing:
    """Phase 2 must hit one batch call inside one batch_mode context."""

    def test_single_batch_call_for_many_files(
        self,
        tmp_path: Path,
        sample_store: VstashStore,
        sample_config: VstashConfig,
        monkeypatch,
    ) -> None:
        sources = [
            _write(tmp_path, f"doc_{i}.md", f"# Doc {i}\n\nContent for document {i}.")
            for i in range(5)
        ]

        batch_calls: list[int] = []
        batch_mode_calls: list[bool] = []
        real_add_batch = sample_store.add_documents_batch
        real_batch_mode = sample_store.batch_mode

        def spy_add_batch(docs):
            batch_calls.append(len(docs))
            return real_add_batch(docs)

        def spy_batch_mode(*, defer_fts: bool = False):
            batch_mode_calls.append(defer_fts)
            return real_batch_mode(defer_fts=defer_fts)

        monkeypatch.setattr(sample_store, "add_documents_batch", spy_add_batch)
        monkeypatch.setattr(sample_store, "batch_mode", spy_batch_mode)

        results = ingest_batch(sources, sample_config, sample_store)

        assert len(results) == 5
        assert all(r.status == "ok" for r in results)
        assert batch_calls == [5], (
            f"expected one add_documents_batch([5 docs]) call, got {batch_calls}"
        )
        assert batch_mode_calls == [True], (
            f"expected one batch_mode(defer_fts=True), got {batch_mode_calls}"
        )


class TestIngestBatchErrorIsolation:
    """A bad file must not kill the batch."""

    def test_bad_file_isolated_from_rest(
        self,
        tmp_path: Path,
        sample_store: VstashStore,
        sample_config: VstashConfig,
    ) -> None:
        good_a = _write(tmp_path, "good_a.md", "# A\n\nAlpha content here.")
        bad = str(tmp_path / "does_not_exist.md")  # path that is not on disk
        good_b = _write(tmp_path, "good_b.md", "# B\n\nBravo content here.")

        results = ingest_batch([good_a, bad, good_b], sample_config, sample_store)

        assert len(results) == 3
        statuses = [r.status for r in results]
        # The two good files land as status="ok" (order preserved among
        # successful docs in phase 2). The bad file lands as the error
        # from phase 1.
        assert "error" in statuses, f"expected at least one error result, got {statuses}"
        assert statuses.count("ok") == 2, f"expected 2 ok results, got {statuses}"


class TestIngestBatchEmpty:
    def test_empty_list_returns_empty_noop(
        self,
        sample_store: VstashStore,
        sample_config: VstashConfig,
        monkeypatch,
    ) -> None:
        called: list[str] = []
        monkeypatch.setattr(
            sample_store,
            "add_documents_batch",
            lambda docs: called.append("add_documents_batch") or [],
        )
        results = ingest_batch([], sample_config, sample_store)
        assert results == []
        assert called == [], "empty input must not call add_documents_batch"


class TestIngestDirectoryStillWorks:
    """Regression: ingest_directory routes through ingest_batch and must
    keep its existing behaviour (progress bar on, limits enforced)."""

    def test_directory_returns_one_result_per_file(
        self,
        tmp_path: Path,
        sample_store: VstashStore,
        sample_config: VstashConfig,
    ) -> None:
        from vstash.ingest import ingest_directory

        for i in range(3):
            _write(tmp_path, f"dir_doc_{i}.md", f"# Dir Doc {i}\n\nBody {i}.")

        results = ingest_directory(str(tmp_path), sample_config, sample_store)

        assert len(results) == 3
        assert all(r.status == "ok" for r in results)
