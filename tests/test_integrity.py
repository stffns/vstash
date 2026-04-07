"""Tests for ingestion idempotency and integrity check (#134)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from vstash.cli import app
from vstash.config import VstashConfig
from vstash.ingest import ingest
from vstash.store import VstashStore

runner = CliRunner()


# ------------------------------------------------------------------ #
# doc_completeness                                                     #
# ------------------------------------------------------------------ #


class TestDocCompleteness:
    def test_missing_when_path_unknown(self, sample_store: VstashStore) -> None:
        assert sample_store.doc_completeness("/nope.md") == "missing"

    def test_complete_after_clean_ingest(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/clean.md",
            title="Clean",
            chunks=["alpha", "beta"],
            embeddings=[[0.1] * dim, [0.2] * dim],
        )
        assert sample_store.doc_completeness("/clean.md") == "complete"

    def test_partial_when_chunk_count_mismatch(self, sample_store: VstashStore) -> None:
        """Manually corrupt the chunk_count and observe 'partial'."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/corrupt.md",
            title="Corrupt",
            chunks=["a", "b"],
            embeddings=[[0.1] * dim, [0.2] * dim],
        )
        # Bump declared chunk_count to a value that no longer matches.
        sample_store._conn.execute(
            "UPDATE documents SET chunk_count = 99 WHERE path = ?", ["/corrupt.md"]
        )
        sample_store._conn.commit()
        assert sample_store.doc_completeness("/corrupt.md") == "partial"

    def test_partial_when_chunks_table_emptied(self, sample_store: VstashStore) -> None:
        """Drop chunk rows directly to simulate a crash mid-write."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/halfwrite.md",
            title="HalfWrite",
            chunks=["a", "b"],
            embeddings=[[0.1] * dim, [0.2] * dim],
        )
        doc_id = sample_store._conn.execute(
            "SELECT id FROM documents WHERE path = ?", ["/halfwrite.md"]
        ).fetchone()[0]
        sample_store._conn.execute("DELETE FROM chunks WHERE doc_id = ?", [doc_id])
        sample_store._conn.commit()
        assert sample_store.doc_completeness("/halfwrite.md") == "partial"


# ------------------------------------------------------------------ #
# Idempotent ingest                                                    #
# ------------------------------------------------------------------ #


class TestIdempotentIngest:
    def test_second_ingest_skips_when_complete(
        self, sample_store: VstashStore, tmp_path: Path
    ) -> None:
        f = tmp_path / "doc.md"
        f.write_text("hello world\n\nthis is content")
        cfg = VstashConfig()

        first = ingest(str(f), cfg, sample_store)
        assert first.status == "ok"

        second = ingest(str(f), cfg, sample_store)
        assert second.status == "skipped"

    def test_partial_doc_is_reingested_without_force(
        self, sample_store: VstashStore, tmp_path: Path
    ) -> None:
        """If the prior ingest left the DB partial, re-running ingest()
        without --force should heal it."""
        f = tmp_path / "broken.md"
        f.write_text("alpha beta gamma\n\nmore content here")
        cfg = VstashConfig()

        first = ingest(str(f), cfg, sample_store)
        assert first.status == "ok"

        # Simulate a crash that left chunks behind but corrupted
        # chunk_count.
        sample_store._conn.execute(
            "UPDATE documents SET chunk_count = 99 WHERE path = ?",
            [str(f.resolve())],
        )
        sample_store._conn.commit()
        assert sample_store.doc_completeness(str(f.resolve())) == "partial"

        # Re-ingest without --force should heal, not skip.
        healed = ingest(str(f), cfg, sample_store)
        assert healed.status == "ok"
        assert sample_store.doc_completeness(str(f.resolve())) == "complete"


# ------------------------------------------------------------------ #
# integrity_check                                                      #
# ------------------------------------------------------------------ #


class TestIntegrityCheck:
    def test_clean_store_passes_all_checks(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["alpha"],
            embeddings=[[0.1] * dim],
        )
        results = sample_store.integrity_check()
        assert all(r.passed for r in results), [
            f"{r.name}: {r.detail}" for r in results if not r.passed
        ]

    def test_chunk_count_mismatch_detected(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["alpha", "beta"],
            embeddings=[[0.1] * dim, [0.2] * dim],
        )
        sample_store._conn.execute(
            "UPDATE documents SET chunk_count = 17 WHERE path = ?", ["/a.md"]
        )
        sample_store._conn.commit()

        results = {c.name: c for c in sample_store.integrity_check()}
        assert results["chunk_count_parity"].passed is False
        assert results["chunk_count_parity"].affected_count == 1
        assert results["chunk_count_parity"].repairable

    def test_orphan_chunk_detected(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["alpha"],
            embeddings=[[0.1] * dim],
        )
        # Manually orphan a chunk (FK enforcement matters here — we
        # bypass it by writing a chunk with a fabricated doc_id).
        sample_store._conn.execute("PRAGMA foreign_keys = OFF")
        sample_store._conn.execute(
            "INSERT INTO chunks (doc_id, seq, text) VALUES (?, 0, 'orphan')",
            ["nonexistent_doc"],
        )
        sample_store._conn.commit()
        sample_store._conn.execute("PRAGMA foreign_keys = ON")

        results = {c.name: c for c in sample_store.integrity_check()}
        assert results["no_orphan_chunks"].passed is False
        assert results["no_orphan_chunks"].affected_count >= 1


# ------------------------------------------------------------------ #
# integrity_repair                                                     #
# ------------------------------------------------------------------ #


class TestIntegrityRepair:
    def test_repair_chunk_count(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["alpha", "beta"],
            embeddings=[[0.1] * dim, [0.2] * dim],
        )
        sample_store._conn.execute(
            "UPDATE documents SET chunk_count = 17 WHERE path = ?", ["/a.md"]
        )
        sample_store._conn.commit()

        repairs = {r.name: r for r in sample_store.integrity_repair()}
        assert repairs["chunk_count_parity"].success
        assert repairs["chunk_count_parity"].affected_count == 1

        results = {c.name: c for c in sample_store.integrity_check()}
        assert results["chunk_count_parity"].passed

    def test_repair_orphan_chunks(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["alpha"],
            embeddings=[[0.1] * dim],
        )
        sample_store._conn.execute("PRAGMA foreign_keys = OFF")
        sample_store._conn.execute(
            "INSERT INTO chunks (doc_id, seq, text) VALUES (?, 0, 'orphan')",
            ["nonexistent_doc"],
        )
        sample_store._conn.commit()
        sample_store._conn.execute("PRAGMA foreign_keys = ON")

        repairs = {r.name: r for r in sample_store.integrity_repair()}
        assert repairs["no_orphan_chunks"].success
        assert repairs["no_orphan_chunks"].affected_count >= 1

        results = {c.name: c for c in sample_store.integrity_check()}
        assert results["no_orphan_chunks"].passed

    def test_repair_is_no_op_when_clean(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["alpha"],
            embeddings=[[0.1] * dim],
        )
        repairs = sample_store.integrity_repair()
        # Nothing to repair → empty list (no false-positive successes).
        assert repairs == []


# ------------------------------------------------------------------ #
# CLI: vstash check                                                    #
# ------------------------------------------------------------------ #


class TestCheckCLI:
    def test_check_clean_store_exit_zero(self, tmp_path: Path) -> None:
        db = tmp_path / "clean.db"
        # Seed a clean store via direct API
        from vstash.embed import get_embedding_dim

        store = VstashStore(str(db), embedding_dim=get_embedding_dim("BAAI/bge-small-en-v1.5"))
        try:
            store.add_document(
                path="/x.md",
                title="X",
                chunks=["hello"],
                embeddings=[[0.1] * store.embedding_dim],
            )
        finally:
            store.close()

        result = runner.invoke(app, ["check"], env={"VSTASH_DB_PATH": str(db)})
        assert result.exit_code == 0, result.output
        assert "PASS" in result.output

    def test_check_repair_heals_chunk_count(self, tmp_path: Path) -> None:
        db = tmp_path / "broken.db"
        from vstash.embed import get_embedding_dim

        dim = get_embedding_dim("BAAI/bge-small-en-v1.5")
        store = VstashStore(str(db), embedding_dim=dim)
        try:
            store.add_document(
                path="/x.md",
                title="X",
                chunks=["hello"],
                embeddings=[[0.1] * dim],
            )
            store._conn.execute("UPDATE documents SET chunk_count = 99 WHERE path = ?", ["/x.md"])
            store._conn.commit()
        finally:
            store.close()

        result = runner.invoke(app, ["check", "--repair"], env={"VSTASH_DB_PATH": str(db)})
        assert result.exit_code == 0, result.output

    def test_check_json_output(self, tmp_path: Path) -> None:
        import json

        db = tmp_path / "json.db"
        from vstash.embed import get_embedding_dim

        dim = get_embedding_dim("BAAI/bge-small-en-v1.5")
        store = VstashStore(str(db), embedding_dim=dim)
        try:
            store.add_document(
                path="/x.md",
                title="X",
                chunks=["hello"],
                embeddings=[[0.1] * dim],
            )
        finally:
            store.close()

        result = runner.invoke(app, ["check", "--json"], env={"VSTASH_DB_PATH": str(db)})
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert "checks" in payload
        assert isinstance(payload["checks"], list)
        assert all("name" in c and "passed" in c for c in payload["checks"])

    def test_add_force_and_resume_mutually_exclusive(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("hello")
        result = runner.invoke(app, ["add", "--force", "--resume", str(f)])
        assert result.exit_code == 2
        assert "mutually exclusive" in result.output


# ------------------------------------------------------------------ #
# Crash-mid-loop simulation                                            #
# ------------------------------------------------------------------ #


class TestPartialIngestRecovery:
    """Simulate what an interrupted ``vstash add directory/`` looks like
    and verify that re-running it heals the DB."""

    def test_resume_after_simulated_crash(self, sample_store: VstashStore, tmp_path: Path) -> None:
        cfg = VstashConfig()
        files = []
        for i in range(3):
            f = tmp_path / f"doc_{i}.md"
            f.write_text(f"content for document {i}\n\nmore words here")
            files.append(f)

        # First batch: ingest all 3.
        for f in files:
            ingest(str(f), cfg, sample_store)

        # Simulate crash during second pass: corrupt one of them so it
        # looks partial.
        target = str(files[1].resolve())
        sample_store._conn.execute("UPDATE documents SET chunk_count = 99 WHERE path = ?", [target])
        sample_store._conn.commit()
        assert sample_store.doc_completeness(target) == "partial"

        # Re-run ingest on the whole set: clean ones skip, partial heals.
        statuses = [ingest(str(f), cfg, sample_store).status for f in files]
        assert statuses[0] == "skipped"
        assert statuses[1] == "ok"  # healed
        assert statuses[2] == "skipped"
        assert sample_store.doc_completeness(target) == "complete"
