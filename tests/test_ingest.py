"""Tests for vstash.ingest — chunking and source detection."""

from __future__ import annotations

import pytest

from vstash.ingest import _get_source_type, _get_title, _is_url, chunk_text


class TestChunkText:
    """Test the token-level chunking function."""

    def test_empty_string_returns_empty(self) -> None:
        assert chunk_text("", chunk_size=1024, overlap=128) == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert chunk_text("   \n\n  ", chunk_size=1024, overlap=128) == []

    def test_short_text_returns_single_chunk(self) -> None:
        text = "This is a short sentence that should fit in one chunk easily."
        chunks = chunk_text(text, chunk_size=1024, overlap=128)
        assert len(chunks) == 1
        assert text in chunks[0]

    def test_text_too_short_is_filtered(self) -> None:
        # Chunks under 20 chars are filtered out
        text = "tiny"
        chunks = chunk_text(text, chunk_size=1024, overlap=128)
        assert len(chunks) == 0

    def test_long_text_produces_multiple_chunks(self) -> None:
        # Create a text that's definitely longer than one chunk
        text = "word " * 2000  # ~2000 tokens
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1

    def test_overlap_creates_redundancy(self) -> None:
        # With overlap, chunks should share some content
        text = "word " * 500
        chunks_with_overlap = chunk_text(text, chunk_size=100, overlap=50)
        chunks_without_overlap = chunk_text(text, chunk_size=100, overlap=0)
        # More overlap = more chunks
        assert len(chunks_with_overlap) > len(chunks_without_overlap)

    def test_chunk_size_respected(self) -> None:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        text = "word " * 1000
        chunks = chunk_text(text, chunk_size=100, overlap=10)

        for chunk in chunks:
            token_count = len(enc.encode(chunk))
            assert token_count <= 100 + 5  # small tolerance for decode boundaries


class TestIsUrl:
    """Test URL detection."""

    def test_http_url(self) -> None:
        assert _is_url("http://example.com") is True

    def test_https_url(self) -> None:
        assert _is_url("https://example.com/page") is True

    def test_file_path_not_url(self) -> None:
        assert _is_url("/home/user/file.pdf") is False

    def test_relative_path_not_url(self) -> None:
        assert _is_url("file.pdf") is False

    def test_ftp_not_url(self) -> None:
        assert _is_url("ftp://example.com") is False


class TestGetTitle:
    """Test title derivation from sources."""

    def test_url_returns_url(self) -> None:
        url = "https://example.com/article"
        assert _get_title(url) == url

    def test_file_returns_stem(self) -> None:
        assert _get_title("/home/user/my_report.pdf") == "My Report"

    def test_dashes_replaced(self) -> None:
        assert _get_title("my-file-name.txt") == "My File Name"


class TestGetSourceType:
    """Test source type detection from file extensions."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/file.pdf", "pdf"),
            ("/file.docx", "docx"),
            ("/file.py", "code"),
            ("/file.js", "code"),
            ("/file.md", "markdown"),
            ("/file.txt", "text"),
            ("/file.html", "html"),
            ("/file.unknown", "file"),
            ("https://example.com", "url"),
        ],
    )
    def test_source_types(self, path: str, expected: str) -> None:
        assert _get_source_type(path) == expected
