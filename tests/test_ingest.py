"""Tests for vstash.ingest — chunking and source detection."""

from __future__ import annotations

import pytest

from vstash.ingest import (
    _fixed_window_chunks,
    _merge_small_chunks,
    _split_by_headers,
    _split_by_paragraphs,
    _get_source_type,
    _get_title,
    _is_url,
    chunk_text,
)


# ------------------------------------------------------------------ #
# Semantic chunk_text (high-level)                                     #
# ------------------------------------------------------------------ #


class TestChunkText:
    """Test the semantic chunking function."""

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
        text = "word " * 2000  # ~2000 tokens
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1

    def test_chunk_size_respected(self) -> None:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        text = "word " * 1000
        chunks = chunk_text(text, chunk_size=100, overlap=10)

        for chunk in chunks:
            token_count = len(enc.encode(chunk))
            assert token_count <= 100 + 5  # small tolerance for decode boundaries

    def test_markdown_sections_stay_together(self) -> None:
        """A Markdown doc with small sections should keep header + body as one chunk."""
        text = (
            "# Introduction\n\n"
            "This is the introduction paragraph with enough text to be meaningful.\n\n"
            "# Methods\n\n"
            "This is the methods paragraph with enough text to be meaningful.\n\n"
            "# Results\n\n"
            "This is the results paragraph with enough text to be meaningful."
        )
        chunks = chunk_text(text, chunk_size=1024, overlap=128)
        # With a large chunk_size, all sections should merge into one or a few chunks
        # but the key property is that no section header is torn from its body
        for chunk in chunks:
            if "# Introduction" in chunk:
                assert "introduction paragraph" in chunk
            if "# Methods" in chunk:
                assert "methods paragraph" in chunk
            if "# Results" in chunk:
                assert "results paragraph" in chunk

    def test_oversized_section_gets_split(self) -> None:
        """A section larger than chunk_size should be split at paragraph boundaries."""
        section = "# Big Section\n\n" + "\n\n".join(
            f"Paragraph {i} with enough content to take up tokens. " * 10
            for i in range(20)
        )
        chunks = chunk_text(section, chunk_size=200, overlap=20)
        assert len(chunks) > 1
        # First chunk should start with the header
        assert chunks[0].startswith("# Big Section")

    def test_plain_text_splits_by_paragraphs(self) -> None:
        """Text without Markdown headers should split at paragraph boundaries."""
        paragraphs = [f"Paragraph {i}. " * 20 for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1


# ------------------------------------------------------------------ #
# _split_by_headers                                                    #
# ------------------------------------------------------------------ #


class TestSplitByHeaders:
    """Test Markdown header splitting."""

    def test_no_headers_returns_whole_text(self) -> None:
        text = "Just some plain text without any headers."
        sections = _split_by_headers(text)
        assert len(sections) == 1
        assert sections[0] == text

    def test_single_header(self) -> None:
        text = "# Title\n\nBody content here."
        sections = _split_by_headers(text)
        assert len(sections) == 1
        assert "# Title" in sections[0]
        assert "Body content" in sections[0]

    def test_multiple_headers(self) -> None:
        text = "# First\n\nBody 1\n\n## Second\n\nBody 2\n\n### Third\n\nBody 3"
        sections = _split_by_headers(text)
        assert len(sections) == 3
        assert "# First" in sections[0]
        assert "## Second" in sections[1]
        assert "### Third" in sections[2]

    def test_preamble_before_first_header(self) -> None:
        text = "Some intro text.\n\n# Header\n\nBody."
        sections = _split_by_headers(text)
        assert len(sections) == 2
        assert "Some intro text" in sections[0]
        assert "# Header" in sections[1]

    def test_empty_string(self) -> None:
        sections = _split_by_headers("")
        assert sections == [""]


# ------------------------------------------------------------------ #
# _split_by_paragraphs                                                 #
# ------------------------------------------------------------------ #


class TestSplitByParagraphs:
    """Test paragraph-level splitting."""

    def test_single_paragraph_fits(self) -> None:
        text = "A short paragraph."
        result = _split_by_paragraphs(text, chunk_size=1024, overlap=128)
        assert len(result) == 1

    def test_multiple_small_paragraphs_merge(self) -> None:
        text = "Para one.\n\nPara two.\n\nPara three."
        result = _split_by_paragraphs(text, chunk_size=1024, overlap=128)
        # All small paragraphs should fit in one chunk
        assert len(result) == 1
        assert "Para one." in result[0]
        assert "Para three." in result[0]

    def test_oversized_paragraph_falls_to_fixed_window(self) -> None:
        text = "word " * 500  # one big paragraph, ~500 tokens
        result = _split_by_paragraphs(text, chunk_size=100, overlap=10)
        assert len(result) > 1


# ------------------------------------------------------------------ #
# _merge_small_chunks                                                  #
# ------------------------------------------------------------------ #


class TestMergeSmallChunks:
    """Test small chunk merging."""

    def test_empty_list(self) -> None:
        assert _merge_small_chunks([], chunk_size=1024) == []

    def test_single_chunk(self) -> None:
        result = _merge_small_chunks(["Hello world."], chunk_size=1024)
        assert result == ["Hello world."]

    def test_small_chunks_get_merged(self) -> None:
        # Two tiny chunks should merge into one
        chunks = ["Hi.", "Bye."]
        result = _merge_small_chunks(chunks, chunk_size=1024)
        assert len(result) == 1
        assert "Hi." in result[0]
        assert "Bye." in result[0]

    def test_small_trailing_chunk_gets_merged(self) -> None:
        """Bidirectional: a small trailing chunk merges into a large predecessor."""
        big = "word " * 200  # ~200 tokens, well above _MIN_CHUNK_TOKENS
        chunks = [big, "Small tail."]
        result = _merge_small_chunks(chunks, chunk_size=1024)
        # "Small tail." is below _MIN_CHUNK_TOKENS, so it should merge
        assert len(result) == 1
        assert "Small tail." in result[0]

    def test_two_large_chunks_stay_separate(self) -> None:
        """Two chunks both above _MIN_CHUNK_TOKENS should not merge."""
        big1 = "word " * 200
        big2 = "other " * 200
        result = _merge_small_chunks([big1, big2], chunk_size=2048)
        assert len(result) == 2

    def test_merge_respects_chunk_size_limit(self) -> None:
        """Even if a chunk is small, don't merge if it would exceed chunk_size."""
        big = "word " * 500  # ~500 tokens
        small = "tail."
        result = _merge_small_chunks([big, small], chunk_size=100)
        assert len(result) == 2


# ------------------------------------------------------------------ #
# _fixed_window_chunks (legacy fallback)                               #
# ------------------------------------------------------------------ #


class TestFixedWindowChunks:
    """Test the fixed-size token window fallback."""

    def test_empty_returns_empty(self) -> None:
        assert _fixed_window_chunks("", chunk_size=100, overlap=10) == []

    def test_short_text_single_chunk(self) -> None:
        result = _fixed_window_chunks(
            "Short text that fits in one window.", chunk_size=100, overlap=10,
        )
        assert len(result) == 1

    def test_overlap_equal_to_chunk_size_does_not_loop(self) -> None:
        """overlap >= chunk_size should be clamped, not cause infinite loop."""
        text = "word " * 500
        # This would infinite loop without the guard
        result = _fixed_window_chunks(text, chunk_size=100, overlap=100)
        assert len(result) > 1

    def test_overlap_greater_than_chunk_size_does_not_loop(self) -> None:
        text = "word " * 500
        result = _fixed_window_chunks(text, chunk_size=100, overlap=200)
        assert len(result) > 1

    def test_overlap_creates_more_chunks(self) -> None:
        text = "word " * 500
        with_overlap = _fixed_window_chunks(text, chunk_size=100, overlap=50)
        without_overlap = _fixed_window_chunks(text, chunk_size=100, overlap=0)
        assert len(with_overlap) > len(without_overlap)


# ------------------------------------------------------------------ #
# URL + title + source type (unchanged)                                #
# ------------------------------------------------------------------ #


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
