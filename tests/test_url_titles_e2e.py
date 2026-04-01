"""End-to-end tests for real URL title extraction in ingest.py.

Verifies that URL documents get meaningful titles extracted from
parsed content instead of the raw URL.
"""

from __future__ import annotations

from vstash.ingest import _extract_title_from_content, _get_title


class TestUrlTitleExtraction:
    """Test the full title extraction pipeline for URLs."""

    def test_get_title_returns_raw_url_for_urls(self) -> None:
        """_get_title still returns raw URL (title upgrade happens after parse)."""
        url = "https://en.wikipedia.org/wiki/Rust_(programming_language)"
        assert _get_title(url) == url

    def test_get_title_humanizes_file_paths(self) -> None:
        """File paths should get human-readable titles."""
        assert _get_title("/path/to/my-document.md") == "My Document"
        assert _get_title("/path/to/hello_world.py") == "Hello World"

    def test_extract_from_markdown_heading(self) -> None:
        """Markdown H1 heading should be extracted as title."""
        content = """# Rust (programming language)

Rust is a multi-paradigm, general-purpose programming language that
emphasizes performance, type safety, and concurrency.
"""
        assert _extract_title_from_content(content) == "Rust (programming language)"

    def test_extract_from_markitdown_output(self) -> None:
        """markitdown typically outputs the page title as first line."""
        # Simulates markitdown output for a Wikipedia page
        content = """Rust (programming language) - Wikipedia

Rust is a multi-paradigm, general-purpose programming language.
It was designed by Graydon Hoare at Mozilla Research.
"""
        title = _extract_title_from_content(content)
        assert title == "Rust (programming language) - Wikipedia"

    def test_extract_skips_very_short_lines(self) -> None:
        """Lines shorter than 3 chars are skipped, but longer lines are found."""
        content = "OK\n\nThis is the real content about something important."
        assert (
            _extract_title_from_content(content)
            == "This is the real content about something important."
        )

    def test_extract_skips_very_long_lines_but_finds_fallback(self) -> None:
        """Lines over 200 chars are skipped, shorter lines below are found."""
        long_line = "A" * 201
        content = f"{long_line}\n\nReal content here."
        assert _extract_title_from_content(content) == "Real content here."

    def test_extract_prefers_h1_over_plain_text(self) -> None:
        """If first non-empty line is H1, use it."""
        content = "# Official Title\n\nThis is the first paragraph."
        assert _extract_title_from_content(content) == "Official Title"

    def test_extract_handles_blank_leading_lines(self) -> None:
        """Should skip blank lines at the start."""
        content = "\n\n\n# Deep Learning Fundamentals\n\nChapter 1"
        assert _extract_title_from_content(content) == "Deep Learning Fundamentals"

    def test_extract_returns_none_for_empty(self) -> None:
        assert _extract_title_from_content("") is None

    def test_extract_returns_none_for_whitespace(self) -> None:
        assert _extract_title_from_content("   \n  \n  ") is None

    def test_real_world_html_title(self) -> None:
        """Simulates what markitdown returns for a typical blog post."""
        content = """How to Build a RAG Pipeline in 2024

Building a Retrieval-Augmented Generation pipeline requires
careful attention to chunking strategy, embedding model selection,
and prompt engineering.

## Prerequisites

You'll need Python 3.10+ and the following packages...
"""
        title = _extract_title_from_content(content)
        assert title == "How to Build a RAG Pipeline in 2024"
