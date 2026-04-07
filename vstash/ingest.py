"""
ingest.py — Document ingestion pipeline.

Flow: file/URL → markitdown → chunk → FastEmbed → sqlite-vec + FTS5

Chunking: semantic-first (Markdown headers / paragraphs) with token-bounded fallback.
Progress: Rich bars at every stage — never looks frozen.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .config import EXCLUDED_DIRS, MAX_DIR_BYTES, MAX_DIR_FILES, SUPPORTED_EXTENSIONS, VstashConfig
from .embed import embed_texts
from .models import IngestResult
from .store import VstashStore

console = Console(stderr=True)

# Lazy-initialized tiktoken encoder — avoids import-time cost when
# chunking is not used (e.g. MCP vstash_list calls).
_enc = None
_SEPARATOR_TOKENS = None


def _get_enc():
    """Get or lazily initialize the tiktoken encoder."""
    global _enc, _SEPARATOR_TOKENS  # noqa: PLW0603
    if _enc is None:
        import tiktoken

        _enc = tiktoken.get_encoding("cl100k_base")
        _SEPARATOR_TOKENS = len(_enc.encode("\n\n"))
    return _enc


# ------------------------------------------------------------------ #
# Chunking                                                            #
# ------------------------------------------------------------------ #


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into semantically coherent, token-bounded chunks.

    Strategy (in order of priority):
      1. Split at Markdown headers (``#``, ``##``, etc.) — keeps sections intact.
      2. Within oversized sections, split at paragraph boundaries (``\\n\\n``).
      3. Within oversized paragraphs, fall back to fixed-token sliding window.
      4. Merge adjacent chunks that are too small (< ``_MIN_CHUNK_TOKENS``).

    Args:
        text: Raw text to chunk.
        chunk_size: Maximum tokens per chunk.
        overlap: Token overlap between consecutive chunks (used only in
            the fixed-window fallback for oversized paragraphs).

    Returns:
        List of non-empty text chunks.
    """
    if not text or not text.strip():
        return []

    # --- Step 1: split by Markdown headers ---
    sections = _split_by_headers(text)

    # --- Step 2 & 3: ensure each section fits in chunk_size ---
    sized_chunks: list[str] = []
    for section in sections:
        token_count = len(_get_enc().encode(section))
        if token_count <= chunk_size:
            sized_chunks.append(section)
        else:
            # Split oversized section by paragraphs first
            sized_chunks.extend(_split_by_paragraphs(section, chunk_size, overlap))

    # --- Step 4: merge small adjacent chunks ---
    merged = _merge_small_chunks(sized_chunks, chunk_size)

    # --- Filter empty / tiny results ---
    return [c for c in merged if c.strip() and len(c.strip()) > _MIN_CHUNK_CHARS]


# Minimum chunk size in tokens — chunks smaller than this get merged
# with their neighbour to avoid low-quality embeddings.
_MIN_CHUNK_TOKENS = 80
_MIN_CHUNK_CHARS = 20

# Pattern that matches Markdown header lines (# through ######)
_HEADER_RE = re.compile(r"^(#{1,6})\s+", re.MULTILINE)

# ------------------------------------------------------------------ #
# Code-aware splitting — hybrid: tree-sitter → parso → regex          #
# ------------------------------------------------------------------ #

from .code_split import EXT_TO_LANG as _EXT_TO_LANG  # noqa: E402
from .code_split import split_code_blocks as _split_code_blocks  # noqa: E402


def chunk_code(text: str, chunk_size: int, overlap: int, language: str) -> list[str]:
    """Split source code into structurally coherent, token-bounded chunks.

    Strategy (in order of priority):
      1. Split at top-level definitions (functions, classes) using regex.
      2. Oversized blocks fall back to paragraph splitting, then fixed window.
      3. Merge adjacent small blocks.

    Args:
        text: Raw source code text.
        chunk_size: Maximum tokens per chunk.
        overlap: Token overlap for fixed-window fallback.
        language: Language key (python, javascript, go, rust, java).

    Returns:
        List of non-empty code chunks.
    """
    if not text or not text.strip():
        return []

    # Step 1: split by code structure
    blocks = _split_code_blocks(text, language)

    # Step 2: ensure each block fits in chunk_size
    sized_chunks: list[str] = []
    for block in blocks:
        token_count = len(_get_enc().encode(block))
        if token_count <= chunk_size:
            sized_chunks.append(block)
        else:
            # Fall back to paragraph/window splitting for oversized blocks
            sized_chunks.extend(_split_by_paragraphs(block, chunk_size, overlap))

    # Step 3: merge small adjacent chunks
    merged = _merge_small_chunks(sized_chunks, chunk_size)

    return [c for c in merged if c.strip() and len(c.strip()) > _MIN_CHUNK_CHARS]


def _split_by_headers(text: str) -> list[str]:
    """Split text at Markdown header lines, keeping each header with its body.

    Returns a list where each element is one Markdown section (header + body).
    Leading content before the first header is its own element.
    """
    positions = [m.start() for m in _HEADER_RE.finditer(text)]

    if not positions:
        # No headers — return the whole text as a single section
        return [text]

    sections: list[str] = []

    # Content before the first header (if any)
    if positions[0] > 0:
        preamble = text[: positions[0]].strip()
        if preamble:
            sections.append(preamble)

    # Each header + everything until the next header
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    return sections


def _split_by_paragraphs(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text at paragraph boundaries (``\\n\\n``), with token-window fallback.

    Paragraphs that individually exceed *chunk_size* tokens are further
    split using the fixed-size sliding window (``_fixed_window_chunks``).
    """
    paragraphs = re.split(r"\n{2,}", text)
    result: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = len(_get_enc().encode(para))

        # If a single paragraph exceeds chunk_size, split it with fixed window
        if para_tokens > chunk_size:
            # Flush accumulator first
            if current:
                result.append("\n\n".join(current))
                current, current_tokens = [], 0
            result.extend(_fixed_window_chunks(para, chunk_size, overlap))
            continue

        # Account for "\n\n" separator tokens when joining paragraphs
        separator_cost = _SEPARATOR_TOKENS if current else 0

        # Would adding this paragraph overflow?
        if current_tokens + separator_cost + para_tokens > chunk_size and current:
            result.append("\n\n".join(current))
            current, current_tokens = [], 0
            separator_cost = 0

        current.append(para)
        current_tokens += separator_cost + para_tokens

    if current:
        result.append("\n\n".join(current))

    return result


def _fixed_window_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Legacy fixed-size token window chunking — used as last-resort fallback.

    Args:
        text: Raw text to chunk.
        chunk_size: Maximum tokens per chunk.
        overlap: Token overlap between consecutive chunks.

    Returns:
        List of non-empty text chunks.
    """
    enc = _get_enc()
    tokens = enc.encode(text)
    if not tokens:
        return []

    # Guard against infinite loop: overlap must be strictly less than chunk_size
    safe_overlap = min(overlap, chunk_size - 1) if chunk_size > 0 else 0

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens).strip()

        if chunk_text_str and len(chunk_text_str) > _MIN_CHUNK_CHARS:
            chunks.append(chunk_text_str)

        if end >= len(tokens):
            break

        start += chunk_size - safe_overlap

    return chunks


def _merge_small_chunks(chunks: list[str], chunk_size: int) -> list[str]:
    """Merge adjacent chunks that are below ``_MIN_CHUNK_TOKENS`` tokens.

    Merging stops when adding the next chunk would exceed *chunk_size*.
    """
    if not chunks:
        return []

    merged: list[str] = []
    current = chunks[0]
    current_tokens = len(_get_enc().encode(current))

    for chunk in chunks[1:]:
        chunk_tokens = len(_get_enc().encode(chunk))

        can_merge = current_tokens + _SEPARATOR_TOKENS + chunk_tokens <= chunk_size

        # Merge if EITHER side is small (bidirectional merging)
        if can_merge and (current_tokens < _MIN_CHUNK_TOKENS or chunk_tokens < _MIN_CHUNK_TOKENS):
            current = current + "\n\n" + chunk
            current_tokens += _SEPARATOR_TOKENS + chunk_tokens
        else:
            merged.append(current)
            current = chunk
            current_tokens = chunk_tokens

    merged.append(current)
    return merged


# ------------------------------------------------------------------ #
# Source detection                                                    #
# ------------------------------------------------------------------ #


def _read_raw_code(source: str) -> str | None:
    """Read source code file as raw text, returning None if unreadable."""
    try:
        return Path(source).read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return None


def _is_url(source: str) -> bool:
    """Check if a source string is an HTTP(S) URL."""
    try:
        result = urlparse(source)
        return result.scheme in ("http", "https")
    except ValueError:
        return False


def _get_title(source: str) -> str:
    """Derive a human-readable title from a file path or URL."""
    if _is_url(source):
        return source
    path = Path(source)
    return path.stem.replace("_", " ").replace("-", " ").title()


def _extract_title_from_content(text: str) -> str | None:
    """Extract a meaningful title from parsed document content.

    Checks for:
      1. Markdown H1 heading (``# Title``) — scanned within first 20 lines
      2. First non-empty line with 3-200 chars (often the page title from markitdown)

    Returns None if no suitable title is found.
    """
    first_candidate: str | None = None
    for i, line in enumerate(text[:8000].splitlines()):
        if i >= 20:
            break
        stripped = line.strip()
        if not stripped:
            continue
        # Markdown H1 — always preferred
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if title:
                return title
        # Remember first suitable line as fallback
        if first_candidate is None and 3 <= len(stripped) <= 200:
            first_candidate = stripped
    return first_candidate


def _get_source_type(source: str) -> str:
    """Determine the document type from a file extension or URL."""
    if _is_url(source):
        return "url"
    suffix = Path(source).suffix.lower()
    type_map: dict[str, str] = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".pptx": "pptx",
        ".xlsx": "xlsx",
        ".md": "markdown",
        ".txt": "text",
        ".py": "code",
        ".js": "code",
        ".ts": "code",
        ".go": "code",
        ".rs": "code",
        ".java": "code",
        ".tsx": "code",
        ".jsx": "code",
        ".html": "html",
        ".htm": "html",
    }
    return type_map.get(suffix, "file")


# ------------------------------------------------------------------ #
# Main ingest function                                                #
# ------------------------------------------------------------------ #


def ingest(
    source: str,
    cfg: VstashConfig,
    store: VstashStore,
    *,
    force: bool = False,
    collection: str = "default",
    project: str | None = None,
    layer: str | None = None,
    tags: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestResult:
    """Ingest a single file or URL into the store.

    Args:
        source: File path or URL to ingest.
        cfg: Vex configuration.
        store: Vector store instance.
        force: If False, skip documents already in the store.
        collection: Named collection to group this document.

    Returns:
        IngestResult with status, chunk count, timing, etc.
    """
    start_time = time.time()

    source_path = str(Path(source).resolve()) if not _is_url(source) else source
    title = _get_title(source)
    source_type = _get_source_type(source)

    # Idempotent re-ingest (#134).  Without --force, we now distinguish
    # three states instead of "exists / not exists":
    #   - "complete": fully ingested → skip
    #   - "partial":  prior ingest crashed mid-flight → drop the partial
    #     rows and re-ingest fresh (still treated as a non-error path)
    #   - "missing":  ingest from scratch
    if not force:
        status = store.doc_completeness(source_path)
        if status == "complete":
            return IngestResult(
                status="skipped",
                source=source,
                title=title,
            )
        if status == "partial":
            # Drop the half-ingested document so the fresh ingest below
            # produces a clean state instead of duplicating chunks.
            store.delete_document(source_path)

    # Should we try code-aware chunking?
    is_code = source_type == "code" and not _is_url(source) and cfg.chunking.code_aware

    # --- Step 1: Parse ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Parsing[/bold cyan] {task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(Path(source).name if not _is_url(source) else source)
        try:
            if is_code:
                text = _read_raw_code(source_path)
                if text is None:
                    text = _parse(source)
                    is_code = False
            else:
                text = _parse(source)
        except Exception as exc:
            console.print(f"[red]✗ Error parsing {source}: {exc}[/red]")
            return IngestResult(
                status="error",
                source=source,
                error=str(exc),
            )

    if not text or not text.strip():
        console.print(f"[yellow]⚠ No text extracted from {source}[/yellow]")
        return IngestResult(status="empty", source=source)

    # For URLs, try to extract a real title from the parsed content
    if _is_url(source):
        extracted = _extract_title_from_content(text)
        if extracted:
            title = extracted

    char_count = len(text)

    # --- Step 2: Extract frontmatter metadata ---
    frontmatter = _extract_frontmatter(text)
    # Explicit params override frontmatter values
    fm_project = project or frontmatter.get("project")
    fm_layer = layer or frontmatter.get("layer")
    # Coerce non-string scalars to str; warn on dicts/lists
    if fm_project is not None:
        if isinstance(fm_project, (dict, list)):
            console.print(
                f"[yellow]⚠ Frontmatter 'project' is a {type(fm_project).__name__}, "
                f"expected string — ignoring.[/yellow]"
            )
            fm_project = None
        else:
            fm_project = str(fm_project)
    if fm_layer is not None:
        if isinstance(fm_layer, (dict, list)):
            console.print(
                f"[yellow]⚠ Frontmatter 'layer' is a {type(fm_layer).__name__}, "
                f"expected string — ignoring.[/yellow]"
            )
            fm_layer = None
        else:
            fm_layer = str(fm_layer)
    fm_tags_raw = tags or frontmatter.get("tags")
    fm_tags: str | None = None
    if fm_tags_raw:
        if isinstance(fm_tags_raw, list):
            fm_tags = ",".join(str(t) for t in fm_tags_raw)
        elif not isinstance(fm_tags_raw, (dict,)):
            fm_tags = str(fm_tags_raw)
    # Strip frontmatter block from text before chunking
    text = _strip_frontmatter(text)

    # --- Step 3: Chunk ---
    with console.status("[bold cyan]Chunking...[/bold cyan]", spinner="dots"):
        cs = chunk_size if chunk_size is not None else cfg.chunking.size
        co = chunk_overlap if chunk_overlap is not None else cfg.chunking.overlap
        if cs <= 0:
            raise ValueError(f"chunk_size must be positive, got {cs}")
        if co < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {co}")
        if co >= cs:
            raise ValueError(f"chunk_overlap ({co}) must be less than chunk_size ({cs})")
        if is_code:
            ext = Path(source).suffix.lower()
            language = _EXT_TO_LANG.get(ext, "")
            chunks = chunk_code(text, cs, co, language)
        else:
            chunks = chunk_text(text, cs, co)

    if not chunks:
        console.print(f"[yellow]⚠ No chunks generated from {source}[/yellow]")
        return IngestResult(status="empty", source=source)

    # --- Step 4: Embed (with progress bar — this is the slow part) ---
    embeddings = _embed_with_progress(chunks, cfg.embeddings.model)

    # --- Step 5: Store ---
    with console.status("[bold cyan]Storing...[/bold cyan]", spinner="dots"):
        doc_id = store.add_document(
            path=source_path,
            title=title,
            chunks=chunks,
            embeddings=embeddings,
            source_type=source_type,
            collection=collection,
            project=fm_project,
            layer=fm_layer,
            tags=fm_tags,
        )

    elapsed = round(time.time() - start_time, 2)

    return IngestResult(
        status="ok",
        doc_id=doc_id,
        source=source,
        title=title,
        chunks=len(chunks),
        chars=char_count,
        elapsed_s=elapsed,
    )


def _generate_title(text: str) -> str:
    """Generate a descriptive title from text content.

    Produces a slug from the first meaningful words plus a timestamp,
    e.g. ``"oauth2-uses-pkce-20260330-143052123"``.
    """
    from datetime import datetime, timezone

    preview = text.strip()[:60].lower()
    preview = re.sub(r"[^a-z0-9\s]", "", preview)
    words = preview.split()[:5]
    slug = "-".join(words) if words else "note"

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S%f")
    return f"{slug}-{ts}"


def ingest_text(
    text: str,
    cfg: VstashConfig,
    store: VstashStore,
    *,
    title: str | None = None,
    collection: str = "default",
    project: str | None = None,
    layer: str | None = None,
    tags: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestResult:
    """Ingest raw text directly — no file or URL needed.

    This is the agent-friendly ingestion path. Instead of writing text
    to a temp file and then ingesting, callers pass the text directly.

    Args:
        text: The content to ingest.
        title: Human-readable title for the document. When *None* (the
            default), a descriptive title is auto-generated from the first
            words of *text* plus a UTC timestamp.
        cfg: Vex configuration.
        store: Vector store instance.
        collection: Named collection to group this document.
        project: Project identifier.
        layer: Layer/category tag.
        tags: Comma-separated tags.

    Returns:
        IngestResult with status, chunk count, timing, etc.
    """
    start_time = time.time()

    if title is None:
        title = _generate_title(text) if text and text.strip() else "note"

    if not text or not text.strip():
        return IngestResult(status="empty", source=f"text://{title}")

    # Use a synthetic path so the store can deduplicate by title+collection
    source_path = f"text://{title}"
    char_count = len(text)

    # Extract frontmatter if present
    frontmatter = _extract_frontmatter(text)
    fm_project = project or frontmatter.get("project")
    fm_layer = layer or frontmatter.get("layer")
    if fm_project is not None:
        if isinstance(fm_project, (dict, list)):
            fm_project = None
        else:
            fm_project = str(fm_project)
    if fm_layer is not None:
        if isinstance(fm_layer, (dict, list)):
            fm_layer = None
        else:
            fm_layer = str(fm_layer)
    fm_tags_raw = tags or frontmatter.get("tags")
    fm_tags: str | None = None
    if fm_tags_raw:
        if isinstance(fm_tags_raw, list):
            fm_tags = ",".join(str(t) for t in fm_tags_raw)
        elif not isinstance(fm_tags_raw, (dict,)):
            fm_tags = str(fm_tags_raw)
    text = _strip_frontmatter(text)

    # Chunk
    cs = chunk_size if chunk_size is not None else cfg.chunking.size
    co = chunk_overlap if chunk_overlap is not None else cfg.chunking.overlap
    if cs <= 0:
        raise ValueError(f"chunk_size must be positive, got {cs}")
    if co < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {co}")
    if co >= cs:
        raise ValueError(f"chunk_overlap ({co}) must be less than chunk_size ({cs})")
    chunks = chunk_text(text, cs, co)
    if not chunks:
        return IngestResult(status="empty", source=source_path)

    # Embed
    embeddings = _embed_with_progress(chunks, cfg.embeddings.model)

    # Store
    doc_id = store.add_document(
        path=source_path,
        title=title,
        chunks=chunks,
        embeddings=embeddings,
        source_type="text",
        collection=collection,
        project=fm_project,
        layer=fm_layer,
        tags=fm_tags,
    )

    elapsed = round(time.time() - start_time, 2)

    return IngestResult(
        status="ok",
        doc_id=doc_id,
        source=source_path,
        title=title,
        chunks=len(chunks),
        chars=char_count,
        elapsed_s=elapsed,
    )


def _load_gitignore(directory: Path) -> list[str]:
    """Load top-level .gitignore patterns from directory.

    Only reads the root .gitignore — nested .gitignore files and global
    git config are not supported. Negation patterns (!) are ignored.
    """
    gitignore = directory / ".gitignore"
    if not gitignore.is_file():
        return []
    patterns: list[str] = []
    for line in gitignore.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("!"):
            patterns.append(line)
    return patterns


def _is_gitignored(file_path: Path, root: Path, patterns: list[str]) -> bool:
    """Check if a file matches any .gitignore pattern (simplified matching).

    Uses POSIX-normalized relative paths for cross-platform compatibility.
    """
    import fnmatch

    rel = file_path.relative_to(root).as_posix()
    for pattern in patterns:
        # Match against full relative path and filename
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(file_path.name, pattern):
            return True
        # Handle directory patterns like "dist/"
        if pattern.endswith("/"):
            dir_pattern = pattern.rstrip("/")
            if any(part == dir_pattern for part in file_path.relative_to(root).parts):
                return True
    return False


def _should_exclude_dir(name: str) -> bool:
    """Check if a directory name should be excluded from ingestion."""
    import fnmatch

    if name.startswith("."):
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDED_DIRS)


def _collect_files(directory: Path) -> list[Path]:
    """Collect files for ingestion using os.walk with directory pruning.

    Prunes excluded directories in-place so they are never descended into,
    which is significantly faster than rglob + post-filtering for large trees.
    Applies top-level .gitignore patterns (simplified matching, no negation).
    """
    import os

    gitignore_patterns = _load_gitignore(directory)

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(directory):
        # Prune excluded directories in-place — os.walk won't descend into them
        dirnames[:] = [d for d in dirnames if not _should_exclude_dir(d)]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if gitignore_patterns and _is_gitignored(fpath, directory, gitignore_patterns):
                continue
            files.append(fpath)
    return files


def ingest_directory(
    directory: str,
    cfg: VstashConfig,
    store: VstashStore,
    *,
    force: bool = False,
    collection: str = "default",
    project: str | None = None,
    layer: str | None = None,
    tags: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[IngestResult]:
    """Recursively ingest all supported files in a directory.

    Respects .gitignore patterns and excludes common non-content directories
    (__pycache__, node_modules, .venv, dist, build, etc.).

    Safety limits: raises ValueError if the directory exceeds MAX_DIR_FILES
    or MAX_DIR_BYTES to prevent accidental resource exhaustion.

    Args:
        directory: Path to directory to scan.
        cfg: Vex configuration.
        store: Vector store instance.
        force: If False, skip documents already in the store.
        collection: Named collection to group ingested documents.
        project: Project tag to apply to all files (overrides frontmatter).
        layer: Layer tag to apply to all files (overrides frontmatter).
        tags: Comma-separated tags to apply to all files (overrides frontmatter).

    Returns:
        List of IngestResult for each processed file.

    Raises:
        ValueError: If file count or total size exceeds safety limits.
    """
    path = Path(directory)
    files = _collect_files(path)

    if not files:
        console.print(f"[yellow]No supported files found in {directory}[/yellow]")
        return []

    # Safety limits
    if len(files) > MAX_DIR_FILES:
        raise ValueError(
            f"Directory has {len(files)} files (limit: {MAX_DIR_FILES}). "
            f"Use a subdirectory or ingest files individually."
        )

    total_bytes = sum(f.stat().st_size for f in files)
    if total_bytes > MAX_DIR_BYTES:
        mb = total_bytes / (1024 * 1024)
        limit_mb = MAX_DIR_BYTES / (1024 * 1024)
        raise ValueError(
            f"Directory is {mb:.0f} MB (limit: {limit_mb:.0f} MB). "
            f"Use a subdirectory or ingest files individually."
        )

    console.print(
        f"[dim]Found {len(files)} files ({total_bytes / 1024:.0f} KB) in {directory}[/dim]"
    )

    results: list[IngestResult] = []
    with (
        store.batch_mode(),
        Progress(
            SpinnerColumn(),
            TextColumn("[bold]Processing directory[/bold]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress,
    ):
        task = progress.add_task("", total=len(files))
        for f in files:
            progress.update(task, description=f.name)
            result = ingest(
                str(f),
                cfg,
                store,
                force=force,
                collection=collection,
                project=project,
                layer=layer,
                tags=tags,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            results.append(result)
            progress.advance(task)

    return results


# ------------------------------------------------------------------ #
# Frontmatter                                                         #
# ------------------------------------------------------------------ #

_FRONTMATTER_RE = re.compile(
    r"\A\s*---[ \t]*\r?\n(.*?\n)---[ \t]*\r?(?:\n|\Z)",
    re.DOTALL,
)


def _extract_frontmatter(text: str) -> dict[str, Any]:
    """Extract YAML frontmatter from document text.

    Parses the YAML block between ``---`` delimiters at the start of the
    document. Only simple key-value pairs and lists are supported.

    Args:
        text: Raw document text.

    Returns:
        Dictionary of parsed frontmatter fields, or empty dict if none found.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}

    try:
        import yaml  # noqa: PLC0415 — optional dependency, lazy import

        data = yaml.safe_load(match.group(1))
        if not isinstance(data, dict):
            # Non-mapping frontmatter (list, scalar) is treated as absent
            return {}
        return dict(data)
    except ImportError:
        # Fallback: parse simple `key: value` lines without pyyaml
        result: dict[str, Any] = {}
        for line in match.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            value = value.strip()
            # Handle YAML lists inline: [a, b, c]
            if value.startswith("[") and value.endswith("]"):
                import csv  # noqa: PLC0415
                import io  # noqa: PLC0415

                s = io.StringIO(value[1:-1])
                try:
                    items = next(csv.reader(s, skipinitialspace=True))
                    result[key.strip()] = [item for item in items if item]
                except StopIteration:
                    result[key.strip()] = []
            else:
                result[key.strip()] = value.strip("'\"")
        return result


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter from document text.

    Args:
        text: Raw document text potentially starting with ``---`` block.

    Returns:
        Text with frontmatter block removed.
    """
    return _FRONTMATTER_RE.sub("", text, count=1)


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


# Maximum download size for URL ingestion (50 MB default)
_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


def _validate_url(url: str) -> None:
    """Validate that a URL does not target internal/private network addresses.

    Prevents SSRF attacks by blocking loopback, link-local, and RFC 1918 addresses.

    Args:
        url: The URL to validate.

    Raises:
        ValueError: If the URL resolves to a private or restricted address.
    """
    import ipaddress
    import socket

    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Invalid URL (no hostname): {url}")

    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve hostname '{hostname}': {exc}") from exc

    for _family, _type, _proto, _canonname, sockaddr in addr_info:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(
                f"URL '{url}' resolves to private/reserved address {ip}. "
                "Ingesting from internal networks is not allowed."
            )


def _read_url_content(url: str, max_bytes: int = _MAX_DOWNLOAD_BYTES) -> bytes:
    """Download URL content with a size limit and SSRF-safe redirect handling.

    Each redirect hop is re-validated against private/reserved IPs to prevent
    SSRF via open redirects.

    Args:
        url: The URL to fetch (must already be validated via ``_validate_url``).
        max_bytes: Maximum number of bytes to download.

    Returns:
        Downloaded content as bytes.

    Raises:
        ValueError: If the response exceeds the size limit or a redirect
            targets a private address.
    """
    import urllib.request

    class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
        """Re-validate every redirect target to prevent SSRF via open redirects."""

        def redirect_request(self, req, fp, code, msg, headers, newurl):
            new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
            if new_req is None:
                return None
            _validate_url(new_req.full_url)
            return new_req

    headers = {"User-Agent": "vstash (https://github.com/stffns/vstash)"}
    req = urllib.request.Request(url, headers=headers)
    opener = urllib.request.build_opener(_SafeRedirectHandler())

    with opener.open(req, timeout=30) as response:
        # Check Content-Length header first (if available)
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError(
                f"Response too large ({int(content_length)} bytes, max {max_bytes} bytes)"
            )

        # Read in chunks with size enforcement
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(
                    f"Response exceeded max size ({max_bytes} bytes). Download aborted."
                )
            chunks.append(chunk)

    return b"".join(chunks)


def _parse(source: str) -> str:
    """Parse file or URL to plain text using markitdown.

    For URLs, validates the target address (preventing SSRF), downloads
    content with a size limit, and converts to text.

    Args:
        source: File path or URL to parse.

    Returns:
        Cleaned plain text content.

    Raises:
        ValueError: If the URL targets a private address or exceeds size limits.
        Exception: If markitdown fails to parse the document.
    """
    import tempfile

    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise ImportError(
            "File ingestion requires markitdown. Install it with: pip install vstash[ingest]"
        ) from exc

    md = MarkItDown()

    if _is_url(source):
        _validate_url(source)
        content = _read_url_content(source)

        suffix = Path(urlparse(source).path).suffix or ".html"
        # Sanitize suffix to prevent path traversal
        suffix = "." + suffix.lstrip(".").split("/")[-1].split("\\")[-1]
        if not suffix or suffix == ".":
            suffix = ".html"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            result = md.convert(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        result = md.convert(source)

    # Clean excessive whitespace while preserving structure
    text: str = re.sub(r"\n{3,}", "\n\n", result.text_content)
    return text.strip()


def _embed_with_progress(chunks: list[str], model_name: str) -> list[list[float]]:
    """Embed chunks with a Rich progress bar. Batches for efficiency.

    Args:
        chunks: Text chunks to embed.
        model_name: FastEmbed model identifier.

    Returns:
        List of embedding vectors.
    """
    BATCH_SIZE = 64
    all_embeddings: list[list[float]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Embedding[/bold cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[rate]}[/dim]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("", total=len(chunks), rate="")
        t0 = time.time()
        processed = 0

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            embeddings = embed_texts(batch, model_name)
            all_embeddings.extend(embeddings)
            processed += len(batch)

            elapsed = time.time() - t0
            rate = f"{processed / elapsed:.0f} chunks/s" if elapsed > 0.1 else ""
            progress.update(task, advance=len(batch), rate=rate)

    return all_embeddings
