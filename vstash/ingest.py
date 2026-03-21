"""
ingest.py — Document ingestion pipeline.

Flow: file/URL → markitdown → chunk → FastEmbed → sqlite-vec + FTS5

Chunking: fixed-size token windows with overlap.
Progress: Rich bars at every stage — never looks frozen.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import tiktoken
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .config import VstashConfig
from .embed import embed_texts
from .models import IngestResult
from .store import VstashStore

console = Console(stderr=True)

# Tiktoken encoder — fast, no network call
_enc = tiktoken.get_encoding("cl100k_base")


# ------------------------------------------------------------------ #
# Chunking                                                            #
# ------------------------------------------------------------------ #


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into token-bounded chunks with overlap.

    Tries to break at sentence boundaries when possible.

    Args:
        text: Raw text to chunk.
        chunk_size: Maximum tokens per chunk.
        overlap: Token overlap between consecutive chunks.

    Returns:
        List of non-empty text chunks.
    """
    tokens = _enc.encode(text)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = _enc.decode(chunk_tokens).strip()

        # Only add non-empty, non-whitespace chunks
        if chunk_text_str and len(chunk_text_str) > 20:
            chunks.append(chunk_text_str)

        if end >= len(tokens):
            break

        # Slide forward by (chunk_size - overlap)
        start += chunk_size - overlap

    return chunks


# ------------------------------------------------------------------ #
# Source detection                                                    #
# ------------------------------------------------------------------ #


def _is_url(source: str) -> bool:
    """Check if a source string is an HTTP(S) URL."""
    try:
        result = urlparse(source)
        return result.scheme in ("http", "https")
    except Exception:
        return False


def _get_title(source: str) -> str:
    """Derive a human-readable title from a file path or URL."""
    if _is_url(source):
        return source
    path = Path(source)
    return path.stem.replace("_", " ").replace("-", " ").title()


def _get_source_type(source: str) -> str:
    """Determine the document type from a file extension or URL."""
    if _is_url(source):
        return "url"
    suffix = Path(source).suffix.lower()
    type_map: dict[str, str] = {
        ".pdf": "pdf", ".docx": "docx", ".pptx": "pptx",
        ".xlsx": "xlsx", ".md": "markdown", ".txt": "text",
        ".py": "code", ".js": "code", ".ts": "code",
        ".go": "code", ".rs": "code", ".java": "code",
        ".html": "html", ".htm": "html",
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

    # Check for duplicates unless force is set
    if not force and store.doc_exists(source_path):
        return IngestResult(
            status="skipped",
            source=source,
            title=title,
        )

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

    char_count = len(text)

    # --- Step 2: Extract frontmatter metadata ---
    frontmatter = _extract_frontmatter(text)
    # Explicit params override frontmatter values
    fm_project = project or frontmatter.get("project")
    fm_layer = layer or frontmatter.get("layer")
    # Coerce non-string scalars to str; ignore dicts/lists
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
    # Strip frontmatter block from text before chunking
    text = _strip_frontmatter(text)

    # --- Step 3: Chunk ---
    with console.status("[bold cyan]Chunking...[/bold cyan]", spinner="dots"):
        chunks = chunk_text(text, cfg.chunking.size, cfg.chunking.overlap)

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
) -> list[IngestResult]:
    """Recursively ingest all supported files in a directory.

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
    """
    SUPPORTED: set[str] = {
        ".pdf", ".docx", ".pptx", ".xlsx", ".md", ".txt",
        ".py", ".js", ".ts", ".go", ".rs", ".java",
        ".html", ".htm", ".csv",
    }
    path = Path(directory)
    files = [
        f for f in path.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED
        and not any(part.startswith(".") for part in f.parts)  # skip hidden
    ]

    if not files:
        console.print(f"[yellow]No supported files found in {directory}[/yellow]")
        return []

    results: list[IngestResult] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]Processing directory[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("", total=len(files))
        for f in files:
            progress.update(task, description=f.name)
            result = ingest(
                str(f), cfg, store,
                force=force, collection=collection,
                project=project, layer=layer, tags=tags,
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


def _parse(source: str) -> str:
    """Parse file or URL to plain text using markitdown.

    For URLs, downloads content first with a proper User-Agent header
    to avoid 403 errors from sites like Wikipedia.

    Args:
        source: File path or URL to parse.

    Returns:
        Cleaned plain text content.

    Raises:
        Exception: If markitdown fails to parse the document.
    """
    import tempfile

    from markitdown import MarkItDown

    md = MarkItDown()

    # URLs need a proper User-Agent or many sites (Wikipedia, etc.) return 403
    if _is_url(source):
        import urllib.request

        headers = {"User-Agent": "vstash (https://github.com/stffns/vstash)"}
        req = urllib.request.Request(source, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
            # Write to temp file so markitdown can detect the format
            suffix = Path(urlparse(source).path).suffix or ".html"
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
            batch = chunks[i:i + BATCH_SIZE]
            embeddings = embed_texts(batch, model_name)
            all_embeddings.extend(embeddings)
            processed += len(batch)

            elapsed = time.time() - t0
            rate = f"{processed / elapsed:.0f} chunks/s" if elapsed > 0.1 else ""
            progress.update(task, advance=len(batch), rate=rate)

    return all_embeddings
