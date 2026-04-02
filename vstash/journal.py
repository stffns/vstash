"""
journal.py — Cross-session memory for agents and humans.

Provides a journaling system that automatically saves and recalls
context across sessions. Designed for:
  - Claude Code hooks (PreCompact, SessionEnd, SessionStart)
  - Python agents via SDK (Memory.journal_*)
  - Claude Desktop via MCP (vstash_journal_*)

Core operations:
  save    — store a journal entry with auto-timestamp and tags
  recall  — retrieve recent/relevant entries for context injection
  log     — chronological view of journal entries
  prune   — remove old entries by age
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import VstashConfig, load_config
from .embed import embed_query, get_embedding_dim
from .profile import PROFILES_DIR
from .store import VstashStore

logger = logging.getLogger(__name__)

# Default journal profile and layer
JOURNAL_PROFILE = "journal"
JOURNAL_LAYER = "journal"
JOURNAL_COLLECTION = "default"


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _auto_title(text: str) -> str:
    """Generate a journal entry title from timestamp + first words."""
    ts = _now_utc().strftime("%Y-%m-%d %H:%M")
    # First meaningful line, truncated
    first_line = ""
    for line in text.strip().splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            first_line = stripped[:60]
            break
    if first_line:
        return f"journal {ts} — {first_line}"
    return f"journal {ts}"


def _get_journal_store(cfg: VstashConfig | None = None) -> tuple[VstashConfig, VstashStore]:
    """Open the journal profile's store."""
    if cfg is None:
        cfg = load_config()

    db_path = PROFILES_DIR / JOURNAL_PROFILE / "memory.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    dim = get_embedding_dim(cfg.embeddings.model)
    store = VstashStore(
        str(db_path),
        embedding_dim=dim,
        vector_backend=cfg.storage.vector_backend,
        snapvec_bits=cfg.storage.snapvec_bits,
    )
    return cfg, store


def _parse_age(age_str: str) -> timedelta:
    """Parse age string like '30d', '2w', '24h' into timedelta."""
    match = re.match(r"^(\d+)([dhw])$", age_str.strip())
    if not match:
        raise ValueError(
            f"Invalid age format {age_str!r}. Use <number><unit> where unit is d(ays), h(ours), or w(eeks)."
        )
    value = int(match.group(1))
    unit = match.group(2)
    if unit == "d":
        return timedelta(days=value)
    elif unit == "h":
        return timedelta(hours=value)
    elif unit == "w":
        return timedelta(weeks=value)
    raise ValueError(f"Unknown unit: {unit}")  # pragma: no cover


# ------------------------------------------------------------------ #
# Core operations                                                      #
# ------------------------------------------------------------------ #


def journal_save(
    text: str,
    *,
    title: str | None = None,
    project: str | None = None,
    tags: str | None = None,
    source: str | None = None,
    cfg: VstashConfig | None = None,
) -> dict:
    """Save a journal entry.

    Uses the ingest pipeline for long text, but stores short entries
    directly (bypassing the 80-token minimum chunk filter) so that
    brief notes and decisions are never silently dropped.

    Args:
        text: Content to save.
        title: Optional title (auto-generated if None).
        project: Project tag (auto-detected from cwd if None).
        tags: Comma-separated tags (auto-adds 'journal' tag).
        source: Source identifier (e.g. 'session', 'precompact', 'hook').
        cfg: Optional pre-loaded config.

    Returns:
        Dict with entry metadata (title, chunks, timestamp).
    """
    if not text or not text.strip():
        return {"status": "empty", "message": "No content to save."}

    if title is None:
        title = _auto_title(text)

    # Auto-detect project from cwd
    if project is None:
        project = _detect_project()

    # Build tags — always include 'journal' and source
    tag_parts = ["journal"]
    if source:
        tag_parts.append(source)
    if tags:
        tag_parts.extend(t.strip() for t in tags.split(",") if t.strip())
    final_tags = ",".join(sorted(set(tag_parts)))

    cfg, store = _get_journal_store(cfg)
    try:
        source_path = f"text://{title}"

        # Try ingest pipeline first (handles chunking for long text)
        from .ingest import ingest_text

        result = ingest_text(
            text,
            cfg,
            store,
            title=title,
            collection=JOURNAL_COLLECTION,
            project=project,
            layer=JOURNAL_LAYER,
            tags=final_tags,
        )

        # If ingest returned "empty" (text too short for chunking),
        # store as a single chunk directly — journal entries can be brief.
        if result.status == "empty" and text.strip():
            from .embed import embed_texts

            embeddings = embed_texts([text.strip()], cfg.embeddings.model)
            store.add_document(
                path=source_path,
                title=title,
                chunks=[text.strip()],
                embeddings=embeddings,
                source_type="text",
                collection=JOURNAL_COLLECTION,
                project=project,
                layer=JOURNAL_LAYER,
                tags=final_tags,
            )
            return {
                "status": "ok",
                "title": title,
                "chunks": 1,
                "project": project,
                "tags": final_tags,
                "timestamp": _now_utc().isoformat(),
            }

        return {
            "status": result.status,
            "title": title,
            "chunks": result.chunks,
            "project": project,
            "tags": final_tags,
            "timestamp": _now_utc().isoformat(),
        }
    finally:
        store.close()


def journal_recall(
    query: str | None = None,
    *,
    top_k: int = 5,
    project: str | None = None,
    cfg: VstashConfig | None = None,
) -> list[dict]:
    """Recall relevant journal entries.

    When query is None, returns the most recent entries.
    When query is provided, performs semantic search.

    Args:
        query: Optional search query. None = recent entries.
        top_k: Number of entries to return.
        project: Filter by project (auto-detected if None).
        cfg: Optional pre-loaded config.

    Returns:
        List of dicts with text, title, score, timestamp.
    """
    cfg, store = _get_journal_store(cfg)
    try:
        if query:
            # Semantic search
            q_embedding = embed_query(query, cfg.embeddings.model)
            results = store.search(
                query_embedding=q_embedding,
                query_text=query,
                top_k=top_k,
                collection=JOURNAL_COLLECTION,
                project=project,
                layer=JOURNAL_LAYER,
            )
            return [
                {
                    "text": r.text,
                    "title": r.title,
                    "score": r.score,
                }
                for r in results
            ]
        else:
            # Recent entries — list documents sorted by newest first
            docs = store.list_documents(
                collection=JOURNAL_COLLECTION,
                project=project,
                layer=JOURNAL_LAYER,
            )
            entries = []
            for doc in docs[:top_k]:
                # Fetch chunk text for each doc
                chunks = store.get_document_chunks(doc.path)
                full_text = "\n".join(chunks) if chunks else ""
                entries.append(
                    {
                        "text": full_text,
                        "title": doc.title,
                        "added_at": getattr(doc, "added_at", None),
                        "tags": doc.tags,
                    }
                )
            return entries
    finally:
        store.close()


def journal_log(
    *,
    limit: int = 20,
    project: str | None = None,
    cfg: VstashConfig | None = None,
) -> list[dict]:
    """Chronological view of journal entries (newest first).

    Args:
        limit: Max number of entries to return.
        project: Filter by project.
        cfg: Optional pre-loaded config.

    Returns:
        List of dicts with title, project, tags, chunk_count, added_at.
    """
    cfg, store = _get_journal_store(cfg)
    try:
        docs = store.list_documents(
            collection=JOURNAL_COLLECTION,
            project=project,
            layer=JOURNAL_LAYER,
        )
        return [
            {
                "title": doc.title,
                "project": doc.project,
                "tags": doc.tags,
                "chunks": doc.chunk_count,
                "chars": doc.char_count,
                "added_at": getattr(doc, "added_at", None),
            }
            for doc in docs[:limit]
        ]
    finally:
        store.close()


def journal_prune(
    age: str,
    *,
    project: str | None = None,
    dry_run: bool = False,
    cfg: VstashConfig | None = None,
) -> dict:
    """Remove journal entries older than the specified age.

    Args:
        age: Age threshold like '30d', '2w', '24h'.
        project: Only prune entries for this project.
        dry_run: If True, report what would be deleted without deleting.
        cfg: Optional pre-loaded config.

    Returns:
        Dict with count of deleted entries and their titles.
    """
    delta = _parse_age(age)
    cutoff = _now_utc() - delta

    cfg, store = _get_journal_store(cfg)
    try:
        docs = store.list_documents(
            collection=JOURNAL_COLLECTION,
            project=project,
            layer=JOURNAL_LAYER,
        )

        to_delete = []
        for doc in docs:
            added_at = getattr(doc, "added_at", None)
            if added_at is None:
                continue
            # Parse the added_at timestamp
            try:
                if isinstance(added_at, str):
                    doc_time = datetime.fromisoformat(added_at.replace("Z", "+00:00"))
                else:
                    doc_time = added_at
                # Make timezone-aware if naive
                if doc_time.tzinfo is None:
                    doc_time = doc_time.replace(tzinfo=timezone.utc)
                if doc_time < cutoff:
                    to_delete.append(doc)
            except (ValueError, TypeError):
                continue

        if dry_run:
            return {
                "status": "dry_run",
                "would_delete": len(to_delete),
                "entries": [d.title for d in to_delete],
                "cutoff": cutoff.isoformat(),
            }

        deleted = 0
        for doc in to_delete:
            if store.delete_document(doc.path):
                deleted += 1

        return {
            "status": "ok",
            "deleted": deleted,
            "entries": [d.title for d in to_delete],
            "cutoff": cutoff.isoformat(),
        }
    finally:
        store.close()


def parse_transcript(transcript_path: str) -> str:
    """Extract key information from a Claude Code transcript JSONL.

    Reads the transcript file and extracts:
    - User requests (summarized)
    - Key decisions made
    - Files modified
    - Errors encountered

    Args:
        transcript_path: Path to the .jsonl transcript file.

    Returns:
        Summarized text suitable for journal_save().
    """
    path = Path(transcript_path)
    if not path.exists():
        return ""

    lines = path.read_text().strip().splitlines()
    if not lines:
        return ""

    user_messages = []
    files_modified = set()
    errors = []

    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        role = entry.get("role", "")

        # Collect user messages
        if role == "user":
            content = entry.get("content", "")
            if isinstance(content, str) and content.strip():
                user_messages.append(content.strip()[:200])
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            user_messages.append(text[:200])

        # Collect file edits from tool use
        if role == "assistant":
            content = entry.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_input = block.get("input", {})
                        fp = tool_input.get("file_path") or tool_input.get("path")
                        if fp:
                            files_modified.add(fp)

        # Collect errors from tool results
        if role == "tool":
            content = entry.get("content", "")
            if isinstance(content, str) and "error" in content.lower()[:50]:
                errors.append(content[:150])

    # Build summary
    parts = []
    if user_messages:
        parts.append("## User requests")
        # Take first and last few to keep it compact
        shown = user_messages[:3]
        if len(user_messages) > 6:
            shown.append("...")
            shown.extend(user_messages[-2:])
        elif len(user_messages) > 3:
            shown.extend(user_messages[3:])
        for msg in shown:
            parts.append(f"- {msg}")

    if files_modified:
        parts.append("\n## Files modified")
        for fp in sorted(files_modified)[:20]:
            parts.append(f"- {fp}")

    if errors:
        parts.append("\n## Errors encountered")
        for err in errors[:5]:
            parts.append(f"- {err}")

    return "\n".join(parts) if parts else ""


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _detect_project() -> str | None:
    """Auto-detect project name from current directory."""
    try:
        cwd = Path.cwd()
        # Use the directory name as project
        return cwd.name
    except OSError:
        return None
