"""
profile.py — Multi-profile management for vstash.

Profiles are isolated SQLite databases under ~/.vstash/profiles/<name>/memory.db.
Resolution order (highest priority first):
  1. VSTASH_DB_PATH env (explicit override)
  2. Explicit profile name (--profile flag)
  3. VSTASH_PROFILE env var
  4. .vstash/memory.db in cwd or parent directories (project-local)
  5. ~/.vstash/memory.db (global default)
"""

from __future__ import annotations

import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .models import SearchResult

VSTASH_HOME = Path.home() / ".vstash"
PROFILES_DIR = VSTASH_HOME / "profiles"
DEFAULT_DB = VSTASH_HOME / "memory.db"

_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def validate_name(name: str) -> str:
    """Validate a profile name.

    Args:
        name: Profile name to validate.

    Returns:
        The validated name.

    Raises:
        ValueError: If the name is invalid.
    """
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid profile name {name!r}. "
            "Must be 1-64 chars, start with alphanumeric, "
            "and contain only letters, digits, hyphens, or underscores."
        )
    return name


def _find_local_db() -> Path | None:
    """Walk from cwd upward looking for .vstash/memory.db."""
    current = Path.cwd().resolve()
    while True:
        candidate = current / ".vstash" / "memory.db"
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def resolve_db_path(profile: str | None = None) -> Path:
    """Resolve the database path using the layered resolution chain.

    Priority:
      1. VSTASH_DB_PATH env var (explicit override, always wins)
      2. Explicit profile name
      3. VSTASH_PROFILE env var
      4. .vstash/memory.db walking cwd upward
      5. ~/.vstash/memory.db (default)

    Args:
        profile: Explicit profile name (e.g. from --profile flag).

    Returns:
        Absolute Path to the resolved memory.db file.
    """
    # 1. VSTASH_DB_PATH always wins
    env_db = os.getenv("VSTASH_DB_PATH")
    if env_db:
        return Path(env_db).expanduser().resolve()

    # 2. Explicit profile name
    if profile:
        validate_name(profile)
        return PROFILES_DIR / profile / "memory.db"

    # 3. VSTASH_PROFILE env var
    env_profile = os.getenv("VSTASH_PROFILE")
    if env_profile:
        validate_name(env_profile)
        return PROFILES_DIR / env_profile / "memory.db"

    # 4. Project-local .vstash/memory.db
    local_db = _find_local_db()
    if local_db is not None:
        return local_db

    # 5. Global default
    return DEFAULT_DB


def list_profiles() -> list[str]:
    """List all named profiles.

    Returns:
        Sorted list of profile names that have a memory.db file.
    """
    if not PROFILES_DIR.is_dir():
        return []
    return sorted(
        d.name for d in PROFILES_DIR.iterdir() if d.is_dir() and (d / "memory.db").exists()
    )


def create_profile(name: str) -> Path:
    """Create a new named profile.

    Args:
        name: Profile name.

    Returns:
        Path to the new memory.db file.

    Raises:
        ValueError: If name is invalid or profile already exists.
    """
    validate_name(name)
    profile_dir = PROFILES_DIR / name
    if profile_dir.exists():
        raise ValueError(f"Profile {name!r} already exists.")
    profile_dir.mkdir(parents=True)
    db_path = profile_dir / "memory.db"
    # Touch the db file so profile list detects it immediately
    db_path.touch()
    return db_path


def delete_profile(name: str) -> None:
    """Delete a named profile and all its data.

    Args:
        name: Profile name.

    Raises:
        ValueError: If name is invalid or profile does not exist.
    """
    validate_name(name)
    profile_dir = PROFILES_DIR / name
    if not profile_dir.exists():
        raise ValueError(f"Profile {name!r} does not exist.")
    shutil.rmtree(profile_dir)


def active_profile_info(explicit: str | None = None) -> tuple[str | None, str]:
    """Determine which profile is active and why.

    Args:
        explicit: Profile name from --profile flag.

    Returns:
        Tuple of (profile_name_or_None, reason_string).
    """
    env_db = os.getenv("VSTASH_DB_PATH")
    if env_db:
        return None, f"VSTASH_DB_PATH={env_db}"

    if explicit:
        return explicit, "--profile flag"

    env_profile = os.getenv("VSTASH_PROFILE")
    if env_profile:
        return env_profile, "VSTASH_PROFILE env var"

    local_db = _find_local_db()
    if local_db is not None:
        return None, f"project-local {local_db}"

    return None, f"default {DEFAULT_DB}"


def federated_search(
    query_embedding: list[float],
    query_text: str,
    *,
    embedding_dim: int,
    vector_backend: str = "sqlite-vec",
    snapvec_bits: int = 4,
    top_k: int = 5,
    collection: str | None = None,
    project: str | None = None,
    layer: str | None = None,
    scoring: object | None = None,
) -> list[tuple[str, SearchResult]]:
    """Search across all profiles and merge results with RRF.

    Opens each profile's VstashStore, runs the query, and merges
    results using Reciprocal Rank Fusion (k=60).

    Args:
        query_embedding: Pre-computed query embedding.
        query_text: Raw query text for FTS.
        embedding_dim: Embedding dimension for stores.
        vector_backend: Vector backend name.
        snapvec_bits: Quantization bits for snapvec.
        top_k: Number of results to return.
        collection: Optional collection filter.
        project: Optional project filter.
        layer: Optional layer filter.
        scoring: Optional ScoringConfig.

    Returns:
        List of (profile_name, SearchResult) tuples sorted by merged score.
        Profile name is "default" for the global database.
    """
    from .store import VstashStore

    # Collect all profile db paths
    profiles: list[tuple[str, Path]] = []

    # Default DB
    if DEFAULT_DB.exists():
        profiles.append(("default", DEFAULT_DB))

    # Named profiles
    for name in list_profiles():
        db_path = PROFILES_DIR / name / "memory.db"
        profiles.append((name, db_path))

    if not profiles:
        return []

    def _search_one(item: tuple[str, Path]) -> list[tuple[str, SearchResult]]:
        name, db_path = item
        try:
            store = VstashStore(
                str(db_path),
                embedding_dim=embedding_dim,
                vector_backend=vector_backend,
                snapvec_bits=snapvec_bits,
            )
            try:
                results = store.search(
                    query_embedding=query_embedding,
                    query_text=query_text,
                    top_k=top_k,
                    collection=collection,
                    project=project,
                    layer=layer,
                    scoring=scoring,
                )
                return [(name, r) for r in results]
            finally:
                store.close()
        except Exception:
            return []

    # Run searches in parallel
    max_workers = min(len(profiles), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        all_results_nested = list(pool.map(_search_one, profiles))

    # Flatten
    all_tagged: list[tuple[str, SearchResult]] = []
    for batch in all_results_nested:
        all_tagged.extend(batch)

    if not all_tagged:
        return []

    # RRF merge: group by (path, chunk) to handle same doc in multiple profiles
    k = 60
    rrf_scores: dict[tuple[str, str, int], float] = {}
    result_map: dict[tuple[str, str, int], tuple[str, SearchResult]] = {}

    # Build per-profile ranked lists and compute RRF
    from itertools import groupby

    # Sort by profile name to group
    sorted_tagged = sorted(all_tagged, key=lambda x: x[0])
    for profile_name, group_iter in groupby(sorted_tagged, key=lambda x: x[0]):
        group_list = list(group_iter)
        # Results are already ranked by score within each profile
        for rank, (pname, result) in enumerate(group_list):
            key = (pname, result.path, result.chunk)
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in result_map:
                result_map[key] = (pname, result)

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)  # type: ignore[arg-type]
    merged = [result_map[key] for key in sorted_keys[:top_k]]

    return merged
