"""Single source of truth for opening a VstashStore from VstashConfig.

Prior to this helper, every public surface (CLI, MCP, web, SDK, journal,
federated search) duplicated the StorageConfig -> VstashStore wiring.
Adding a new storage knob meant touching 7+ files, and each new surface
risked silently dropping IVFPQ / observability / limits / cache fields
(see issue #297).

``open_store_for_config`` centralizes the wiring. Callers pass a
``VstashConfig`` plus optional db_path / embedding_dim overrides.
"""

from __future__ import annotations

from .config import VstashConfig
from .embed import get_embedding_dim
from .store import VstashStore


def open_store_for_config(
    cfg: VstashConfig,
    *,
    db_path: str | None = None,
    profile: str | None = None,
    embedding_dim: int | None = None,
) -> VstashStore:
    """Open a VstashStore wired from a VstashConfig.

    Args:
        cfg: Loaded ``VstashConfig``. Provides every ``StorageConfig``
            knob plus observability, limits, and cache.
        db_path: Optional explicit database path. When omitted, resolved
            via the standard profile chain (``--profile``, env, project-
            local, default). Used by ``--store path:alias`` overrides
            and by per-profile stores in federated search.
        profile: Named profile to resolve. Ignored when ``db_path`` is
            given.
        embedding_dim: Optional embedding dimension override. Used by
            ``vstash reindex`` to open the store with the *current*
            model's dim while preparing to migrate to a new dim.
            Defaults to ``get_embedding_dim(cfg.embeddings.model)``.

    Returns:
        A ``VstashStore`` with every storage knob, observability,
        limits, and cache forwarded from ``cfg``.
    """
    if db_path is None:
        from .profile import resolve_db_path

        db_path = str(resolve_db_path(profile, config_db_path=cfg.storage.db_path))

    if embedding_dim is None:
        embedding_dim = get_embedding_dim(cfg.embeddings.model)

    return VstashStore(
        db_path,
        embedding_dim=embedding_dim,
        vector_backend=cfg.storage.vector_backend,
        snapvec_bits=cfg.storage.snapvec_bits,
        ivfpq_nlist=cfg.storage.ivfpq_nlist,
        ivfpq_M=cfg.storage.ivfpq_M,
        ivfpq_K=cfg.storage.ivfpq_K,
        ivfpq_rerank_candidates=cfg.storage.ivfpq_rerank_candidates,
        ivfpq_nprobe=cfg.storage.ivfpq_nprobe,
        observability=cfg.observability,
        limits=cfg.limits,
        cache=cfg.cache,
    )
