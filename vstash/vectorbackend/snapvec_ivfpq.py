"""IVFPQ backend adapter for VstashStore.

Wraps snapvec.IVFPQSnapIndex so it can slot into the same hot-path that
the flat SnapIndex uses. The key adaptation is that IVFPQ requires an
explicit `fit()` call on a training sample before the first `add_batch`;
the adapter tracks fitted state and forwards a consistent API.

Lifecycle:
    1. Store is created with vector_backend="snapvec-ivfpq". The .snpi
       file does not exist yet, so IVFPQBackend(fitted=False). sqlite-vec
       receives every chunk as usual (dual-population).
    2. Search falls back to sqlite-vec while len(backend) == 0 (matches
       the existing `elif self._snap is not None and len(self._snap) > 0`
       branch in VstashStore.search).
    3. User runs `vstash snapvec fit`. The CLI reads all embeddings from
       vec_chunks, calls backend.fit(sample) + backend.add_batch(all),
       backend.save(path).
    4. On next store open, IVFPQBackend.load() reads the .snpi and the
       presence of the file is authoritative for fitted state.
    5. Post-fit add_document calls are forwarded to the fitted index.
    6. Deletes forward through in all states (no-op before fit).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class IVFPQBackend:
    """Thin adapter around snapvec.IVFPQSnapIndex with pre-fit safe API.

    Attributes mirror SnapIndex where the search hot-path reads them,
    so VstashStore does not need a type switch on the snap object.

    The underlying IVFPQSnapIndex is built with ``normalized=True``
    (cosine scoring assumes unit-length inputs), so this adapter
    L2-normalizes at every entry point — fit, add_batch, and search.
    """

    def __init__(
        self,
        dim: int,
        *,
        nlist: int,
        M: int = 96,
        K: int = 256,
        seed: int = 0,
        keep_full_precision: bool = True,
        rerank_candidates: int = 100,
        nprobe: int = 0,
    ) -> None:
        from snapvec import IVFPQSnapIndex

        if dim % M != 0:
            raise ValueError(
                f"ivfpq_M={M} must divide embedding_dim={dim}. "
                f"Pick a divisor (e.g. 96 for dim=384, 128 for dim=1024)."
            )
        self.dim = dim
        self._nlist = nlist
        self._M = M
        self._K = K
        self._seed = seed
        self._keep_full_precision = keep_full_precision
        self._rerank_candidates = rerank_candidates
        self._nprobe = nprobe
        self._index = IVFPQSnapIndex(
            dim=dim,
            nlist=nlist,
            M=M,
            K=K,
            seed=seed,
            normalized=True,
            use_rht=False,
            keep_full_precision=keep_full_precision,
        )
        self._fitted: bool = False

    # ------------------------------------------------------------------ #
    # state                                                               #
    # ------------------------------------------------------------------ #

    @property
    def fitted(self) -> bool:
        return self._fitted

    def __len__(self) -> int:
        # Before fit, the index has no routable vectors — report 0 so the
        # store's search code falls through to sqlite-vec.
        return len(self._index) if self._fitted else 0

    # ------------------------------------------------------------------ #
    # normalization                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            norm = float(np.linalg.norm(x))
            return x / norm if norm > 1e-9 else x
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms > 1e-9, norms, 1.0)

    # ------------------------------------------------------------------ #
    # training and build                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, training_vectors: np.ndarray) -> None:
        if self._fitted:
            raise RuntimeError("IVFPQBackend already fitted")
        self._index.fit(self._normalize(training_vectors))
        self._fitted = True

    def add_batch(self, ids: list[Any], vectors: np.ndarray) -> None:
        if not self._fitted:
            # Silently drop pre-fit adds. sqlite-vec is the source of
            # truth until fit() runs and the CLI repopulates from there.
            return
        self._index.add_batch(list(ids), self._normalize(vectors))

    def delete(self, id_: Any) -> bool:
        if not self._fitted:
            return False
        return self._index.delete(id_)

    # ------------------------------------------------------------------ #
    # search                                                              #
    # ------------------------------------------------------------------ #

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[Any, float]]:
        if not self._fitted:
            return []
        kwargs: dict[str, Any] = {}
        if self._nprobe > 0:
            kwargs["nprobe"] = self._nprobe
        if self._rerank_candidates > 0 and self._keep_full_precision:
            kwargs["rerank_candidates"] = max(self._rerank_candidates, k)
        results = self._index.search(self._normalize(query), k=k, **kwargs)
        # IVFPQ returns similarity scores, not cosine distance. Convert
        # to a pseudo-distance in [0, 2] so downstream relevance_tier()
        # and ranking heuristics keep working. cosine_dist = 1 - cos_sim.
        return [(id_, max(0.0, 1.0 - float(score))) for id_, score in results]

    # ------------------------------------------------------------------ #
    # persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        if self._fitted:
            self._index.save(path)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        dim: int,
        nlist: int,
        M: int = 96,
        K: int = 256,
        seed: int = 0,
        keep_full_precision: bool = True,
        rerank_candidates: int = 100,
        nprobe: int = 0,
    ) -> IVFPQBackend:
        """Load a fitted index from disk, or return an unfit backend.

        The presence of the index file is authoritative: if ``path``
        exists it's treated as a fitted index. This avoids the split
        between an index file and a sidecar flag getting out of sync
        after a crash. Constructor validation (dim % M) still runs via
        the normal ``__init__`` path.
        """
        from snapvec import IVFPQSnapIndex

        backend = cls(
            dim=dim,
            nlist=nlist,
            M=M,
            K=K,
            seed=seed,
            keep_full_precision=keep_full_precision,
            rerank_candidates=rerank_candidates,
            nprobe=nprobe,
        )
        if Path(path).exists():
            backend._index = IVFPQSnapIndex.load(path)
            backend._fitted = True
        return backend
