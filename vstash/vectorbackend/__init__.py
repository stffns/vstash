"""vstash.vectorbackend -- pluggable ANN backend adapters.

vstash supports three vector index backends today: sqlite-vec
(default, embedded in :class:`vstash.store.VstashStore` via the
``vec_chunks`` virtual table), snapvec (flat quantized), and
snapvec-ivfpq (IVF + product quantization with optional fp16
rerank). Historically each backend was wired into the store via a
hand-rolled if/elif chain (``_init_snapvec``, ``_init_ivfpq``,
``_save_snapvec``, ``_rebuild_snapvec_from_vec_chunks``, etc.),
which made adding or auditing a backend a per-call-site exercise.

This package defines a single :class:`VectorBackend` Protocol that
every backend implements: ``add_batch``, ``search``, ``delete``,
``save``, ``__len__``. The store calls into the protocol; backend-
specific lifecycle (``fit``, ``load`` factory, training-sample
sizing) lives on the concrete adapter rather than on the protocol
because the call patterns differ per backend (sqlite-vec persists
implicitly via the SQLite file, snapvec backends round-trip a
binary blob, IVFPQ requires an explicit ``fit`` before adds).

Migration is staged across multiple PRs to keep risk bounded:

- v0.37 (this package): Protocol defined; IVFPQ adapter moved here
  from ``vstash/_ivfpq_backend.py`` (with a deprecated re-export
  shim). sqlite-vec stays embedded in the store; snapvec flat
  stays inline. Behaviour is unchanged.
- v0.38 (follow-up): extract snapvec flat into
  ``vectorbackend/snapvec_flat.py``.
- v0.39+ (later): consider extracting sqlite-vec; lower priority
  because the virtual-table integration is already encapsulated by
  the existing SQL queries.
"""

from __future__ import annotations

from .base import VectorBackend
from .snapvec_ivfpq import IVFPQBackend

__all__ = [
    "IVFPQBackend",
    "VectorBackend",
]
