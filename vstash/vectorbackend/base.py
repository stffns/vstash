"""vstash.vectorbackend.base -- the VectorBackend Protocol.

Every concrete vector backend that vstash speaks to (sqlite-vec,
snapvec flat, snapvec-ivfpq) presents the same shape to the store
hot path. This Protocol pins that shape so the store does not need
backend-specific type switches and a future contributor can add a
new backend by implementing the protocol plus a registration entry
in :mod:`vstash.vectorbackend`.

The protocol is structural (``Protocol`` from :mod:`typing`), so
existing classes that already match the shape (such as
:class:`vstash.vectorbackend.snapvec_ivfpq.IVFPQBackend`) satisfy
it without inheritance. ``runtime_checkable`` lets adapters that
want a defensive ``isinstance(b, VectorBackend)`` assert it.

Notes on shape:

- ``add_batch(ids, vectors)`` accepts a list of opaque chunk-id
  values plus an array of vectors. Vector arrays are typed
  ``np.ndarray`` rather than ``Sequence[Sequence[float]]`` because
  every existing implementation does numpy work internally and
  enforcing the type at the protocol surface saves redundant
  conversions.
- ``search(query, k)`` returns ``list[tuple[id, distance]]``.
  Distance semantics are cosine in [0, 2] for snapvec backends
  (after the similarity-to-distance conversion in
  ``IVFPQBackend.search``) and squared cosine for sqlite-vec under
  schema v2.
- ``__len__`` returns the count of *routable* vectors, which is 0
  before an IVFPQ index has been fit() even though the underlying
  storage may already hold all chunks.
- ``save`` and ``load`` work with a path string. snapvec backends
  write a single binary blob; sqlite-vec persists implicitly via
  the SQLite database file, so its adapter (when extracted) will
  no-op these methods.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorBackend(Protocol):
    """Structural protocol every vector backend implements.

    Concrete implementations:

    - :class:`vstash.vectorbackend.snapvec_ivfpq.IVFPQBackend` --
      IVF + product quantization with optional fp16 rerank.
      Requires an explicit ``fit`` call before adds; this is a
      backend-specific lifecycle that the protocol does not
      surface (callers that need it know they instantiated an
      IVFPQ adapter).
    - snapvec flat (extracted in a follow-up PR).
    - sqlite-vec (extracted in a later PR if the cost/benefit
      proves out -- the virtual-table integration in
      :class:`vstash.store.VstashStore` is already encapsulated
      by the existing SQL queries, so the win is marginal).
    """

    dim: int

    def __len__(self) -> int:
        """Routable vector count.

        Returns 0 for unfit IVFPQ adapters even when chunks exist
        upstream. This contract lets the store fall back to
        sqlite-vec while a snapvec index is being trained.
        """
        ...

    def add_batch(self, ids: list[Any], vectors: np.ndarray) -> None:
        """Insert ``len(ids)`` vectors. Pre-fit IVFPQ silently no-ops."""
        ...

    def delete(self, id_: Any) -> bool:
        """Remove the vector with key ``id_``. Returns True if found."""
        ...

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[Any, float]]:
        """Return the ``k`` nearest neighbours as ``[(id, distance), ...]``.

        Distance is cosine in [0, 2] for snapvec backends, squared
        cosine for sqlite-vec under schema v2. Empty list if the
        backend is not yet ready to answer (pre-fit IVFPQ).
        """
        ...

    def save(self, path: str) -> None:
        """Persist the index to ``path``. No-op for adapters that
        store their state in the SQLite file."""
        ...
