"""TurboQuant vector search index.

Provides add / search / delete / save / load on compressed vectors.
For collections < 100K, brute-force NumPy dot products are fast enough.

Optimizations vs v0:
  - Batch WHT: all vectors rotated in one FWHT call during add_batch
  - Vectorized quantization: np.searchsorted on full (N, padded_dim) matrix
  - Precomputed stored_scaled: centroids[indices] cached as float16 matrix
    so each search is a single matmul with no lookup overhead
  - Cached sign vectors via lru_cache in rotation.py
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .codebook import get_codebook
from .quantizer import PolarQuantizer
from .rotation import pad_to_power_of_2, rotate

_MAGIC = b"TQVS"
_VERSION = 1


class TurboQuantIndex:
    """In-memory vector search index with PolarQuant compression.

    Parameters
    ----------
    dim : int
        Original embedding dimension.
    bits : int
        Quantization bits per coordinate (2, 3, or 4).
    seed : int
        Rotation seed (must be consistent across save/load).
    """

    def __init__(self, dim: int, bits: int = 4, seed: int = 0) -> None:
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.quantizer = PolarQuantizer(dim=dim, bits=bits, seed=seed)
        self._ids: list = []
        self._indices_matrix = np.zeros((0, self.quantizer.padded_dim), dtype=np.uint8)
        self._norms = np.zeros(0, dtype=np.float32)
        self._centroids, _ = get_codebook(bits)
        # Precomputed float16 matrix for fast dot products — invalidated on writes
        self._stored_scaled: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._ids)

    # ------------------------------------------------------------------ #
    # Write operations                                                     #
    # ------------------------------------------------------------------ #

    def add(self, id: object, vector: np.ndarray) -> None:
        """Add a single vector to the index."""
        cv = self.quantizer.quantize(vector)
        self._ids.append(id)
        row = cv.indices.reshape(1, -1)
        norm_row = np.array([cv.norm], dtype=np.float32)
        if len(self._indices_matrix) == 0:
            self._indices_matrix = row
            self._norms = norm_row
        else:
            self._indices_matrix = np.vstack([self._indices_matrix, row])
            self._norms = np.concatenate([self._norms, norm_row])
        self._stored_scaled = None  # invalidate cache

    def add_batch(self, ids: list, vectors: np.ndarray) -> None:
        """Add multiple vectors using a single batched WHT call.

        ~50x faster than calling add() in a loop for large batches.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        n = len(vectors)
        if n == 0:
            return

        # Compute norms and normalize to unit sphere
        norms = np.linalg.norm(vectors, axis=1)  # (n,)
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        units = vectors / safe_norms[:, None]

        # Pad to next power-of-2 in one shot: (n, padded_dim)
        pdim = self.quantizer.padded_dim
        padded = np.zeros((n, pdim), dtype=np.float32)
        padded[:, : self.dim] = units

        # Single batched rotate call — FWHT operates on last axis
        rotated = rotate(padded, self.seed)              # (n, pdim)
        scaled = rotated * np.sqrt(pdim)                 # (n, pdim)

        # Vectorized quantization: one searchsorted over full matrix
        _, boundaries = get_codebook(self.bits)
        flat_indices = np.searchsorted(boundaries, scaled.ravel()).astype(np.uint8)
        batch_indices = flat_indices.reshape(n, pdim)
        batch_norms = np.where(norms > 1e-10, norms, 0.0).astype(np.float32)

        self._ids.extend(ids)
        if len(self._indices_matrix) == 0:
            self._indices_matrix = batch_indices
            self._norms = batch_norms
        else:
            self._indices_matrix = np.vstack([self._indices_matrix, batch_indices])
            self._norms = np.concatenate([self._norms, batch_norms])

        self._stored_scaled = None  # invalidate cache

    def delete(self, id: object) -> bool:
        """Remove a vector by ID. Returns True if found."""
        if id not in self._ids:
            return False
        idx = self._ids.index(id)
        self._ids.pop(idx)
        self._indices_matrix = np.delete(self._indices_matrix, idx, 0)
        self._norms = np.delete(self._norms, idx)
        self._stored_scaled = None  # invalidate cache
        return True

    # ------------------------------------------------------------------ #
    # Search                                                              #
    # ------------------------------------------------------------------ #

    def _get_stored_scaled(self) -> np.ndarray:
        """Return precomputed dequantized matrix as float16.

        Built once, cached until the index is modified.
        Avoids per-query fancy indexing on self._indices_matrix.
        """
        if self._stored_scaled is None:
            self._stored_scaled = self._centroids[self._indices_matrix].astype(np.float16)
        return self._stored_scaled

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[object, float]]:
        """Find k nearest neighbors by approximate cosine similarity.

        Asymmetric search: query is rotated float32, stored vectors are
        precomputed float16 centroid lookup — single matmul per query.

        Returns list of (id, score) sorted by descending similarity.
        """
        if len(self._ids) == 0:
            return []

        q = np.asarray(query, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            return []

        q_unit = q / q_norm
        q_padded, _ = pad_to_power_of_2(q_unit)
        q_rotated = rotate(q_padded, self.seed)
        q_scaled = (q_rotated * np.sqrt(self.quantizer.padded_dim)).astype(np.float32)

        # Single matmul — stored_scaled is (n, pdim) float16
        stored = self._get_stored_scaled()
        dots = (stored @ q_scaled).astype(np.float32) / self.quantizer.padded_dim

        actual_k = min(k, len(self._ids))
        top_idx = np.argpartition(dots, -actual_k)[-actual_k:]
        top_idx = top_idx[np.argsort(dots[top_idx])[::-1]]

        return [(self._ids[i], float(dots[i])) for i in top_idx]

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Persist index to binary file."""
        path = Path(path)
        n = len(self._ids)
        with open(path, "wb") as f:
            f.write(_MAGIC)
            f.write(struct.pack("<IIIII", _VERSION, self.dim, self.bits, self.seed, n))
            f.write(self._indices_matrix.tobytes())
            f.write(self._norms.tobytes())
            for id_val in self._ids:
                encoded = str(id_val).encode("utf-8")
                f.write(struct.pack("<H", len(encoded)))
                f.write(encoded)

    @classmethod
    def load(cls, path: str | Path) -> "TurboQuantIndex":
        """Load index from binary file."""
        path = Path(path)
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != _MAGIC:
                raise ValueError(f"Invalid file: expected TQVS magic, got {magic!r}")
            version, dim, bits, seed, n = struct.unpack("<IIIII", f.read(20))
            if version != _VERSION:
                raise ValueError(f"Unsupported version: {version}")
            index = cls(dim=dim, bits=bits, seed=seed)
            pdim = index.quantizer.padded_dim
            indices_bytes = f.read(n * pdim)
            norms_bytes = f.read(n * 4)
            index._indices_matrix = (
                np.frombuffer(indices_bytes, dtype=np.uint8).reshape(n, pdim).copy()
            )
            index._norms = np.frombuffer(norms_bytes, dtype=np.float32).copy()
            for _ in range(n):
                (id_len,) = struct.unpack("<H", f.read(2))
                raw = f.read(id_len).decode("utf-8")
                try:
                    index._ids.append(int(raw))
                except ValueError:
                    try:
                        index._ids.append(float(raw))
                    except ValueError:
                        index._ids.append(raw)
        return index

    def stats(self) -> dict:
        """Return index statistics."""
        n = len(self._ids)
        orig_bytes = n * self.dim * 4
        compressed_bytes = self._indices_matrix.nbytes + self._norms.nbytes
        cache_bytes = self._stored_scaled.nbytes if self._stored_scaled is not None else 0
        return {
            "n": n,
            "dim": self.dim,
            "bits": self.bits,
            "padded_dim": self.quantizer.padded_dim,
            "orig_bytes": orig_bytes,
            "compressed_bytes": compressed_bytes,
            "cache_bytes": cache_bytes,
            "compression_ratio": orig_bytes / max(compressed_bytes, 1),
        }
