"""TurboQuant vector search index.

Provides add / search / delete / save / load on compressed vectors.
For collections < 100K, brute-force NumPy dot products are fast enough.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .codebook import dequantize_scalar, get_codebook
from .quantizer import CompressedVector, PolarQuantizer
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

    def __len__(self) -> int:
        return len(self._ids)

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

    def add_batch(self, ids: list, vectors: np.ndarray) -> None:
        """Add multiple vectors efficiently."""
        new_indices = []
        new_norms = []
        for i, v in enumerate(vectors):
            cv = self.quantizer.quantize(v)
            self._ids.append(ids[i])
            new_indices.append(cv.indices)
            new_norms.append(cv.norm)
        batch_indices = np.array(new_indices, dtype=np.uint8)
        batch_norms = np.array(new_norms, dtype=np.float32)
        if len(self._indices_matrix) == 0:
            self._indices_matrix = batch_indices
            self._norms = batch_norms
        else:
            self._indices_matrix = np.vstack([self._indices_matrix, batch_indices])
            self._norms = np.concatenate([self._norms, batch_norms])

    def delete(self, id: object) -> bool:
        """Remove a vector by ID. Returns True if found."""
        if id not in self._ids:
            return False
        idx = self._ids.index(id)
        self._ids.pop(idx)
        self._indices_matrix = np.delete(self._indices_matrix, idx, 0)
        self._norms = np.delete(self._norms, idx)
        return True

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[object, float]]:
        """Find k nearest neighbors by approximate cosine similarity.

        The query is rotated but NOT quantized (asymmetric search) for
        higher accuracy.

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
        q_scaled = q_rotated * np.sqrt(self.quantizer.padded_dim)

        # Asymmetric search: query stays float, stored vectors are dequantized
        stored_scaled = self._centroids[self._indices_matrix]  # (n, padded_dim)
        dots = stored_scaled @ q_scaled / self.quantizer.padded_dim  # (n,)

        actual_k = min(k, len(self._ids))
        top_indices = np.argpartition(dots, -actual_k)[-actual_k:]
        top_indices = top_indices[np.argsort(dots[top_indices])[::-1]]

        return [(self._ids[i], float(dots[i])) for i in top_indices]

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
            padded_dim = index.quantizer.padded_dim
            indices_bytes = f.read(n * padded_dim)
            norms_bytes = f.read(n * 4)
            index._indices_matrix = (
                np.frombuffer(indices_bytes, dtype=np.uint8).reshape(n, padded_dim).copy()
            )
            index._norms = np.frombuffer(norms_bytes, dtype=np.float32).copy()
            for _ in range(n):
                (id_len,) = struct.unpack("<H", f.read(2))
                raw = f.read(id_len).decode("utf-8")
                # Restore numeric IDs to their original type
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
        return {
            "n": n,
            "dim": self.dim,
            "bits": self.bits,
            "padded_dim": self.quantizer.padded_dim,
            "orig_bytes": orig_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": orig_bytes / max(compressed_bytes, 1),
        }
