"""TurboQuant vector search index.

Two search modes:

  TurboQuant_mse  (use_prod=False, default)
    Pipeline: normalize → pad → WHT → Lloyd-Max(b bits) → uint8 indices
    Score:    dot(centroids[idx], q_scaled) / pdim
    Bias:     small systematic underestimate (cosine bias ~2/π at 1-bit)

  TurboQuant_prod (use_prod=True)
    Pipeline: same WHT, but Lloyd-Max at (b-1) bits + 1-bit QJL correction
    Score:    mse_score + sqrt(π/2)/pdim * ||r|| * dot(S·q, sign(S·r))
    Bias:     zero — unbiased inner product estimator (Zandieh et al. 2025)
    Recall:   significantly higher at equal bit-width vs mse-only

Reference: TurboQuant — Online Vector Quantization with Near-optimal
Distortion Rate, Zandieh, Daliri, Hadian, Mirrokni (arXiv:2504.19874)

File format versions:
  v1  legacy — mse only
  v2  adds flags field; bit-0=use_prod, stores qjl_signs + residual_norms
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .codebook import get_codebook
from .quantizer import PolarQuantizer
from .rotation import pad_to_power_of_2, rotate

_MAGIC = b"TQVS"
_VERSION = 2

_FLAG_USE_PROD = 0x1


class TurboQuantIndex:
    """In-memory vector search index with TurboQuant compression.

    Parameters
    ----------
    dim : int
        Original embedding dimension.
    bits : int
        Total quantization bits per coordinate (2, 3, or 4).
        In prod mode, (bits-1) go to the MSE stage and 1 bit to QJL.
        Minimum bits=3 when use_prod=True.
    seed : int
        Rotation seed — must be consistent across save/load.
    use_prod : bool
        When True, enables TurboQuant_prod: unbiased inner product
        estimation via QJL residual correction.  Higher recall at the
        same bit-width, ~2x search cost.
    """

    def __init__(
        self,
        dim: int,
        bits: int = 4,
        seed: int = 0,
        use_prod: bool = False,
    ) -> None:
        if use_prod and bits < 3:
            raise ValueError(
                f"use_prod=True requires bits >= 3 (got {bits}). "
                "TurboQuant_prod uses (bits-1) for MSE stage, minimum 2 bits."
            )
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.use_prod = use_prod
        self.quantizer = PolarQuantizer(dim=dim, bits=self._mse_bits, seed=seed)

        self._ids: list = []
        # O(1) id→position lookup; kept in sync with self._ids on every write
        self._id_to_pos: dict = {}
        self._indices_matrix = np.zeros((0, self.quantizer.padded_dim), dtype=np.uint8)
        self._norms = np.zeros(0, dtype=np.float32)
        self._centroids, _ = get_codebook(self._mse_bits)

        # TurboQuant_prod fields — only allocated when use_prod=True
        # _qjl_signs:       (N, pdim) int8, ±1  — sign(S · r) per stored vector
        # _residual_norms:  (N,) float32         — ||r_scaled|| per stored vector
        # _S:               (pdim, pdim) float32  — fixed random Gaussian matrix
        self._qjl_signs: np.ndarray | None = None
        self._residual_norms_prod: np.ndarray | None = None
        self._S: np.ndarray | None = None
        if use_prod:
            self._S = self._make_S(self.quantizer.padded_dim, seed)

        # Float16 cache for MSE matmul — invalidated on writes
        self._stored_scaled: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def _mse_bits(self) -> int:
        """Bits used for the MSE stage (full bits in mse mode, bits-1 in prod)."""
        return self.bits - 1 if self.use_prod else self.bits

    @staticmethod
    def _make_S(pdim: int, seed: int) -> np.ndarray:
        """Generate the fixed (pdim, pdim) random Gaussian matrix for QJL.

        Uses seed+1337 to keep it distinct from the WHT rotation seed.
        Entries are i.i.d. N(0, 1) — the 1/d scaling is applied at score time.
        """
        rng = np.random.default_rng(seed + 1337)
        return rng.standard_normal((pdim, pdim)).astype(np.float32)

    def __len__(self) -> int:
        return len(self._ids)

    # ------------------------------------------------------------------ #
    # Write operations                                                     #
    # ------------------------------------------------------------------ #

    def add(self, id: object, vector: np.ndarray) -> None:
        """Add a single vector. Use add_batch for bulk inserts (50x faster)."""
        self.add_batch([id], np.asarray(vector, dtype=np.float32).reshape(1, -1))

    def add_batch(self, ids: list, vectors: np.ndarray) -> None:
        """Add multiple vectors in one batched WHT call.

        ~50x faster than calling add() in a loop for large batches.
        In prod mode also computes QJL signs and residual norms.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        n = len(vectors)
        if n == 0:
            return

        # Normalize to unit sphere, preserve original norms
        norms = np.linalg.norm(vectors, axis=1)           # (n,)
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        units = vectors / safe_norms[:, None]

        # Pad to power-of-2 and batch-WHT in one call — O(n · d log d)
        pdim = self.quantizer.padded_dim
        padded = np.zeros((n, pdim), dtype=np.float32)
        padded[:, : self.dim] = units
        rotated = rotate(padded, self.seed)               # (n, pdim)
        scaled = rotated * np.sqrt(pdim)                  # (n, pdim) ≈ N(0,1)

        # MSE quantization at _mse_bits
        _, boundaries = get_codebook(self._mse_bits)
        flat_indices = np.searchsorted(boundaries, scaled.ravel()).astype(np.uint8)
        batch_indices = flat_indices.reshape(n, pdim)
        batch_norms = np.where(norms > 1e-10, norms, 0.0).astype(np.float32)

        # TurboQuant_prod: QJL on the quantization residual
        if self.use_prod:
            assert self._S is not None
            # r_scaled = x_scaled - x̃_mse_scaled
            centroids_mse = self._centroids           # (2^mse_bits,)
            reconstructed = centroids_mse[batch_indices]   # (n, pdim)
            r_scaled = (scaled - reconstructed).astype(np.float32)   # (n, pdim)

            # Work in unscaled rotated space (r_rot = r_scaled / sqrt(pdim)).
            # sign is scale-invariant so sign(S·r_rot) = sign(S·r_scaled).
            r_rot = r_scaled / np.sqrt(pdim)              # (n, pdim)

            # QJL signs: sign(S · r_rot)  ← sign-invariant, use r_scaled for speed
            S_r = (self._S @ r_scaled.T).T                # (n, pdim)
            qjl = np.sign(S_r).astype(np.int8)
            qjl[qjl == 0] = 1                             # avoid zero

            # Store ‖r_rot‖ — the norm in unscaled space, as required by the formula:
            #   correction = sqrt(π/2)/pdim · ‖r_rot‖ · dot(S·q_rot, sign(S·r_rot))
            r_norms = np.linalg.norm(r_rot, axis=1).astype(np.float32)  # (n,)

        # Update ids and position map
        start = len(self._ids)
        self._ids.extend(ids)
        for i, id_val in enumerate(ids):
            self._id_to_pos[id_val] = start + i

        # Append indices and norms
        if len(self._indices_matrix) == 0:
            self._indices_matrix = batch_indices
            self._norms = batch_norms
        else:
            self._indices_matrix = np.vstack([self._indices_matrix, batch_indices])
            self._norms = np.concatenate([self._norms, batch_norms])

        # Append QJL data
        if self.use_prod:
            if self._qjl_signs is None:
                self._qjl_signs = qjl
                self._residual_norms_prod = r_norms
            else:
                self._qjl_signs = np.vstack([self._qjl_signs, qjl])
                self._residual_norms_prod = np.concatenate(
                    [self._residual_norms_prod, r_norms]
                )

        self._stored_scaled = None  # invalidate cache

    def delete(self, id: object) -> bool:
        """Remove a vector by ID. O(1) lookup, O(n) position compaction."""
        if id not in self._id_to_pos:
            return False
        idx = self._id_to_pos.pop(id)
        self._ids.pop(idx)
        self._indices_matrix = np.delete(self._indices_matrix, idx, 0)
        self._norms = np.delete(self._norms, idx)
        if self._qjl_signs is not None:
            self._qjl_signs = np.delete(self._qjl_signs, idx, 0)
            self._residual_norms_prod = np.delete(self._residual_norms_prod, idx)
        # Compact position map
        for id_val, pos in self._id_to_pos.items():
            if pos > idx:
                self._id_to_pos[id_val] = pos - 1
        self._stored_scaled = None
        return True

    # ------------------------------------------------------------------ #
    # Search                                                              #
    # ------------------------------------------------------------------ #

    def _get_stored_scaled(self) -> np.ndarray:
        """Precomputed (N, pdim) float16 centroid matrix — built once, cached."""
        if self._stored_scaled is None:
            self._stored_scaled = self._centroids[self._indices_matrix].astype(np.float16)
        return self._stored_scaled

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[object, float]]:
        """Find k nearest neighbors by approximate cosine similarity.

        TurboQuant_mse  (use_prod=False):
            Single matmul on float16 centroid cache — O(N·d) per query.

        TurboQuant_prod (use_prod=True):
            MSE score + QJL correction term — two matmuls, zero bias:
            score = dot(x̃_mse, q)/pdim
                  + sqrt(π/2)/pdim · ||r|| · dot(S·q, sign(S·r))

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
        pdim = self.quantizer.padded_dim
        q_scaled = (q_rotated * np.sqrt(pdim)).astype(np.float32)

        # Stage 1 — MSE score (always computed)
        stored = self._get_stored_scaled()                       # (N, pdim) float16
        dots = (stored @ q_scaled).astype(np.float32) / pdim    # (N,)

        # Stage 2 — QJL correction (prod mode only)
        # Formula (Zandieh et al., Lemma 4, in unscaled rotated space d=pdim):
        #   correction_i = sqrt(π/2)/pdim · ‖r_rot_i‖ · dot(S·q_rot, sign(S·r_rot_i))
        # q_rot  = q_scaled / sqrt(pdim)   — unit vector in rotated space
        # r_rot  = r_scaled / sqrt(pdim)   — stored norm already in unscaled units
        # sign(S·r_rot) = sign(S·r_scaled) — scale-invariant
        if self.use_prod and self._qjl_signs is not None:
            assert self._S is not None
            q_rot = q_scaled / np.sqrt(pdim)                      # (pdim,) unit
            S_q_rot = self._S @ q_rot                             # (pdim,)
            # dot(qjl_signs_i, S_q_rot) for all i: (N, pdim) @ (pdim,) → (N,)
            qjl_dots = self._qjl_signs.astype(np.float32) @ S_q_rot  # (N,)
            correction = (
                np.sqrt(np.pi / 2.0)
                / pdim
                * self._residual_norms_prod   # ‖r_rot‖ per stored vector
                * qjl_dots
            )
            dots = dots + correction

        actual_k = min(k, len(self._ids))
        top_idx = np.argpartition(dots, -actual_k)[-actual_k:]
        top_idx = top_idx[np.argsort(dots[top_idx])[::-1]]

        return [(self._ids[i], float(dots[i])) for i in top_idx]

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pack_indices(mat: np.ndarray, bits: int) -> bytes:
        """Pack uint8 index matrix into a bit-packed byte string."""
        if bits == 8:
            return mat.tobytes()
        indices_per_byte = 8 // bits
        mask = (1 << bits) - 1
        flat = mat.ravel()
        pad = (-len(flat)) % indices_per_byte
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
        out = np.zeros(len(flat) // indices_per_byte, dtype=np.uint8)
        for i in range(indices_per_byte):
            out |= (flat[i::indices_per_byte] & mask).astype(np.uint8) << (i * bits)
        return out.tobytes()

    @staticmethod
    def _unpack_indices(data: bytes, n_rows: int, n_cols: int, bits: int) -> np.ndarray:
        """Unpack bit-packed bytes back to uint8 matrix."""
        if bits == 8:
            return np.frombuffer(data, dtype=np.uint8).reshape(n_rows, n_cols).copy()
        indices_per_byte = 8 // bits
        mask = (1 << bits) - 1
        packed = np.frombuffer(data, dtype=np.uint8)
        total = n_rows * n_cols
        out = np.zeros(len(packed) * indices_per_byte, dtype=np.uint8)
        for i in range(indices_per_byte):
            out[i::indices_per_byte] = (packed >> (i * bits)) & mask
        return out[:total].reshape(n_rows, n_cols).copy()

    def save(self, path: str | Path) -> None:
        """Persist index to binary file (v2 format, bit-packed MSE indices)."""
        path = Path(path)
        n = len(self._ids)
        flags = _FLAG_USE_PROD if self.use_prod else 0
        packed = self._pack_indices(self._indices_matrix, self._mse_bits)

        with open(path, "wb") as f:
            # v2 header: magic + version + dim + bits + seed + n + flags
            f.write(_MAGIC)
            f.write(struct.pack("<IIIIII", _VERSION, self.dim, self.bits, self.seed, n, flags))
            # MSE indices
            f.write(struct.pack("<I", len(packed)))
            f.write(packed)
            # Vector norms
            f.write(self._norms.tobytes())
            # QJL data (prod mode only)
            if self.use_prod and self._qjl_signs is not None:
                f.write(self._qjl_signs.tobytes())           # (n, pdim) int8
                f.write(self._residual_norms_prod.tobytes())  # (n,) float32
            # IDs
            for id_val in self._ids:
                encoded = str(id_val).encode("utf-8")
                f.write(struct.pack("<H", len(encoded)))
                f.write(encoded)

    @classmethod
    def load(cls, path: str | Path) -> "TurboQuantIndex":
        """Load index from binary file (supports v1 and v2 formats)."""
        path = Path(path)
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != _MAGIC:
                raise ValueError(f"Invalid file: expected TQVS magic, got {magic!r}")

            # Detect format version from header size
            header_raw = f.read(4)
            (version,) = struct.unpack("<I", header_raw)

            if version == 1:
                # Legacy v1: no flags field
                dim, bits, seed, n = struct.unpack("<IIII", f.read(16))
                use_prod = False
            elif version == 2:
                dim, bits, seed, n, flags = struct.unpack("<IIIII", f.read(20))
                use_prod = bool(flags & _FLAG_USE_PROD)
            else:
                raise ValueError(f"Unsupported file version: {version}")

            index = cls(dim=dim, bits=bits, seed=seed, use_prod=use_prod)
            pdim = index.quantizer.padded_dim
            mse_bits = index._mse_bits

            (packed_len,) = struct.unpack("<I", f.read(4))
            packed_bytes = f.read(packed_len)
            norms_bytes = f.read(n * 4)

            index._indices_matrix = cls._unpack_indices(packed_bytes, n, pdim, mse_bits)
            index._norms = np.frombuffer(norms_bytes, dtype=np.float32).copy()

            # QJL data (v2 prod mode only)
            if use_prod and n > 0:
                qjl_bytes = f.read(n * pdim)         # (n, pdim) int8
                rnorm_bytes = f.read(n * 4)           # (n,) float32
                index._qjl_signs = np.frombuffer(
                    qjl_bytes, dtype=np.int8
                ).reshape(n, pdim).copy()
                index._residual_norms_prod = np.frombuffer(
                    rnorm_bytes, dtype=np.float32
                ).copy()

            # IDs + rebuild position map
            for pos in range(n):
                (id_len,) = struct.unpack("<H", f.read(2))
                raw = f.read(id_len).decode("utf-8")
                try:
                    id_val: object = int(raw)
                except ValueError:
                    try:
                        id_val = float(raw)
                    except ValueError:
                        id_val = raw
                index._ids.append(id_val)
                index._id_to_pos[id_val] = pos

        return index

    # ------------------------------------------------------------------ #
    # Stats                                                               #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        """Return index statistics including compression ratio and QJL overhead."""
        n = len(self._ids)
        orig_bytes = n * self.dim * 4  # float32 baseline
        compressed_bytes = self._indices_matrix.nbytes + self._norms.nbytes
        qjl_bytes = 0
        if self._qjl_signs is not None:
            qjl_bytes = self._qjl_signs.nbytes + self._residual_norms_prod.nbytes
        cache_bytes = self._stored_scaled.nbytes if self._stored_scaled is not None else 0
        S_bytes = self._S.nbytes if self._S is not None else 0
        return {
            "n": n,
            "dim": self.dim,
            "bits": self.bits,
            "mse_bits": self._mse_bits,
            "use_prod": self.use_prod,
            "padded_dim": self.quantizer.padded_dim,
            "orig_bytes": orig_bytes,
            "compressed_bytes": compressed_bytes,
            "qjl_bytes": qjl_bytes,
            "total_bytes": compressed_bytes + qjl_bytes,
            "cache_bytes": cache_bytes,
            "S_bytes": S_bytes,
            "compression_ratio": orig_bytes / max(compressed_bytes + qjl_bytes, 1),
        }
