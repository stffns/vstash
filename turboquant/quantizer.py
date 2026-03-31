"""PolarQuant core: quantize and dequantize embedding vectors.

Pipeline:
  quantize:   vector → normalize → pad → rotate → scalar quantize
  dequantize: indices → scalar dequantize → unrotate → unpad → rescale
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codebook import dequantize_scalar, quantize_scalar
from .rotation import pad_to_power_of_2, rotate, unrotate


@dataclass
class CompressedVector:
    """A quantized embedding vector."""

    indices: np.ndarray   # uint8, shape (padded_dim,)
    norm: np.float32
    orig_dim: int
    bits: int
    seed: int

    @property
    def nbytes(self) -> int:
        """Compressed size in bytes."""
        return self.indices.nbytes + 13  # 4 (norm) + 4 (orig_dim) + 2 (bits+seed) + 3 padding

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs float32."""
        original_bytes = self.orig_dim * 4
        return original_bytes / self.nbytes


class PolarQuantizer:
    """PolarQuant vector quantizer.

    Parameters
    ----------
    dim : int
        Original embedding dimension (e.g. 384 for BGE-small).
    bits : int
        Bits per coordinate (2, 3, or 4).
    seed : int
        Deterministic seed for the rotation matrix.
    """

    def __init__(self, dim: int, bits: int = 4, seed: int = 0) -> None:
        self.dim = dim
        self.bits = bits
        self.seed = seed
        dummy = np.zeros(dim, dtype=np.float32)
        padded, _ = pad_to_power_of_2(dummy)
        self.padded_dim = padded.shape[0]

    def quantize(self, vector: np.ndarray) -> CompressedVector:
        """Compress a single embedding vector.

        Parameters
        ----------
        vector : array of shape (dim,), float32

        Returns
        -------
        CompressedVector with quantized indices and metadata.
        """
        v = np.asarray(vector, dtype=np.float32)
        if v.shape != (self.dim,):
            raise ValueError(f"Expected shape ({self.dim},), got {v.shape}")

        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return CompressedVector(
                indices=np.zeros(self.padded_dim, dtype=np.uint8),
                norm=np.float32(0.0),
                orig_dim=self.dim,
                bits=self.bits,
                seed=self.seed,
            )

        unit = v / norm
        padded, _ = pad_to_power_of_2(unit)
        rotated = rotate(padded, self.seed)
        scaled = rotated * np.sqrt(self.padded_dim)
        indices, _ = quantize_scalar(scaled, self.bits)

        return CompressedVector(
            indices=indices,
            norm=np.float32(norm),
            orig_dim=self.dim,
            bits=self.bits,
            seed=self.seed,
        )

    def dequantize(self, cv: CompressedVector) -> np.ndarray:
        """Reconstruct an approximate vector from compressed form.

        Returns array of shape (dim,), float32.
        """
        if cv.norm < 1e-10:
            return np.zeros(cv.orig_dim, dtype=np.float32)

        scaled = dequantize_scalar(cv.indices, cv.bits)
        rotated = scaled / np.sqrt(self.padded_dim)
        padded_unit = unrotate(rotated, cv.seed)
        unit = padded_unit[: cv.orig_dim]
        unit_norm = np.linalg.norm(unit)
        if unit_norm > 1e-10:
            unit = unit / unit_norm
        return (unit * cv.norm).astype(np.float32)

    def quantize_batch(self, vectors: np.ndarray) -> list[CompressedVector]:
        """Quantize a batch of vectors.

        Parameters
        ----------
        vectors : array of shape (n, dim), float32

        Returns
        -------
        List of CompressedVector.
        """
        return [self.quantize(v) for v in vectors]

    def approximate_cosine(self, cv_a: CompressedVector, cv_b: CompressedVector) -> float:
        """Approximate cosine similarity between two compressed vectors.

        Uses quantized dot product in the rotated domain.
        """
        a_scaled = dequantize_scalar(cv_a.indices, cv_a.bits)
        b_scaled = dequantize_scalar(cv_b.indices, cv_b.bits)
        dot = np.dot(a_scaled[: self.padded_dim], b_scaled[: self.padded_dim])
        return float(dot / self.padded_dim)

    def approximate_cosine_query(self, query: np.ndarray, cv: CompressedVector) -> float:
        """Cosine similarity between a raw query and a compressed vector.

        The query is rotated (not quantized) for higher accuracy.
        """
        q = np.asarray(query, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            return 0.0
        q_unit = q / q_norm
        q_padded, _ = pad_to_power_of_2(q_unit)
        q_rotated = rotate(q_padded, self.seed)
        q_scaled = q_rotated * np.sqrt(self.padded_dim)
        v_scaled = dequantize_scalar(cv.indices, cv.bits)
        dot = np.dot(q_scaled, v_scaled[: self.padded_dim])
        return float(dot / self.padded_dim)
