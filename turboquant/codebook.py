"""Lloyd-Max optimal scalar quantizer for Gaussian distribution.

After the randomized Hadamard transform, each coordinate of a unit vector
is approximately i.i.d. N(0, 1/d).  We precompute Lloyd-Max codebooks
for the standard normal and scale at runtime.

The codebook consists of:
  - centroids: 2^bits reconstruction values
  - boundaries: 2^bits - 1 decision thresholds
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def _lloyd_max_gaussian(bits: int, n_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Compute Lloyd-Max optimal quantizer for standard normal N(0,1).

    Parameters
    ----------
    bits : int
        Number of bits (2, 3, or 4).
    n_iter : int
        Lloyd iterations.

    Returns
    -------
    centroids : array of shape (2^bits,)
        Reconstruction values, sorted ascending.
    boundaries : array of shape (2^bits - 1,)
        Decision thresholds between adjacent centroids.
    """
    n_levels = 2**bits
    centroids = np.linspace(-3.0, 3.0, n_levels)

    for _ in range(n_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        edges = np.concatenate([[-np.inf], boundaries, [np.inf]])
        new_centroids = np.empty_like(centroids)
        for i in range(n_levels):
            lo, hi = edges[i], edges[i + 1]
            num, _ = quad(lambda x: x * norm.pdf(x), lo, hi)
            den, _ = quad(norm.pdf, lo, hi)
            new_centroids[i] = num / (den + 1e-15)
        if np.max(np.abs(new_centroids - centroids)) < 1e-15:
            break
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids.astype(np.float32), boundaries.astype(np.float32)


@lru_cache(maxsize=None)
def get_codebook(bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Get (or compute and cache) the Lloyd-Max codebook for N(0,1).

    Parameters
    ----------
    bits : 2, 3, or 4

    Returns
    -------
    (centroids, boundaries) — both float32 arrays.
    """
    if bits not in (2, 3, 4):
        raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
    return _lloyd_max_gaussian(bits)


def quantize_scalar(values: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantize an array of scalars using Lloyd-Max for N(0,1).

    Parameters
    ----------
    values : array of float32
        Values to quantize (should be approximately N(0, sigma) distributed).
    bits : int
        Quantization bits.

    Returns
    -------
    indices : array of uint8
        Codebook index for each value.
    centroids : array of float32
        The codebook centroids (for reconstruction).
    """
    centroids, boundaries = get_codebook(bits)
    indices = np.searchsorted(boundaries, values).astype(np.uint8)
    return indices, centroids


def dequantize_scalar(indices: np.ndarray, bits: int) -> np.ndarray:
    """Reconstruct values from codebook indices.

    Parameters
    ----------
    indices : array of uint8
    bits : int

    Returns
    -------
    Reconstructed float32 values.
    """
    centroids, _ = get_codebook(bits)
    return centroids[indices]
