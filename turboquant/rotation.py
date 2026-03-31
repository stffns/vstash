"""Fast Walsh-Hadamard Transform with random sign flips.

The randomized Hadamard transform (RHT) isotropizes unit vectors so that each
coordinate becomes approximately i.i.d. Gaussian(0, 1/d).  This enables
distribution-agnostic scalar quantization.

Complexity: O(d log d) — no matrix multiplication needed.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n == 0:
        return 1
    return 1 << (n - 1).bit_length()


def _fwht_inplace(x: np.ndarray) -> None:
    """In-place Fast Walsh-Hadamard Transform.

    Operates on the last axis.  Length must be a power of 2.
    """
    n = x.shape[-1]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            a = x[..., i : i + h].copy()
            b = x[..., i + h : i + 2 * h]
            x[..., i : i + h] = a + b
            x[..., i + h : i + 2 * h] = a - b
        h *= 2


@lru_cache(maxsize=64)
def _generate_signs(dim: int, seed: int) -> np.ndarray:
    """Generate a deterministic ±1 sign vector from a seed (cached)."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=dim).astype(np.float32)


def pad_to_power_of_2(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad vector(s) with zeros to the next power-of-2 length.

    Returns (padded_array, original_dim).
    """
    orig_dim = x.shape[-1]
    target = _next_power_of_2(orig_dim)
    if target == orig_dim:
        return x.copy(), orig_dim
    if x.ndim == 1:
        padded = np.zeros(target, dtype=x.dtype)
        padded[:orig_dim] = x
    else:
        padded = np.zeros((*x.shape[:-1], target), dtype=x.dtype)
        padded[..., :orig_dim] = x
    return padded, orig_dim


def rotate(x: np.ndarray, seed: int) -> np.ndarray:
    """Apply randomized Hadamard transform.

    Parameters
    ----------
    x : array of shape (..., d) where d is a power of 2
        Input vector(s).  Must already be padded.
    seed : int
        Deterministic seed for sign flips.

    Returns
    -------
    Rotated array of same shape, normalized by 1/sqrt(d).
    """
    d = x.shape[-1]
    signs = _generate_signs(d, seed).astype(np.float32)
    y = x * signs
    _fwht_inplace(y)
    return y / np.sqrt(d)


def unrotate(y: np.ndarray, seed: int) -> np.ndarray:
    """Inverse randomized Hadamard transform.

    WHT is its own inverse up to normalization, so:
        unrotate(y) = signs * WHT(y) / sqrt(d)
    """
    d = y.shape[-1]
    signs = _generate_signs(d, seed)
    x = y.copy().astype(np.float32)
    _fwht_inplace(x)
    return x * signs / np.sqrt(d)
