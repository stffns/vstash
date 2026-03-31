"""Lloyd-Max optimal scalar quantizer for Gaussian distribution.

After the randomized Hadamard transform, each coordinate of a unit vector
is approximately i.i.d. N(0, 1/d).  We use precomputed Lloyd–Max codebooks
for the standard normal (2, 3, 4 bits) so runtime needs only NumPy — no SciPy.

Centroids and boundaries match the former scipy.integrate implementation
(float32); see ``scripts/precompute_lloyd_max_codebooks.py`` to regenerate.
"""
from __future__ import annotations

import numpy as np

# Precomputed Lloyd–Max codebooks for N(0,1), dtype float32 (legacy-compatible).
_PRECOMPUTED: dict[int, tuple[np.ndarray, np.ndarray]] = {
    2: (
        np.array(
            [-1.5104176, -0.45278004, 0.45278004, 1.5104176],
            dtype=np.float32,
        ),
        np.array(
            [-0.9815988, 0.0, 0.9815988],
            dtype=np.float32,
        ),
    ),
    3: (
        np.array(
            [
                -2.1519468,
                -1.3439103,
                -0.75600606,
                -0.24509446,
                0.24509446,
                0.75600606,
                1.3439103,
                2.1519468,
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                -1.7479285,
                -1.0499582,
                -0.50055027,
                0.0,
                0.50055027,
                1.0499582,
                1.7479285,
            ],
            dtype=np.float32,
        ),
    ),
    4: (
        np.array(
            [
                -2.7455692,
                -2.0836654,
                -1.6329745,
                -1.2702826,
                -0.95448357,
                -0.66611624,
                -0.39394602,
                -0.13040923,
                0.13040923,
                0.39394602,
                0.66611624,
                0.95448357,
                1.2702826,
                1.6329745,
                2.0836654,
                2.7455692,
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                -2.4146173,
                -1.8583199,
                -1.4516286,
                -1.1123831,
                -0.81029987,
                -0.53003114,
                -0.26217762,
                1.2490009e-16,
                0.26217762,
                0.53003114,
                0.81029987,
                1.1123831,
                1.4516286,
                1.8583199,
                2.4146173,
            ],
            dtype=np.float32,
        ),
    ),
}


def get_codebook(bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroids, boundaries) for Lloyd–Max N(0,1) quantizer.

    Parameters
    ----------
    bits : 2, 3, or 4

    Returns
    -------
    (centroids, boundaries) — both float32 arrays; shared read-only views.
    """
    if bits not in _PRECOMPUTED:
        raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
    return _PRECOMPUTED[bits]


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
