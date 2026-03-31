#!/usr/bin/env python3
"""Regenerate float32 Lloyd–Max codebooks for ``turboquant/codebook.py``.

Requires SciPy (dev-only). From repo root::

    pip install scipy
    python scripts/precompute_lloyd_max_codebooks.py

Copy the printed blocks into ``_PRECOMPUTED`` in ``turboquant/codebook.py`` if
the numerical recipe changes.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def _lloyd_max_gaussian(bits: int, n_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
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


def main() -> None:
    for bits in (2, 3, 4):
        c, b = _lloyd_max_gaussian(bits)
        print(f"    {bits}: (")
        print(f"        np.array({c.tolist()}, dtype=np.float32),")
        print(f"        np.array({b.tolist()}, dtype=np.float32),")
        print("    ),")


if __name__ == "__main__":
    main()
