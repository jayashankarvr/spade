"""Zernike-moment descriptor (rotation-invariant).

Theory
------
Zernike polynomials V_{n,m}(rho, theta) form a complete orthogonal basis on
the unit disk. The magnitudes |Z_{n,m}| of the projection of an image onto
this basis are invariant to in-plane rotation. Khotanzad and Hong (1990)
showed they form a tight discriminative basis with size ~ O(N^2 / 2) for
order-N decomposition - perfect for tiny patches where we need rotation
invariance with minimal coefficients.

We implement real-valued Zernike polynomials directly: for each (n, m) with
n - |m| even and |m| <= n, V_{n,m} = R_{n,|m|}(rho) * cos(m*theta) [or sin
for negative m]. The descriptor is the magnitude of the projection vector
for each (n, |m|) pair.

Order tracks patch size so that the basis fits within the patch's degrees
of freedom: 3x3 -> order 4, 4x4 -> 5, 5x5 -> 6, 6x6 -> 8.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple
import numpy as np

from spade.descriptors.core import DescriptorStrategy, _to_gray, _normalize


def _radial_polynomial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """R_{n,m}(rho) - the radial Zernike polynomial."""
    if (n - m) % 2 != 0 or m > n:
        return np.zeros_like(rho)
    out = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coeff = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)
            )
        )
        out += coeff * rho ** (n - 2 * k)
    return out


def _moment_indices(max_order: int) -> list[Tuple[int, int]]:
    """List of (n, m) pairs with n - m even, 0 <= m <= n <= max_order."""
    return [
        (n, m)
        for n in range(max_order + 1)
        for m in range(0, n + 1)
        if (n - m) % 2 == 0
    ]


@lru_cache(maxsize=16)
def _basis_for_size(size: int, max_order: int) -> Tuple[np.ndarray, np.ndarray, list]:
    """Precompute Zernike basis matrices for an `size` x `size` grid.

    Returns:
        cos_basis: (k, size*size) - real (cos) basis vectors
        sin_basis: (k, size*size) - imaginary (sin) basis vectors (zero for m=0)
        indices:   list of (n, m) tuples in matching order
    """
    # Pixel centers normalized to [-1, 1]; mask out points outside unit disk
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size * 2.0 - 1.0
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    rho = np.sqrt(xx * xx + yy * yy)
    theta = np.arctan2(yy, xx)
    inside = rho <= 1.0

    indices = _moment_indices(max_order)
    cos_basis = np.zeros((len(indices), size * size), dtype=np.float32)
    sin_basis = np.zeros((len(indices), size * size), dtype=np.float32)
    for k, (n, m) in enumerate(indices):
        rad = _radial_polynomial(n, m, rho)
        cos_b = (rad * np.cos(m * theta)) * inside
        sin_b = (rad * np.sin(m * theta)) * inside
        # Normalization factor (n+1)/pi * pixel_area approximated as constant
        norm = (n + 1) / math.pi
        cos_basis[k] = (norm * cos_b).ravel()
        sin_basis[k] = (norm * sin_b).ravel()
    return cos_basis, sin_basis, indices


# Order-by-size schedule (matches the multi-scale design notes)
ORDER_FOR_SIZE = {3: 4, 4: 5, 5: 6, 6: 8}


class ZernikeDescriptor(DescriptorStrategy):
    """Rotation-invariant magnitude features from Zernike moments."""

    def __init__(self, max_order: int | None = None):
        self.max_order = max_order  # if None, set per-patch via ORDER_FOR_SIZE

    def compute(self, patch: np.ndarray) -> np.ndarray:
        gray = _to_gray(patch).astype(np.float32)
        size = gray.shape[0]
        order = self.max_order if self.max_order is not None else ORDER_FOR_SIZE.get(size, 4)
        cos_b, sin_b, _ = _basis_for_size(size, order)
        flat = gray.ravel()
        a = cos_b @ flat
        b = sin_b @ flat
        # Magnitude per (n, m): sqrt(a^2 + b^2). For m=0, sin coeff is 0 and
        # the magnitude reduces to |a| - already rotation-invariant.
        magnitudes = np.sqrt(a * a + b * b).astype(np.float32)
        return _normalize(magnitudes)


def feature_dim_for_size(size: int) -> int:
    """How many Zernike magnitudes the descriptor returns for this size."""
    order = ORDER_FOR_SIZE.get(size, 4)
    return len(_moment_indices(order))
