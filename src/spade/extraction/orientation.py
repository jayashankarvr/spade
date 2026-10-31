"""Canonical orientation for rotation-invariant matching.

Each patch gets an angle from the intensity centroid of a disc around it (the
ORB recipe): theta = atan2(m01, m10), where m10 and m01 are first moments of
the mean-removed grey levels. Unlike a structure-tensor axis this is a
direction in [-pi, pi), so a rotated copy gets the same canonical frame as its
source. Patches are then resampled on a grid turned by theta, which makes the
luminance pattern - and the per-patch fit - the same up to interpolation.

The disc is larger than the patch: the centroid of a 5x5 patch is too noisy to
orient by, a radius of about twice the patch size is stable.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import correlate, map_coordinates, uniform_filter

from spade.descriptors.core import LUMA


def orientation_radius(patch_size: int) -> int:
    return max(4, 2 * patch_size)


def _moment_kernels(radius: int):
    v, u = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float64)
    disc = (u * u + v * v) <= radius * radius
    return u * disc, v * disc


def patch_orientations(image: np.ndarray, coords: np.ndarray, patch_size: int) -> np.ndarray:
    """Angle (radians) of each patch whose top-left corner is in ``coords`` (x, y)."""
    gray = image @ LUMA
    radius = orientation_radius(patch_size)
    centered = gray - uniform_filter(gray, size=2 * radius + 1, mode="reflect")
    ku, kv = _moment_kernels(radius)
    m10 = correlate(centered, ku, mode="reflect")
    m01 = correlate(centered, kv, mode="reflect")
    half = (patch_size - 1) // 2
    cx = coords[:, 0] + half
    cy = coords[:, 1] + half
    return np.arctan2(m01[cy, cx], m10[cy, cx])


def canonical_patches(image: np.ndarray, coords: np.ndarray, patch_size: int, theta: np.ndarray) -> np.ndarray:
    """Resample each patch on a grid rotated by its ``theta`` about its centre.

    Returns (N, s, s, 3) float32, the same layout as axis-aligned patches.
    """
    n = len(coords)
    if n == 0:
        return np.zeros((0, patch_size, patch_size, 3), dtype=np.float32)
    offset = np.arange(patch_size) - (patch_size - 1) / 2.0
    gv, gu = np.meshgrid(offset, offset, indexing="ij")          # (s, s) row, col offsets
    cos, sin = np.cos(theta)[:, None, None], np.sin(theta)[:, None, None]
    cx = (coords[:, 0] + (patch_size - 1) / 2.0)[:, None, None]
    cy = (coords[:, 1] + (patch_size - 1) / 2.0)[:, None, None]
    # Grid axis u is turned to point along theta.
    xs = cx + cos * gu - sin * gv
    ys = cy + sin * gu + cos * gv
    out = np.empty((n, patch_size, patch_size, 3), dtype=np.float32)
    for c in range(3):
        out[..., c] = map_coordinates(image[..., c], [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(
            n, patch_size, patch_size
        )
    return out
