"""Rotation-invariant matching on rings.

In rotation mode a patch isn't a square of pixels but colour samples on four
circles around its centre (radius 1 out to one patch width, 16 samples each).

* The descriptor is the magnitude of each grey ring's angular Fourier
  coefficients. A rotation only shifts the samples around the circle, which
  leaves those magnitudes unchanged, so no orientation has to be estimated.
* The turn between two matched patches is where the rings' circular
  cross-correlation peaks; the target's rings are then read at that angle, so
  the pixel fits compare the pair already aligned.

On rotated, recolored JPEG photo splices this puts the true partner among the
8 nearest neighbours for 58% of interior source patches, against 36% for the
earlier approach (each patch resampled in its own intensity-centroid
orientation, which needed a 10 px disc and ran into host pixels near a
fragment's edge), and measures the angle within 10 degrees for 99% of true
matches, against 82%.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

from spade.descriptors.core import LUMA

RING_ANGLES = 16
_RINGS = 4


def ring_radii(patch_size: int) -> np.ndarray:
    return np.linspace(1.0, max(patch_size - 1, 2), _RINGS)


def ring_footprint(patch_size: int) -> int:
    """Width of the disc the rings cover, for counting how much patches overlap."""
    return 2 * int(np.ceil(ring_radii(patch_size)[-1])) + 1


def ring_samples(image: np.ndarray, coords: np.ndarray, patch_size: int,
                 phase: Optional[np.ndarray] = None) -> np.ndarray:
    """Samples on rings around each patch centre: (N, rings, angles) for a grey
    image, (N, rings, angles, 3) for colour.

    ``phase`` (radians, one per patch) turns where each set of rings starts,
    which is how a target is read at a match's measured angle.
    """
    radii = ring_radii(patch_size)
    a = 2 * np.pi * np.arange(RING_ANGLES) / RING_ANGLES
    a = a[None, :] + (0.0 if phase is None else np.asarray(phase, dtype=np.float64)[:, None])
    a = np.broadcast_to(a, (len(coords), RING_ANGLES))
    half = (patch_size - 1) / 2.0
    cx = coords[:, 0] + half
    cy = coords[:, 1] + half
    xs = cx[:, None, None] + radii[None, :, None] * np.cos(a)[:, None, :]
    ys = cy[:, None, None] + radii[None, :, None] * np.sin(a)[:, None, :]
    shape = (len(coords), len(radii), RING_ANGLES)
    if image.ndim == 2:
        return map_coordinates(image, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(shape)
    out = np.empty(shape + (image.shape[2],), dtype=np.float32)
    for c in range(image.shape[2]):
        out[..., c] = map_coordinates(image[..., c], [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(shape)
    return out


class RingDescriptor:
    """Rotation-invariant descriptor: angular Fourier magnitudes of the grey rings, unit norm."""

    def __init__(self, patch_size: int, min_std: float = 0.01):
        self.patch_size = patch_size
        self.min_std = min_std

    @property
    def dim(self) -> int:
        return _RINGS * (RING_ANGLES // 2 + 1)

    def compute_batch(self, rings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(N, rings, angles, 3) colour samples -> (descriptors (N, dim) float32, textured (N,) bool).

        Rings too flat to describe (same rule as the luminance descriptor) get
        a zero descriptor and textured=False.
        """
        gray = (rings.astype(np.float32) @ LUMA).reshape(len(rings), _RINGS, RING_ANGLES)
        gray -= gray.mean(axis=(1, 2), keepdims=True)
        textured = np.linalg.norm(gray.reshape(len(gray), -1), axis=1) > self.min_std * np.sqrt(_RINGS * RING_ANGLES)
        mag = np.abs(np.fft.rfft(gray, axis=2)).reshape(len(gray), -1).astype(np.float32)
        norm = np.linalg.norm(mag, axis=1, keepdims=True)
        mag = np.where(textured[:, None], mag / np.maximum(norm, 1e-12), 0.0).astype(np.float32)
        return mag, textured


def relative_angles(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation (radians) taking each source ring set onto its target ring set.

    Grey rings, (N, rings, angles). The circular cross-correlation, summed over
    radii, peaks at the turn between them; a parabola through the peak refines
    it below one sample step.
    """
    s = np.fft.fft(source - source.mean(axis=2, keepdims=True), axis=2)
    t = np.fft.fft(target - target.mean(axis=2, keepdims=True), axis=2)
    cc = np.real(np.fft.ifft(np.conj(s) * t, axis=2)).sum(axis=1)
    k = cc.argmax(axis=1)
    rows = np.arange(len(k))
    left, mid, right = cc[rows, (k - 1) % RING_ANGLES], cc[rows, k], cc[rows, (k + 1) % RING_ANGLES]
    curve = left - 2 * mid + right
    step = np.where(np.abs(curve) > 1e-12, 0.5 * (left - right) / np.where(curve == 0, 1, curve), 0.0)
    return (k + step) * 2 * np.pi / RING_ANGLES
