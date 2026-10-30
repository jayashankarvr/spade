"""Patch descriptor: the normalized luminance pattern.

A patch is described by its grey levels minus their mean, scaled to unit
length. That's exactly invariant to brightness and contrast changes and, for
the color grades people actually apply (white balance, exposure, saturation,
mild channel mixing), close to invariant to the full affine color change the
verifier models.

Why not something exactly invariant to any 3x3 color matrix? We built one (the
projector onto the span of the centered RGB columns) and it's excellent on
uncompressed images, but JPEG's chroma subsampling wipes out exactly the
per-pixel color detail it relies on: recall at 0.1% false matches fell from
~0.95 to ~0.07 at quality 90. Luminance is stored at full resolution by JPEG,
and this descriptor kept ~0.9 recall in the same test while also beating the
previous five-part composite under compression. Evidence for the full color
transform is gathered later, per region, where there are enough pixels to fit
it.

Patches with less than ``min_std`` grey-level spread are flat: they match each
other trivially and carry no evidence, so callers drop them.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # ITU-R BT.601


def luminance(patches: np.ndarray) -> np.ndarray:
    """(N, ..., 3) RGB -> (N, ...) grey levels."""
    return patches @ LUMA


class LuminanceDescriptor:
    """Zero-mean, unit-norm grey-level pattern (see module docstring)."""

    def __init__(self, patch_size: int, min_std: float = 0.01):
        self.patch_size = patch_size
        self.min_std = min_std

    @property
    def dim(self) -> int:
        return self.patch_size * self.patch_size

    def compute_batch(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Describe a batch of patches.

        Args:
            patches: (N, s, s, 3) float array in [0, 1].

        Returns:
            (descriptors (N, s*s) float32, textured (N,) bool). Descriptors of
            flat patches (textured == False) are zero.
        """
        if patches.ndim != 4 or patches.shape[1:3] != (self.patch_size, self.patch_size):
            raise ValueError(f"expected {self.patch_size}x{self.patch_size} patches, got {patches.shape[1:3]}")
        gray = luminance(patches.astype(np.float32)).reshape(len(patches), self.dim)
        gray -= gray.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(gray, axis=1)
        textured = norm > self.min_std * np.sqrt(self.dim)
        gray[textured] /= norm[textured, None]
        gray[~textured] = 0.0
        return gray, textured
