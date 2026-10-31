"""Learned linear descriptor: the luminance pattern, projected where edits matter least.

A fixed matrix fitted offline (``python -m spade.bench.train_descriptor``) maps
the normalized luminance pattern to a few dimensions that tell patches apart
while being as insensitive as possible to color grades and JPEG. It's plain
linear algebra at run time - no learning framework needed - and the weights
ship with the package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from spade.descriptors.core import LuminanceDescriptor

DEFAULT_DIMS = 8


def data_path(patch_size: int) -> Path:
    return Path(__file__).parent / "data" / f"linear_{patch_size}x{patch_size}.npz"


def available_sizes():
    folder = Path(__file__).parent / "data"
    return sorted(int(p.stem.split("_")[1].split("x")[0]) for p in folder.glob("linear_*x*.npz"))


class LearnedDescriptor:
    """Projection of ``LuminanceDescriptor`` output; same interface."""

    def __init__(self, patch_size: int):
        path = data_path(patch_size)
        if not path.exists():
            raise ValueError(
                f"no learned descriptor for {patch_size}x{patch_size} patches (have: {available_sizes()}); "
                "train one with python -m spade.bench.train_descriptor"
            )
        data = np.load(path)
        dims = int(data["dims"])
        self.patch_size = patch_size
        self.projection = data["projection"][:, :dims].astype(np.float32)
        # Squared-distance threshold matched to the luminance descriptor's default.
        self.distance_threshold = float(data["threshold"])
        self._luminance = LuminanceDescriptor(patch_size)

    @property
    def dim(self) -> int:
        return self.projection.shape[1]

    def compute_batch(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pattern, textured = self._luminance.compute_batch(patches)
        out = pattern @ self.projection
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        out = np.where(textured[:, None], out / np.maximum(norm, 1e-9), 0.0).astype(np.float32)
        return out, textured
