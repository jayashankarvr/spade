"""Photometric-invariant projection (Schur-Weyl style).

Projects each patch onto the invariant subspace of the photometric Lie group
action. The action we factor out:

    p'(x) = a * p(x) + b * 1   (per-patch gain a > 0, bias b)
    + an achromatic shift along the (1,1,1) direction in RGB space

The orbit of any patch under this group is a 3-dimensional submanifold of
patch space. We project onto its orthogonal complement by:

    1. subtract per-patch mean        (removes global brightness / bias)
    2. subtract per-pixel achromatic   (removes (1,1,1)-direction shifts)
    3. L2-normalize                    (removes gain / contrast)

The result lies on the unit sphere of the photometric-invariant residual.
This is the canonical photometric-quotient embedding used by both the
diffusion-map manifold trainer and the runtime prior.
"""

from __future__ import annotations

import numpy as np


def photometric_normalize(patches: np.ndarray) -> np.ndarray:
    """Project a batch of patches onto the photometric-invariant subspace.

    Args:
        patches: (N, H, W, 3) float array, values in [0, 1] preferred.

    Returns:
        (N, H * W * 3) float32 unit-norm vectors.
    """
    if patches.ndim != 4 or patches.shape[-1] != 3:
        raise ValueError(f"expected (N, H, W, 3), got {patches.shape}")

    n, h, w, _ = patches.shape
    flat = patches.reshape(n, -1).astype(np.float32, copy=True)

    # 1. Remove per-patch mean (brightness / bias)
    flat -= flat.mean(axis=1, keepdims=True)

    # 2. Remove the achromatic (1,1,1) component per pixel
    rgb = flat.reshape(n, h * w, 3)
    rgb -= rgb.mean(axis=2, keepdims=True)
    flat = rgb.reshape(n, -1)

    # 3. L2-normalize (gain / contrast)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    np.maximum(norms, 1e-8, out=norms)
    flat /= norms
    return flat
