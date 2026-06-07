"""Native-scale (effective-resolution) estimation and the scale-inconsistency cue.

Two related capabilities, both built on one idea: *how far can an image be
downsampled before it starts losing information?* Below its native resolution,
downsample+upsample can no longer reconstruct it (round-trip error rises); at or
above native, it's near-lossless. The knee in that error curve is the native scale.

1. ``native_scale_fraction`` - estimate an image's native resolution as a fraction
   of its current size (1.0 = already native; 0.5 = was ~2x upscaled). Useful for
   normalizing two images to a common detail scale before matching.

2. ``scale_inconsistency_map`` - a sliding-window native-scale map. A spliced region
   that was resized before pasting has a *different* native scale than the host, so
   it stands out - an independent forensic cue (cf. resampling detection), complementary
   to SPADE's color-transform evidence.

Note: the knee is sharp for native and moderate upscaling (~2x), softer for heavy
upscaling (~3x+), where the absolute estimate under-reads but the *relative*
difference that the cue relies on stays clear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    img = image.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def native_scale_fraction(
    image: np.ndarray,
    min_fraction: float = 0.3,
    steps: int = 15,
    tol: float = 0.08,
) -> float:
    """Estimate native resolution as a fraction of the current size, in (0, 1].

    1.0 means the image is already at its native resolution; 0.5 means it looks
    like it was upscaled ~2x (downsampling by half loses nothing).
    """
    arr = _as_uint8(image)
    h, w = arr.shape[:2]
    fractions = np.linspace(1.0, min_fraction, steps)

    errs = []
    for f in fractions:
        sw, sh = max(8, int(round(w * f))), max(8, int(round(h * f)))
        small = Image.fromarray(arr).resize((sw, sh), Image.LANCZOS)
        back = np.asarray(small.resize((w, h), Image.BICUBIC))
        errs.append(float(np.mean((arr.astype(np.float32) - back.astype(np.float32)) ** 2)))

    errs = np.array(errs)
    floor = errs[fractions >= 0.95].mean() if (fractions >= 0.95).any() else errs[0]
    thr = floor + tol * (errs.max() - floor) + 1.0  # small tolerance above the lossless floor
    ok = fractions[errs <= thr]
    return float(ok.min()) if len(ok) else 1.0


def scale_inconsistency_map(
    image: np.ndarray,
    window: int = 64,
    stride: int = 32,
    **kwargs,
) -> np.ndarray:
    """Sliding-window native-scale map (lower = more upscaled/blurry).

    Returns a 2D array of native-scale fractions, one per window. A splice that
    was resized shows up as a block whose value differs from the host.
    """
    h, w = image.shape[:2]
    ys = list(range(0, max(1, h - window + 1), stride))
    xs = list(range(0, max(1, w - window + 1), stride))
    grid = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            grid[i, j] = native_scale_fraction(image[y:y + window, x:x + window], **kwargs)
    return grid


@dataclass
class ScaleInconsistency:
    """Image-level resize/resampling cue.

    ``score`` is ~0 for a uniformly-native image and grows when one region's
    native scale differs from the rest (a splice that was resized before pasting).
    ``anomaly_bbox`` is the window with the lowest native scale - the likely
    resized region.
    """
    score: float
    min_fraction: float
    median_fraction: float
    anomaly_bbox: Tuple[int, int, int, int]   # (x, y, w, h)
    window: int
    stride: int


def scale_inconsistency(
    image: np.ndarray, window: int = 64, stride: int = 32, **kwargs
) -> ScaleInconsistency:
    """Compute the resize-inconsistency cue and locate the most anomalous window."""
    h, w = image.shape[:2]
    ys = list(range(0, max(1, h - window + 1), stride))
    xs = list(range(0, max(1, w - window + 1), stride))
    grid = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            grid[i, j] = native_scale_fraction(image[y:y + window, x:x + window], **kwargs)

    vals = grid[np.isfinite(grid)]
    if vals.size < 2:
        bw, bh = min(window, w), min(window, h)
        return ScaleInconsistency(0.0, 1.0, 1.0, (0, 0, bw, bh), window, stride)

    i, j = np.unravel_index(np.nanargmin(grid), grid.shape)
    x0, y0 = xs[j], ys[i]
    return ScaleInconsistency(
        score=float(np.median(vals) - vals.min()),
        min_fraction=float(vals.min()),
        median_fraction=float(np.median(vals)),
        anomaly_bbox=(x0, y0, min(window, w - x0), min(window, h - y0)),
        window=window,
        stride=stride,
    )


def scale_inconsistency_score(image: np.ndarray, window: int = 64, stride: int = 32, **kwargs) -> float:
    """Single image-level score: how much the native scale varies across the image.

    ~0 for a uniformly-native image; larger when some region was resized relative
    to the rest (a splice cue). Computed as (median - min) of the window map.
    """
    return scale_inconsistency(image, window=window, stride=stride, **kwargs).score
