"""Turn a set of matches into a clean localization mask.

SPADE's dense 3x3 matching produces true matches concentrated on the (contiguous)
spliced region plus scattered false matches on coincidentally-similar background.
A spatial-density filter — keep the largest connected component of the matched
footprints — removes the scattered false positives while preserving the region.

Empirically this is the single biggest precision lever: on synthetic recolored
splices it lifts precision ~0.40 -> ~0.73 and IoU ~0.38 -> ~0.69 with negligible
recall loss (see BENCHMARKS.md). It operates in *source* (query) coordinates,
where the splice lives.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.ndimage import label


def match_footprint_mask(matches: Sequence, shape: Tuple[int, int]) -> np.ndarray:
    """Boolean mask (H, W) marking the source-patch footprint of each match."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for m in matches:
        x, y = int(m.source_coord[0]), int(m.source_coord[1])
        ps = int(m.patch_size)
        mask[max(0, y):min(h, y + ps), max(0, x):min(w, x + ps)] = True
    return mask


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected True component of a boolean mask."""
    if not mask.any():
        return mask
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, n = label(mask, structure=structure)
    if n <= 1:
        return mask
    # Bincount over labels (skip background label 0) and keep the biggest.
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = int(counts.argmax())
    return labeled == keep


def localize(
    matches: Sequence,
    shape: Tuple[int, int],
    largest_component: bool = True,
) -> np.ndarray:
    """Predicted localization mask (H, W) in source/query coordinates.

    Args:
        matches: matches with ``source_coord`` and ``patch_size``.
        shape: (height, width) of the query image.
        largest_component: keep only the largest connected component (default).
    """
    mask = match_footprint_mask(matches, shape)
    if largest_component:
        mask = largest_connected_component(mask)
    return mask


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Bounding box (x, y, w, h) of the True pixels, or (0,0,0,0) if empty."""
    if not mask.any():
        return (0, 0, 0, 0)
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
