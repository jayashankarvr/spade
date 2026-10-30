"""Spatial aggregation and heatmap generation."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class Match:
    """A verified match between source and target patches."""
    source_coord: Tuple[int, int]
    target_coord: Tuple[int, int]   # full-resolution target coordinates
    patch_size: int                 # side of the source patch, in pixels
    probability: float
    image_id: str = ""
    # Recovered affine color transform (target = M @ source + b) from verification.
    # Optional so existing callers/tests that build Match directly keep working.
    color_M: Optional[np.ndarray] = None
    color_b: Optional[np.ndarray] = None
    # Pyramid level the target patch came from (1.0 = full resolution).
    target_scale: float = 1.0


class SpatialAggregator:
    """Turns matches into a target-image heatmap of where the copied pixels came from."""

    def __init__(self, smoothing_sigma: float = 2.0):
        """
        Args:
            smoothing_sigma: Gaussian smoothing sigma for final heatmap (0 = off)
        """
        self.smoothing_sigma = smoothing_sigma

    def aggregate(
        self,
        matches: List[Match],
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Mean match probability over each match's target footprint.

        Args:
            matches: List of Match objects
            target_shape: (height, width) of target image

        Returns:
            Heatmap array (H, W) normalized to [0, 1]
        """
        height, width = target_shape
        heatmap = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)

        for match in matches:
            x, y = int(match.target_coord[0]), int(match.target_coord[1])
            size = max(1, int(round(match.patch_size / match.target_scale)))
            x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + size), min(height, y + size)
            if x0 >= x1 or y0 >= y1:
                continue
            heatmap[y0:y1, x0:x1] += match.probability
            counts[y0:y1, x0:x1] += 1.0

        mask = counts > 0
        heatmap[mask] /= counts[mask]

        if self.smoothing_sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=self.smoothing_sigma)

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap
