"""Spatial aggregation and heatmap generation."""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy.ndimage import gaussian_filter, label


@dataclass
class Match:
    """A verified match between source and target patches."""
    source_coord: Tuple[int, int]
    target_coord: Tuple[int, int]
    patch_size: int
    probability: float
    image_id: str = ""


class SpatialAggregator:
    """
    Aggregates matches into forensic heatmaps.

    Combines match probabilities at each location with optional
    scale-based weighting (larger patches are more reliable).
    """

    def __init__(
        self,
        scale_weights: Optional[Dict[int, float]] = None,
        smoothing_sigma: float = 2.0,
    ):
        """
        Args:
            scale_weights: Weight per patch size (default: larger = higher weight)
            smoothing_sigma: Gaussian smoothing sigma for final heatmap
        """
        self.scale_weights = scale_weights or {3: 1.0, 4: 1.2, 5: 1.5, 6: 2.0}
        self.smoothing_sigma = smoothing_sigma

    def aggregate(
        self,
        matches: List[Match],
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Create aggregated heatmap from matches.

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
            x, y = match.target_coord
            if not (0 <= x < width and 0 <= y < height):
                continue

            weight = self.scale_weights.get(match.patch_size, 1.0)
            heatmap[y, x] += match.probability * weight
            counts[y, x] += weight

        # Normalize by counts
        mask = counts > 0
        heatmap[mask] /= counts[mask]

        # Apply Gaussian smoothing
        if self.smoothing_sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=self.smoothing_sigma)

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

    def find_regions(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.5,
        min_area: int = 9,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Find high-probability regions in heatmap.

        Args:
            heatmap: Aggregated heatmap (H, W)
            threshold: Minimum probability threshold
            min_area: Minimum region area in pixels

        Returns:
            List of bounding boxes (x, y, width, height)
        """
        binary = heatmap > threshold
        labeled, num_features = label(binary)

        regions = []
        for i in range(1, num_features + 1):
            ys, xs = np.where(labeled == i)
            if len(ys) < min_area:
                continue

            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())

            regions.append((
                x_min,
                y_min,
                x_max - x_min + 1,
                y_max - y_min + 1,
            ))

        return regions
