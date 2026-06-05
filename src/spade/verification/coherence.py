"""Spatial coherence verification for match clustering."""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict


@dataclass
class CoherentRegion:
    """A cluster of spatially consistent matches."""
    offset: Tuple[int, int]          # (dx, dy) translation vector
    matches: List["Match"]           # Matches in this cluster
    source_bbox: Tuple[int, int, int, int]  # (x, y, w, h) in source
    target_bbox: Tuple[int, int, int, int]  # (x, y, w, h) in target
    confidence: float                # Aggregate confidence score


class SpatialCoherenceVerifier:
    """
    Groups matches by translation consistency.

    Matches that share the same offset (dx, dy) between source and target
    are likely part of the same copied region. Random false positives
    have random offsets and won't cluster.
    """

    def __init__(
        self,
        offset_tolerance: int = 2,
        min_cluster_size: int = 3,
        min_cluster_probability: float = 0.5,
    ):
        """
        Args:
            offset_tolerance: Max difference in offset to consider matches consistent
            min_cluster_size: Minimum matches to form a valid cluster
            min_cluster_probability: Minimum average probability for cluster
        """
        self.offset_tolerance = offset_tolerance
        self.min_cluster_size = min_cluster_size
        self.min_cluster_probability = min_cluster_probability

    def find_coherent_regions(self, matches: List) -> List[CoherentRegion]:
        """
        Group matches into spatially coherent regions.

        Args:
            matches: List of Match objects with source_coord, target_coord, probability

        Returns:
            List of CoherentRegion, sorted by confidence (highest first)
        """
        if len(matches) < self.min_cluster_size:
            return []

        # Compute offset for each match
        match_offsets = []
        for m in matches:
            dx = m.target_coord[0] - m.source_coord[0]
            dy = m.target_coord[1] - m.source_coord[1]
            match_offsets.append((m, (dx, dy)))

        # Cluster by offset using greedy approach
        clusters = self._cluster_by_offset(match_offsets)

        # Convert to CoherentRegion objects
        regions = []
        for offset, cluster_matches in clusters.items():
            if len(cluster_matches) < self.min_cluster_size:
                continue

            avg_prob = np.mean([m.probability for m in cluster_matches])
            if avg_prob < self.min_cluster_probability:
                continue

            # Compute bounding boxes
            source_xs = [m.source_coord[0] for m in cluster_matches]
            source_ys = [m.source_coord[1] for m in cluster_matches]
            target_xs = [m.target_coord[0] for m in cluster_matches]
            target_ys = [m.target_coord[1] for m in cluster_matches]

            # Account for patch size in bbox
            patch_size = cluster_matches[0].patch_size if hasattr(cluster_matches[0], 'patch_size') else 3

            source_bbox = (
                min(source_xs),
                min(source_ys),
                max(source_xs) - min(source_xs) + patch_size,
                max(source_ys) - min(source_ys) + patch_size,
            )
            target_bbox = (
                min(target_xs),
                min(target_ys),
                max(target_xs) - min(target_xs) + patch_size,
                max(target_ys) - min(target_ys) + patch_size,
            )

            # Confidence combines cluster size and probability
            confidence = avg_prob * np.log2(len(cluster_matches) + 1)

            regions.append(CoherentRegion(
                offset=offset,
                matches=cluster_matches,
                source_bbox=source_bbox,
                target_bbox=target_bbox,
                confidence=confidence,
            ))

        # Sort by confidence
        regions.sort(key=lambda r: r.confidence, reverse=True)
        return regions

    def _cluster_by_offset(
        self,
        match_offsets: List[Tuple],
    ) -> Dict[Tuple[int, int], List]:
        """
        Cluster matches by similar offsets using grid-based bucketing.

        O(n) complexity - assigns each match to a grid cell based on tolerance.
        """
        if not match_offsets:
            return {}

        # Grid-based clustering: assign to cell, then merge adjacent cells
        clusters = defaultdict(list)
        tol = max(1, self.offset_tolerance)

        for match, (dx, dy) in match_offsets:
            # Quantize to grid cell
            cell = (dx // tol, dy // tol)
            clusters[cell].append(match)

        # Merge adjacent cells into final clusters keyed by canonical offset
        final_clusters: Dict[Tuple[int, int], List] = defaultdict(list)
        processed = set()

        for cell in clusters:
            if cell in processed:
                continue

            # Collect this cell and all adjacent cells
            merged = []
            cells_to_merge = [cell]
            while cells_to_merge:
                c = cells_to_merge.pop()
                if c in processed:
                    continue
                if c not in clusters:
                    continue
                processed.add(c)
                merged.extend(clusters[c])

                # Check 8-connected neighbors
                cx, cy = c
                for nx in (cx - 1, cx, cx + 1):
                    for ny in (cy - 1, cy, cy + 1):
                        neighbor = (nx, ny)
                        if neighbor not in processed and neighbor in clusters:
                            cells_to_merge.append(neighbor)

            if merged:
                # Use cell center as canonical offset
                canonical = (cell[0] * tol, cell[1] * tol)
                final_clusters[canonical].extend(merged)

        return dict(final_clusters)
