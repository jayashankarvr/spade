"""Group matches into regions that share one translation."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CoherentRegion:
    """A cluster of matches that agree on one (dx, dy) translation."""
    offset: Tuple[int, int]          # (dx, dy), in the matched target level's pixels
    matches: List["Match"]           # Matches in this cluster
    source_bbox: Tuple[int, int, int, int]  # (x, y, w, h) in source
    target_bbox: Tuple[int, int, int, int]  # (x, y, w, h) in target
    confidence: float                # mean probability * log2(size + 1), a ranking heuristic
    image_id: str = ""
    # Filled in by region-level verification (ForensicsEngine.match): one color
    # transform fitted over every pixel in the region, the a-contrario
    # significance of the patches that agree with it, and the threshold that
    # significance is judged against (log10 of Config.max_nfa).
    color_M: Optional[np.ndarray] = None
    color_b: Optional[np.ndarray] = None
    log10_nfa: Optional[float] = None
    max_log10_nfa: float = 0.0

    @property
    def significant(self) -> bool:
        """True when the region's NFA is below the threshold (NFA < 1 by default)."""
        return self.log10_nfa is not None and self.log10_nfa < self.max_log10_nfa


def match_offset(match) -> Tuple[int, int]:
    """Translation from source to target, measured at the target's pyramid level.

    A copy that was resized shows a constant offset only in the level where the
    scale matches, so offsets are taken there rather than in full-res coords.
    """
    scale = getattr(match, "target_scale", 1.0)
    tx = int(round(match.target_coord[0] * scale))
    ty = int(round(match.target_coord[1] * scale))
    return tx - match.source_coord[0], ty - match.source_coord[1]


class SpatialCoherenceVerifier:
    """
    Groups matches by translation consistency.

    A copied region produces many matches with the same offset; false positives
    scatter over random offsets. Matches are grouped per target image and scale,
    then clustered around offset peaks: the most common offset claims every
    match within ``offset_tolerance`` of it, then the next one, and so on. A
    cluster can't grow beyond its peak's tolerance window, so slowly drifting
    offsets never chain into one giant cluster.
    """

    def __init__(self, offset_tolerance: int = 2, min_cluster_size: int = 3):
        """
        Args:
            offset_tolerance: Max per-axis distance from a cluster's peak offset
            min_cluster_size: Minimum matches to form a cluster
        """
        self.offset_tolerance = offset_tolerance
        self.min_cluster_size = min_cluster_size

    def find_coherent_regions(self, matches: List) -> List[CoherentRegion]:
        """Cluster matches; returns regions sorted by confidence (highest first)."""
        if len(matches) < self.min_cluster_size:
            return []

        groups: Dict[Tuple[str, float], List] = defaultdict(list)
        for m in matches:
            groups[(m.image_id, getattr(m, "target_scale", 1.0))].append(m)

        regions = []
        for (image_id, _), group in groups.items():
            for offset, cluster in self._cluster_by_offset(group):
                regions.append(make_region(offset, cluster, image_id))

        regions.sort(key=lambda r: r.confidence, reverse=True)
        return regions

    def _cluster_by_offset(self, matches: List) -> List[Tuple[Tuple[int, int], List]]:
        by_offset: Dict[Tuple[int, int], List] = defaultdict(list)
        for m in matches:
            by_offset[match_offset(m)].append(m)

        counts = Counter({o: len(ms) for o, ms in by_offset.items()})
        tol = self.offset_tolerance
        claimed = set()
        clusters = []
        # Ties broken by offset so the result doesn't depend on input order.
        for peak, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if peak in claimed:
                continue
            window = [
                (peak[0] + dx, peak[1] + dy)
                for dx in range(-tol, tol + 1)
                for dy in range(-tol, tol + 1)
            ]
            window = [o for o in window if o in by_offset and o not in claimed]
            members = [m for o in window for m in by_offset[o]]
            if len(members) < self.min_cluster_size:
                continue
            claimed.update(window)
            clusters.append((peak, members))
        return clusters


def make_region(offset: Tuple[int, int], cluster: List, image_id: str = "") -> CoherentRegion:
    """Build a region (bounding boxes, confidence) from a list of matches."""
    sx = [m.source_coord[0] for m in cluster]
    sy = [m.source_coord[1] for m in cluster]
    tx = [m.target_coord[0] for m in cluster]
    ty = [m.target_coord[1] for m in cluster]
    size = max(m.patch_size for m in cluster)
    # A patch matched at pyramid scale s covers patch_size / s full-res target pixels.
    target_size = int(round(size / getattr(cluster[0], "target_scale", 1.0)))
    avg_prob = float(np.mean([m.probability for m in cluster]))
    return CoherentRegion(
        offset=offset,
        matches=cluster,
        source_bbox=(min(sx), min(sy), max(sx) - min(sx) + size, max(sy) - min(sy) + size),
        target_bbox=(min(tx), min(ty), max(tx) - min(tx) + target_size, max(ty) - min(ty) + target_size),
        confidence=avg_prob * float(np.log2(len(cluster) + 1)),
        image_id=image_id,
    )
