"""Group matches into regions that share one translation."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
    # Rotation-invariant mode: target = R(rotation) @ source + offset, about patch centres.
    rotation: float = 0.0
    # The source side is a horizontal mirror image of the target side.
    mirrored: bool = False
    # Scale the source was searched at (1.0 unless query_scales found it).
    source_scale: float = 1.0
    # Region hypotheses tried in the search that produced this region; NFA is
    # n_tests times the region's own p-value (used for FDR across searches).
    n_tests: Optional[float] = None

    @property
    def log10_p(self) -> Optional[float]:
        """log10 of the region's own p-value (its NFA without the multiple-testing factor)."""
        if self.log10_nfa is None or not self.n_tests:
            return None
        return self.log10_nfa - float(np.log10(self.n_tests))

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
        group_of: Dict[Tuple[str, float], int] = {}
        groups = [group_of.setdefault((m.image_id, getattr(m, "target_scale", 1.0)), len(group_of)) for m in matches]
        offsets = np.array([match_offset(m) for m in matches], dtype=np.int64)
        regions = [
            make_region(peak, [matches[i] for i in rows], matches[rows[0]].image_id)
            for peak, rows in cluster_offsets(offsets, groups, self.offset_tolerance, self.min_cluster_size)
        ]
        regions.sort(key=lambda r: r.confidence, reverse=True)
        return regions


def cluster_offsets(
    offsets: np.ndarray, groups: Sequence[int], tol: int, min_size: int,
    scores: Optional[np.ndarray] = None, limit: Optional[int] = None,
) -> List[Tuple[Tuple[int, int], np.ndarray]]:
    """Peak clustering of match offsets, separately within each group.

    Within a group, distinct offsets are visited most common first (ties by
    offset). Each one not yet claimed takes every unclaimed offset within
    ``tol`` of it on both axes, provided they hold at least ``min_size``
    matches between them.

    Works on arrays: offsets become integer cell keys, their neighbours are
    found with one searchsorted, and offsets whose whole window holds fewer
    than ``min_size`` matches are skipped before the (cheap) greedy pass.

    Returns ``(peak offset, member rows)`` per cluster, groups in order of
    first appearance. Members are listed window cell by cell (dx outer, dy
    inner), input order within a cell.

    With ``limit``, returns only the ``limit`` biggest clusters, biggest first,
    equal sizes by mean ``scores``. Most clusters are a handful of chance
    matches, and this skips building their member lists.
    """
    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
    groups = np.asarray(groups)
    if len(offsets) < min_size:
        return []
    _, first, group_idx = np.unique(groups, return_index=True, return_inverse=True)
    d = np.arange(-tol, tol + 1)
    found = []   # (size, score sum, peak, cell list, per-group cell -> rows lookup)
    for g in np.argsort(first):
        rows = np.nonzero(group_idx == g)[0]
        if len(rows) < min_size:
            continue
        off = offsets[rows]
        # Pad by tol on each side so a neighbour's dy never wraps into another dx.
        lo = off.min(axis=0) - tol
        span_y = int(off[:, 1].max() - lo[1]) + tol + 1
        keys = (off[:, 0] - lo[0]) * span_y + (off[:, 1] - lo[1])
        cells, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        wanted = cells[:, None] + (d[:, None] * span_y + d[None, :]).ravel()[None, :]
        pos = np.minimum(np.searchsorted(cells, wanted), len(cells) - 1)
        nbr = np.where(cells[pos] == wanted, pos, -1)
        support = np.where(nbr >= 0, counts[np.maximum(nbr, 0)], 0).sum(axis=1)

        order = np.lexsort((cells, -counts))
        order = order[support[order] >= min_size]
        if len(order) == 0:
            continue
        nbr_list, count_list = nbr.tolist(), counts.tolist()
        claimed = [False] * len(cells)
        by_cell = rows[np.argsort(inverse, kind="stable")]
        start = np.concatenate([[0], np.cumsum(counts)]).tolist()
        cell_score = (np.bincount(inverse.reshape(-1), weights=np.asarray(scores)[rows]).tolist()
                      if scores is not None else None)
        lookup = (by_cell, start)
        for peak in order.tolist():
            if claimed[peak]:
                continue
            window = [c for c in nbr_list[peak] if c >= 0 and not claimed[c]]
            if sum(count_list[c] for c in window) < min_size:
                continue
            for c in window:
                claimed[c] = True
            px, py = divmod(int(cells[peak]), span_y)
            size = sum(count_list[c] for c in window)
            score = sum(cell_score[c] for c in window) if cell_score is not None else 0.0
            found.append((size, score, (px + int(lo[0]), py + int(lo[1])), window, lookup))

    if limit is not None:
        rank = sorted(range(len(found)), key=lambda i: (-found[i][0], -found[i][1] / found[i][0]))
        found = [found[i] for i in rank[:limit]]
    return [
        (peak, np.concatenate([by_cell[start[c]:start[c + 1]] for c in window]))
        for _, _, peak, window, (by_cell, start) in found
    ]


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _rotate(points: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * points[:, 0] - s * points[:, 1], s * points[:, 0] + c * points[:, 1]], axis=1)


def find_rigid_regions(
    matches: List,
    offset_tolerance: int = 2,
    min_cluster_size: int = 3,
    angle_tolerance: float = np.radians(15),
    seeds: int = 64,
    max_regions: int = 100,
) -> List[CoherentRegion]:
    """Group matches that agree on one rotation + translation (rotation-invariant mode).

    Every match already implies a rotation (its ``rotation``) and, with it, a
    translation, so one match is a full hypothesis. Repeatedly: score a few
    hundred matches as seeds by how many others agree (angle within
    ``angle_tolerance``, predicted target centre within the offset tolerance,
    loosened for the seed's own angle error), keep the best, refine its angle
    by least squares over the agreeing centres (better than any single
    patch's angle), take the final inliers, and remove them.

    The region's ``offset`` is the translation and ``rotation`` the angle,
    with target centre = R(rotation) @ source centre + offset. Stops after
    ``max_regions``: past that point, what's left are chance clusters of a
    few matches that couldn't reach significance anyway.
    """
    rng = np.random.RandomState(0)
    tol = max(offset_tolerance, 1)
    regions = []
    by_image: Dict[str, List] = defaultdict(list)
    for m in matches:
        by_image[m.image_id].append(m)

    for image_id, group in by_image.items():
        half = (group[0].patch_size - 1) / 2.0
        src = np.array([m.source_coord for m in group], dtype=np.float64) + half
        tgt = np.array([m.target_coord for m in group], dtype=np.float64) + half
        rot = np.array([m.rotation for m in group], dtype=np.float64)
        alive = np.ones(len(group), dtype=bool)
        # Matches sorted by angle (with a wrapped copy at both ends), so each seed
        # only looks at the ~1/12 of matches whose angle can agree with it.
        order = np.argsort(rot)
        ext_rot = np.concatenate([rot[order] - 2 * np.pi, rot[order], rot[order] + 2 * np.pi])
        ext_idx = np.concatenate([order, order, order])

        def agreeing(seed: int, theta: float, shift: np.ndarray, slack: float) -> np.ndarray:
            lo, hi = np.searchsorted(ext_rot, [theta - angle_tolerance, theta + angle_tolerance + 1e-12])
            # A window narrower than a full turn never visits a match twice.
            cand = ext_idx[lo:hi]
            cand = cand[alive[cand]]
            pred = _rotate(src[cand], theta) + shift
            return cand[np.abs(pred - tgt[cand]).max(axis=1) <= slack]

        while alive.sum() >= min_cluster_size and len(regions) < max_regions:
            remaining = np.nonzero(alive)[0]
            best = None
            for seed in rng.choice(remaining, min(seeds, len(remaining)), replace=False):
                theta = rot[seed]
                shift = tgt[seed] - _rotate(src[seed:seed + 1], theta)[0]
                members = agreeing(seed, theta, shift, 2 * tol + 2)
                if best is None or len(members) > len(best):
                    best = members
            if best is None or len(best) < min_cluster_size:
                break
            # Least-squares rotation of the agreeing centres, then the final inliers.
            a = src[best] - src[best].mean(axis=0)
            b = tgt[best] - tgt[best].mean(axis=0)
            theta = float(np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum()))
            shift = tgt[best].mean(axis=0) - _rotate(src[best].mean(axis=0)[None, :], theta)[0]
            inliers = agreeing(-1, theta, shift, tol)
            if len(inliers) < min_cluster_size:
                alive[best] = False
                continue
            region = make_region((int(round(shift[0])), int(round(shift[1]))), [group[i] for i in inliers], image_id)
            region.rotation = theta
            regions.append(region)
            alive[inliers] = False

    regions.sort(key=lambda r: r.confidence, reverse=True)
    return regions


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
