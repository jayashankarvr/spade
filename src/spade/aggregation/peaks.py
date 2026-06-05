"""Persistent-homology peak detection on the Hough accumulator.

Every local maximum in the accumulator corresponds to a candidate forgery
hypothesis. Rather than threshold the accumulator at some fixed value, we
compute the *persistence* of each peak under a superlevel-set sweep:

    threshold t goes from +inf down to -inf
    peaks are born when their cell first exceeds t
    peaks die when they merge into an older (taller) peak (elder rule)
    persistence = birth_height - death_height

Persistence is a parameter-free measure of how confidently a peak stands
above the noise floor. The bottleneck-stability theorem (Cohen-Steiner)
guarantees small input perturbations cause small persistence-diagram
perturbations. Stable, principled, no thresholds.

We project the 4-D Hough accumulator down to its (dx, dy) plane (summing
over rotation and scale) for peak detection, since 2-D persistence on a
regular grid has a clean implementation. Each persistent (dx, dy) peak
then has its rotation/scale estimated from the contributing 4-D cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from spade.aggregation.hough import HoughAccumulator4D


@dataclass
class PeakHypothesis:
    """A persistent peak in the (dx, dy) Hough accumulator."""
    dx: float
    dy: float
    rotation_rad: float
    scale: float
    weight: float           # peak height (accumulated log-BF)
    persistence: float      # superlevel-set persistence
    votes: int              # contributing matches across all 4-D cells


class _UF:
    """Union-find with elder rule (used by both persistence and Hough peaks)."""

    __slots__ = ("parent", "birth_height")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.birth_height = [-float("inf")] * n

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union_elder(self, i: int, j: int) -> Tuple[int, int]:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return ri, ri
        # Elder = born earlier under superlevel sweep = higher birth_height
        if self.birth_height[ri] >= self.birth_height[rj]:
            self.parent[rj] = ri
            return ri, rj
        else:
            self.parent[ri] = rj
            return rj, ri


def superlevel_persistence_2d(grid: np.ndarray) -> List[Tuple[Tuple[int, int], float, float]]:
    """0-D persistence of the superlevel-set filtration of a 2-D grid.

    Returns:
        List of (peak_xy, birth_height, persistence) tuples. The single
        infinite-persistence peak (the global max) gets persistence equal to
        (max - min) so all values are finite.
    """
    h, w = grid.shape
    n = h * w
    flat = grid.ravel()
    # Sweep highest -> lowest
    order = np.argsort(-flat, kind="stable")

    uf = _UF(n)
    activated = np.zeros(n, dtype=bool)
    peak_for_root: dict[int, Tuple[int, int]] = {}
    pairs: List[Tuple[Tuple[int, int], float, float]] = []

    def neighbors(i: int):
        y, x = divmod(i, w)
        if x > 0:    yield i - 1
        if x < w-1:  yield i + 1
        if y > 0:    yield i - w
        if y < h-1:  yield i + w

    for idx in order:
        v = float(flat[idx])
        uf.birth_height[idx] = v
        activated[idx] = True

        # Each newly-activated pixel begins a new peak at its own location;
        # neighbors already activated will merge into one of these.
        peak_for_root[idx] = divmod(idx, w)

        for nb in neighbors(idx):
            if not activated[nb]:
                continue
            ri, rj = uf.find(idx), uf.find(nb)
            if ri == rj:
                continue
            survivor, dead = uf.union_elder(ri, rj)
            # The dead peak is born at its own birth_height, dies at v
            persistence = uf.birth_height[dead] - v
            pairs.append((peak_for_root[dead], uf.birth_height[dead], persistence))
            # The merged component's representative is now `survivor`; bring
            # peak metadata along.
            peak_for_root[survivor] = peak_for_root[survivor]

    # The global max is the one essential class; treat its persistence as full range.
    if n > 0:
        gmax = float(flat.max())
        gmin = float(flat.min())
        argmax = int(np.argmax(flat))
        pairs.append((divmod(argmax, w), gmax, max(gmax - gmin, 1e-9)))

    # Sort by persistence descending - most prominent peaks first
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


class PersistentPeakDetector:
    """Detect peaks in a HoughAccumulator4D via 2-D persistent homology."""

    def __init__(self, persistence_floor: float = 0.0, max_peaks: int = 50):
        """
        Args:
            persistence_floor: discard peaks below this persistence (in log-BF nats).
                Default 0.0 keeps all peaks - the persistence ranking itself is
                the parameter-free confidence and downstream consumers can pick.
            max_peaks: cap on returned peaks for downstream BP cost.
        """
        self.persistence_floor = persistence_floor
        self.max_peaks = max_peaks

    def detect(self, accumulator: HoughAccumulator4D) -> List[PeakHypothesis]:
        if len(accumulator) == 0:
            return []
        (x_lo, x_hi), (y_lo, y_hi) = accumulator.cell_index_bounds()
        # Pad to give peaks at the boundary room to merge cleanly
        x_lo, x_hi = x_lo - 1, x_hi + 1
        y_lo, y_hi = y_lo - 1, y_hi + 1
        dense = accumulator.to_dense_dxy((x_lo, x_hi), (y_lo, y_hi))

        diagram = superlevel_persistence_2d(dense)

        # For each persistent (dx, dy) peak, recover the dominant rotation/scale
        # cell among the 4-D cells projecting to that (dx, dy).
        cells_by_xy: dict[Tuple[int, int], List[Tuple[Tuple[int, int, int, int], float, int]]] = {}
        for cell, weight, count in accumulator.items():
            xy = (cell[0], cell[1])
            cells_by_xy.setdefault(xy, []).append((cell, weight, count))

        out: List[PeakHypothesis] = []
        for (peak_y, peak_x), birth, persistence in diagram:
            if persistence < self.persistence_floor:
                continue
            ix = peak_x + x_lo
            iy = peak_y + y_lo
            entries = cells_by_xy.get((ix, iy))
            if not entries:
                continue
            # Dominant 4-D cell at this (dx, dy)
            entries.sort(key=lambda e: e[1], reverse=True)
            best_cell, best_weight, best_count = entries[0]
            transform = accumulator._transform_for_cell(best_cell)
            total_votes = sum(c for _, _, c in entries)
            out.append(PeakHypothesis(
                dx=transform[0],
                dy=transform[1],
                rotation_rad=transform[2],
                scale=transform[3],
                weight=birth,
                persistence=float(persistence),
                votes=int(total_votes),
            ))
            if len(out) >= self.max_peaks:
                break

        return out
