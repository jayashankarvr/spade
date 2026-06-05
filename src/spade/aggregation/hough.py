"""4-D Hough accumulator for forensic match aggregation.

Each candidate match casts a vote into a (dx, dy, dtheta, ds) accumulator,
weighted by its log Bayes factor. Discretization:

    dx, dy:   2-pixel cells   (integer cell index = round(d / 2))
    dtheta:   ~5-degree cells (integer cell index = round(theta / 5))
    ds:       log-scale steps (cell index = round(log2(s) * 4))

Stored as a sparse dict-of-counts because the populated cells are
typically << 10^7 even for high-resolution images.

Multi-scale matches (3x3 .. 6x6) all vote into the same accumulator,
weighted equally on the log-BF scale - their information is naturally
combined by addition in log-likelihood space.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List
import math
import numpy as np


@dataclass(frozen=True)
class HoughConfig:
    dxy_cell: float = 2.0          # pixels per dx/dy cell
    dtheta_cell_deg: float = 5.0   # degrees per dtheta cell
    ds_log2_cell: float = 0.25     # log2 scale steps (4 cells per octave)


@dataclass(frozen=True)
class HoughCandidate:
    """A single source-target patch correspondence to vote into the accumulator."""
    source_xy: Tuple[float, float]
    target_xy: Tuple[float, float]
    log_bf: float                          # vote weight in nats
    patch_size: int = 3
    rotation_rad: float = 0.0
    scale: float = 1.0


@dataclass
class HoughPeak:
    cell: Tuple[int, int, int, int]        # (idx_dx, idx_dy, idx_dtheta, idx_ds)
    transform: Tuple[float, float, float, float]   # (dx, dy, dtheta_rad, ds)
    weight: float                          # accumulated log-BF
    votes: int                             # number of contributing matches


class HoughAccumulator4D:
    """Sparse 4-D Hough accumulator. Cells -> aggregated weight + count."""

    def __init__(self, cfg: HoughConfig | None = None):
        self.cfg = cfg or HoughConfig()
        self._weights: Dict[Tuple[int, int, int, int], float] = defaultdict(float)
        self._counts: Dict[Tuple[int, int, int, int], int] = defaultdict(int)

    def reset(self) -> None:
        self._weights.clear()
        self._counts.clear()

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def vote(self, candidate: HoughCandidate) -> None:
        cell = self._cell_for(candidate)
        # Use a small floor so bad matches still register but with low weight.
        # Negative log-BFs are clipped to 0 (votes only contribute positive evidence).
        w = max(float(candidate.log_bf), 0.0)
        if w == 0.0:
            return
        self._weights[cell] += w
        self._counts[cell] += 1

    def vote_many(self, candidates: Iterable[HoughCandidate]) -> None:
        for c in candidates:
            self.vote(c)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._weights)

    def items(self):
        for cell, w in self._weights.items():
            yield cell, w, self._counts[cell]

    def top_k(self, k: int = 10) -> List[HoughPeak]:
        """Largest-weight cells. Cheap; useful as a baseline before peak detection."""
        ordered = sorted(self._weights.items(), key=lambda kv: kv[1], reverse=True)
        out: List[HoughPeak] = []
        for cell, weight in ordered[:k]:
            out.append(HoughPeak(
                cell=cell,
                transform=self._transform_for_cell(cell),
                weight=float(weight),
                votes=self._counts[cell],
            ))
        return out

    def to_dense_dxy(self, dx_range: Tuple[int, int], dy_range: Tuple[int, int]) -> np.ndarray:
        """Project the 4-D accumulator down to a dense 2-D (dx, dy) heatmap.

        Useful for the persistent-homology peak detector, which works on a
        regular grid. Other dimensions (dtheta, ds) are summed out.
        """
        x_lo, x_hi = dx_range
        y_lo, y_hi = dy_range
        h = y_hi - y_lo + 1
        w = x_hi - x_lo + 1
        dense = np.zeros((h, w), dtype=np.float32)
        for (ix, iy, _, _), weight in self._weights.items():
            if x_lo <= ix <= x_hi and y_lo <= iy <= y_hi:
                dense[iy - y_lo, ix - x_lo] += float(weight)
        return dense

    def cell_index_bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """(dx range, dy range) over all populated cells."""
        if not self._weights:
            return (0, 0), (0, 0)
        xs = [c[0] for c in self._weights]
        ys = [c[1] for c in self._weights]
        return (min(xs), max(xs)), (min(ys), max(ys))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cell_for(self, c: HoughCandidate) -> Tuple[int, int, int, int]:
        dx = c.target_xy[0] - c.source_xy[0]
        dy = c.target_xy[1] - c.source_xy[1]
        ix = int(round(dx / self.cfg.dxy_cell))
        iy = int(round(dy / self.cfg.dxy_cell))
        itheta = int(round(math.degrees(c.rotation_rad) / self.cfg.dtheta_cell_deg))
        s = max(c.scale, 1e-6)
        is_ = int(round(math.log2(s) / self.cfg.ds_log2_cell))
        return ix, iy, itheta, is_

    def _transform_for_cell(self, cell: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        ix, iy, itheta, is_ = cell
        dx = ix * self.cfg.dxy_cell
        dy = iy * self.cfg.dxy_cell
        dtheta = math.radians(itheta * self.cfg.dtheta_cell_deg)
        ds = 2.0 ** (is_ * self.cfg.ds_log2_cell)
        return dx, dy, dtheta, ds
