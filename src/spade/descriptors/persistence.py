"""Sublevel-set persistence diagram descriptor for tiny patches.

Theory
------
For a grayscale function f on a small grid, sweep a threshold t from -inf to
+inf and watch the connected components of {f <= t} appear and merge. Each
component has a (birth, death) pair. The multiset of these pairs is the
0-dimensional persistence diagram. Cohen-Steiner et al (2007) proved the
bottleneck distance between two diagrams is bounded by ||f - g||_inf, so the
descriptor is provably stable to noise and exactly invariant to monotone
intensity remappings (gamma, brightness, contrast, any tone curve).

We implement sublevel-set persistence directly via a union-find over pixels
sorted by intensity (the standard elder-rule construction). For 3x3..6x6
grids this is O(n alpha(n)) with n <= 36 - effectively free.

The diagram is then vectorized to a fixed-length descriptor by binning into a
"persistence image" (Adams et al, 2017): each (b, d) point becomes a
Gaussian blob with weight d - b in a 2-D image, which is then flattened.
This yields a Hilbert-space embedding stable in the bottleneck metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from spade.descriptors.core import DescriptorStrategy, _to_gray, _normalize


@dataclass(frozen=True)
class PersistenceImageConfig:
    """Persistence-image vectorization parameters."""
    resolution: int = 8           # output grid is resolution x resolution
    sigma: float = 0.05           # Gaussian bandwidth on the (birth, persistence) plane
    weight_power: float = 1.0     # weight = persistence ** power


class _UnionFind:
    """Tiny union-find with elder-rule (older root wins ties)."""

    __slots__ = ("parent", "birth")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.birth = [float("inf")] * n  # birth time of the component containing i

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union_elder(self, i: int, j: int) -> Tuple[int, int]:
        """Merge by elder rule. Returns (survivor_root, dead_root)."""
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return ri, ri
        # Younger component (later birth) dies
        if self.birth[ri] <= self.birth[rj]:
            self.parent[rj] = ri
            return ri, rj
        else:
            self.parent[ri] = rj
            return rj, ri


def sublevel_persistence_0d(values: np.ndarray) -> np.ndarray:
    """Compute the 0-D sublevel-set persistence diagram of a 2-D scalar field.

    Args:
        values: (H, W) float array. Treated as a function on a 4-connected grid.

    Returns:
        (k, 2) array of (birth, death) pairs. The single infinite-persistence
        bar (the global minimum) is replaced by (min, max + eps) so that all
        pairs are finite and the descriptor is bounded.
    """
    h, w = values.shape
    n = h * w
    flat = values.ravel()

    # Sort pixels by value (sublevel sweep)
    order = np.argsort(flat, kind="stable")

    uf = _UnionFind(n)
    activated = np.zeros(n, dtype=bool)
    pairs = []

    # 4-connected neighbors of pixel index i
    def neighbors(i: int):
        y, x = divmod(i, w)
        if x > 0:    yield i - 1
        if x < w-1:  yield i + 1
        if y > 0:    yield i - w
        if y < h-1:  yield i + w

    for idx in order:
        v = float(flat[idx])
        uf.birth[idx] = v
        activated[idx] = True
        # Connect to already-activated neighbors
        for nb in neighbors(idx):
            if activated[nb]:
                ri, rj = uf.find(idx), uf.find(nb)
                if ri != rj:
                    survivor, dead = uf.union_elder(ri, rj)
                    # The dead component "dies" at the current threshold v
                    pairs.append((uf.birth[dead], v))

    # Add the one essential class (global minimum, never dies) with finite death
    vmin, vmax = float(flat.min()), float(flat.max())
    pairs.append((vmin, vmax + 1e-6))

    if not pairs:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pairs, dtype=np.float32)


def persistence_image(diagram: np.ndarray, cfg: PersistenceImageConfig) -> np.ndarray:
    """Vectorize a persistence diagram into a fixed-length feature.

    Args:
        diagram: (k, 2) array of (birth, death) pairs in [0, 1] (approx).
        cfg: image configuration.

    Returns:
        (resolution * resolution,) float32 vector.
    """
    res = cfg.resolution
    img = np.zeros((res, res), dtype=np.float32)
    if diagram.size == 0:
        return img.ravel()

    # Switch to (birth, persistence) coordinates - the standard PI domain
    births = diagram[:, 0]
    persistences = diagram[:, 1] - diagram[:, 0]
    # Filter zero-persistence noise
    keep = persistences > 1e-8
    births = births[keep]
    persistences = persistences[keep]
    if births.size == 0:
        return img.ravel()

    weights = np.power(persistences, cfg.weight_power, dtype=np.float32)

    # Fixed [0, 1] x [0, 1] grid - rank-normalized coordinates always live here.
    # A fixed grid is essential for invariance: a diagram-dependent grid would
    # destroy monotone-tone invariance (max persistence varies across patches).
    bx = np.linspace(0.0, 1.0, res, dtype=np.float32)
    by = np.linspace(0.0, 1.0, res, dtype=np.float32)
    gx, gy = np.meshgrid(bx, by)  # (res, res)

    inv_2sigma2 = 1.0 / (2.0 * cfg.sigma * cfg.sigma)
    for b, p, w in zip(births, persistences, weights):
        img += w * np.exp(-((gx - b) ** 2 + (gy - p) ** 2) * inv_2sigma2)

    return img.ravel()


def rank_normalize_diagram(diagram: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map (birth, death) values to [0, 1] via the empirical CDF of `values`.

    The empirical CDF rank is a *complete invariant* of any monotone tone
    curve: if f -> phi(f) for monotone phi, then rank_phi(phi(t)) = rank_f(t)
    for every t. Composing this with the persistence-image vectorization
    yields a descriptor exactly invariant to gamma, brightness, contrast, and
    any other monotone intensity remapping.
    """
    if diagram.size == 0:
        return diagram
    sorted_values = np.sort(values.ravel())
    n = len(sorted_values)
    if n == 0:
        return diagram
    # searchsorted gives the rank in [0, n]; divide by n to land in [0, 1]
    ranks = np.searchsorted(sorted_values, diagram, side="right") / float(n)
    return ranks.astype(np.float32)


class PersistenceDescriptor(DescriptorStrategy):
    """Stable, monotone-tone-invariant patch descriptor via 0-D persistence.

    Pipeline: grayscale patch -> sublevel-set persistence diagram -> rank-
    normalize via empirical CDF -> persistence image -> L2 normalize.

    The rank-normalization step makes the descriptor *exactly* invariant to
    any monotone tone curve (Cohen-Steiner stability + rank invariance).
    """

    def __init__(self, cfg: PersistenceImageConfig | None = None):
        self.cfg = cfg or PersistenceImageConfig()

    def compute(self, patch: np.ndarray) -> np.ndarray:
        gray = _to_gray(patch).astype(np.float32)
        diagram = sublevel_persistence_0d(gray)
        diagram = rank_normalize_diagram(diagram, gray)
        feature = persistence_image(diagram, self.cfg)
        return _normalize(feature)
