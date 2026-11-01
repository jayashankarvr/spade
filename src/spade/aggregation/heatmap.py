"""Spatial aggregation and heatmap generation."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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
    # Rotation-invariant mode: the measured turn from source to target, radians,
    # with target = R(rotation) @ source about the patch centres.
    rotation: float = 0.0
    # Found by searching the horizontally mirrored source (flip-invariant mode).
    mirrored: bool = False
    # Scale the source was searched at (query_scales); its footprint in the
    # original source is patch_size / source_scale.
    source_scale: float = 1.0


class MatchList(Sequence):
    """Verified matches kept as arrays; a Match object is made only when one is read.

    A search verifies millions of patch pairs and almost all of them are
    chance matches nobody looks at, so building an object for each one cost
    more time than the search itself (and hundreds of MB). Behaves like a
    read-only list of Match. An object, once made, is kept, so changes to it
    (a region setting its colour transform) stick.
    """

    _FIELDS = ("src_xy", "tgt_xy", "prob", "image_ids", "scale", "rotation", "mirrored", "source_scale")

    def __init__(self, patch_size: int, src_xy, tgt_xy, prob, image_ids, scale, rotation=None,
                 mirrored=None, source_scale=None):
        n = len(prob)
        self.patch_size = patch_size
        self.src_xy = np.asarray(src_xy, dtype=np.int64).reshape(n, 2)
        self.tgt_xy = np.asarray(tgt_xy, dtype=np.int64).reshape(n, 2)
        self.prob = np.asarray(prob, dtype=np.float64)
        self.image_ids = np.asarray(image_ids, dtype=object)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.rotation = np.zeros(n) if rotation is None else np.asarray(rotation, dtype=np.float64)
        self.mirrored = np.zeros(n, dtype=bool) if mirrored is None else np.asarray(mirrored, dtype=bool)
        self.source_scale = np.ones(n) if source_scale is None else np.asarray(source_scale, dtype=np.float64)
        self._made: Dict[int, Match] = {}

    @classmethod
    def empty(cls, patch_size: int) -> "MatchList":
        return cls(patch_size, [], [], [], [], [])

    @classmethod
    def concat(cls, lists: List["MatchList"]) -> "MatchList":
        """One list holding all of ``lists``; objects already made carry over."""
        out = cls(lists[0].patch_size, *(np.concatenate([getattr(m, f) for m in lists]) for f in cls._FIELDS))
        start = 0
        for m in lists:
            out._made.update({start + i: match for i, match in m._made.items()})
            start += len(m)
        return out

    def __len__(self) -> int:
        return len(self.prob)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        i = int(i)
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        match = self._made.get(i)
        if match is None:
            match = Match(
                source_coord=(int(self.src_xy[i, 0]), int(self.src_xy[i, 1])),
                target_coord=(int(self.tgt_xy[i, 0]), int(self.tgt_xy[i, 1])),
                patch_size=self.patch_size,
                probability=float(self.prob[i]),
                image_id=self.image_ids[i],
                target_scale=float(self.scale[i]),
                rotation=float(self.rotation[i]),
                mirrored=bool(self.mirrored[i]),
                source_scale=float(self.source_scale[i]),
            )
            self._made[i] = match
        return match

    def __eq__(self, other) -> bool:
        if not isinstance(other, (list, tuple, MatchList)):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    __hash__ = None

    def __repr__(self) -> str:
        return f"MatchList({len(self)} matches)"

    def best(self) -> Optional[Match]:
        """The match with the highest fit p-value (first one on ties)."""
        return self[int(np.argmax(self.prob))] if len(self) else None

    def for_image(self, image_id: str) -> List[Match]:
        return [self[i] for i in np.nonzero(self.image_ids == image_id)[0]]

    def map_to_original(self, scale: float, mirrored: bool, pass_width: int) -> None:
        """Move source coordinates from a rescaled and/or mirrored search pass back to the original image."""
        x = self.src_xy[:, 0].astype(np.float64)
        if mirrored:
            x = pass_width - self.patch_size - x
        self.src_xy = np.stack([np.round(x / scale), np.round(self.src_xy[:, 1] / scale)], axis=1).astype(np.int64)
        self.mirrored[:] = mirrored
        self.source_scale[:] = scale
        for i, m in self._made.items():
            m.source_coord = (int(self.src_xy[i, 0]), int(self.src_xy[i, 1]))
            m.mirrored, m.source_scale = mirrored, scale


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
