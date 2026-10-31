"""Main forensics engine that orchestrates the full pipeline.

Pipeline for ``match``:

1. Extract source patches and describe them by their normalized luminance
   pattern (``LuminanceDescriptor``).
2. Batched exact k-NN search against every indexed target patch.
3. Cheap per-patch check: fit grey-level gain and offset to each candidate pair
   and keep those whose residual passes a chi-square test.
4. Cluster surviving matches by translation (per target image and scale).
5. For each cluster fit *one* 3x3 color transform over all of its pixels, keep
   the patches whose luminance agrees with it, and score the region with an
   a-contrario NFA.

Step 5 is where the decision is made. A single patch says little on its own;
a single transform shared by dozens of neighbouring patches at one offset is
hard to get by accident, and the NFA says how hard. Residuals are tested on
luminance because JPEG keeps it at full resolution while subsampling color.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from spade.aggregation.heatmap import Match, SpatialAggregator
from spade.aggregation.localize import Localization, localize_region, mask_to_bbox
from spade.descriptors.core import LUMA, LuminanceDescriptor, luminance
from spade.exceptions import (
    ConfigurationError,
    ImageSizeError,
    IndexStoreError,
    InvalidImageError,
    TargetExistsError,
)
from spade.extraction.patches import PatchExtractor, max_entropy
from spade.extraction.orientation import canonical_patches, patch_orientations
from spade.extraction.pyramid import ImagePyramid
from spade.jpeg import quantization_sigma
from spade.logging_config import get_logger
from spade.search.index import ANNIndex
from spade.verification.affine import AffineVerifier, fit_gain_offset_batch
from spade.verification.coherence import (
    CoherentRegion,
    SpatialCoherenceVerifier,
    find_rigid_regions,
    make_region,
)
from spade.verification.nfa import log10_nfa, overlap_factor
from spade.verification.probability import ProbabilityModel

logger = get_logger("engine")

# Regions are verified biggest-first; past this many, the rest are tiny clusters
# that couldn't reach significance anyway.
_MAX_REGIONS = 100
# Pixel pairs used for one region-level color fit. Plenty for 12 parameters.
_MAX_FIT_PIXELS = 20000
_FIT_CHUNK = 20000
_BUNDLE_VERSION = 1
# Rotation-invariant mode: matches within this angle of a region's rotation agree.
_ANGLE_TOLERANCE = float(np.radians(15))


def _env_int(key: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """
    Get integer from environment variable with validation.

    Args:
        key: Environment variable name
        default: Default value if not set
        min_val: Optional minimum allowed value
        max_val: Optional maximum allowed value

    Returns:
        Validated integer value

    Raises:
        ConfigurationError: If value is invalid or out of range
    """
    val_str = os.environ.get(key)
    if val_str is None:
        return default

    try:
        val = int(val_str)
    except (ValueError, TypeError) as e:
        raise ConfigurationError(
            f"Invalid integer value for {key}='{val_str}'. "
            f"Expected integer, got: {type(val_str).__name__}. "
            f"Example: {key}={default}"
        ) from e

    if min_val is not None and val < min_val:
        raise ConfigurationError(
            f"Value for {key}={val} is below minimum allowed value {min_val}. "
            f"Please set {key} >= {min_val}"
        )

    if max_val is not None and val > max_val:
        raise ConfigurationError(
            f"Value for {key}={val} exceeds maximum allowed value {max_val}. "
            f"Please set {key} <= {max_val}"
        )

    return val


def _env_float(key: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """
    Get float from environment variable with validation.

    Args:
        key: Environment variable name
        default: Default value if not set
        min_val: Optional minimum allowed value
        max_val: Optional maximum allowed value

    Returns:
        Validated float value

    Raises:
        ConfigurationError: If value is invalid or out of range
    """
    val_str = os.environ.get(key)
    if val_str is None:
        return default

    try:
        val = float(val_str)
    except (ValueError, TypeError) as e:
        raise ConfigurationError(
            f"Invalid float value for {key}='{val_str}'. "
            f"Expected number, got: {type(val_str).__name__}. "
            f"Example: {key}={default}"
        ) from e

    if min_val is not None and val < min_val:
        raise ConfigurationError(
            f"Value for {key}={val} is below minimum allowed value {min_val}. "
            f"Please set {key} >= {min_val}"
        )

    if max_val is not None and val > max_val:
        raise ConfigurationError(
            f"Value for {key}={val} exceeds maximum allowed value {max_val}. "
            f"Please set {key} <= {max_val}"
        )

    return val


def _env_bool(key: str, default: bool) -> bool:
    """
    Get boolean from environment variable with validation.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Boolean value

    Raises:
        ConfigurationError: If value is invalid
    """
    val = os.environ.get(key)
    if val is None:
        return default

    val_lower = val.lower()
    if val_lower in ("true", "1", "yes", "on"):
        return True
    elif val_lower in ("false", "0", "no", "off"):
        return False
    else:
        raise ConfigurationError(
            f"Invalid boolean value for {key}='{val}'. "
            f"Expected one of: true/false, 1/0, yes/no, on/off (case-insensitive). "
            f"Example: {key}=true"
        )


def _env_optional_float(key: str, min_val: Optional[float] = None) -> Optional[float]:
    """Float from the environment, or None when the variable is unset or empty."""
    if not os.environ.get(key):
        return None
    return _env_float(key, 0.0, min_val=min_val)


@dataclass
class Config:
    """
    Configuration for the forensics engine.

    Every field can be set through an environment variable with the SPADE_
    prefix (SPADE_PATCH_SIZE, SPADE_K_NEIGHBORS, SPADE_MAX_NFA, ...).
    """
    patch_size: int = field(default_factory=lambda: _env_int("SPADE_PATCH_SIZE", 5, min_val=3, max_val=16))
    stride: int = field(default_factory=lambda: _env_int("SPADE_STRIDE", 1, min_val=1))
    # Optional grey-level entropy floor in bits. Flat patches are always dropped by
    # the descriptor, so this is only for pruning weakly textured ones.
    entropy_threshold: Optional[float] = field(default_factory=lambda: _env_optional_float("SPADE_ENTROPY_THRESHOLD", min_val=0.0))
    k_neighbors: int = field(default_factory=lambda: _env_int("SPADE_K_NEIGHBORS", 8, min_val=1))
    # Squared L2 distance between descriptors (squared chordal distance between
    # color subspaces).
    distance_threshold: float = field(default_factory=lambda: _env_float("SPADE_DISTANCE_THRESHOLD", 0.5, min_val=0.0))
    # Per-patch chi-square p-value floor; the cheap pre-filter, not the decision.
    min_probability: float = field(default_factory=lambda: _env_float("SPADE_MIN_PROBABILITY", 0.01, min_val=0.0, max_val=1.0))
    noise_sigma: float = field(default_factory=lambda: _env_float("SPADE_NOISE_SIGMA", 0.02, min_val=0.0))
    smoothing_sigma: float = field(default_factory=lambda: _env_float("SPADE_SMOOTHING_SIGMA", 2.0, min_val=0.0))
    # Region clustering and significance
    coherence_enabled: bool = field(default_factory=lambda: _env_bool("SPADE_COHERENCE_ENABLED", True))
    coherence_offset_tolerance: int = field(default_factory=lambda: _env_int("SPADE_COHERENCE_OFFSET_TOLERANCE", 2, min_val=0))
    coherence_min_cluster: int = field(default_factory=lambda: _env_int("SPADE_COHERENCE_MIN_CLUSTER", 3, min_val=2))
    # A region is reported as significant when its NFA is below this.
    max_nfa: float = field(default_factory=lambda: _env_float("SPADE_MAX_NFA", 1.0, min_val=0.0))
    # Match copies at any rotation: patches are described in their own
    # orientation and regions are rigid transforms instead of translations.
    rotation_invariant: bool = field(default_factory=lambda: _env_bool("SPADE_ROTATION_INVARIANT", False))
    # Multi-scale pyramid (targets only)
    pyramid_enabled: bool = field(default_factory=lambda: _env_bool("SPADE_PYRAMID_ENABLED", False))
    pyramid_scales: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.125])
    pyramid_min_size: int = field(default_factory=lambda: _env_int("SPADE_PYRAMID_MIN_SIZE", 16, min_val=3))
    # Memory management: least recently indexed targets are evicted past this.
    max_targets: Optional[int] = field(default_factory=lambda: _env_int("SPADE_MAX_TARGETS", 1000, min_val=1) if os.environ.get("SPADE_MAX_TARGETS") else None)

    def __post_init__(self):
        if not (3 <= self.patch_size <= 16):
            raise ConfigurationError(f"patch_size must be between 3 and 16, got {self.patch_size}")
        if self.stride < 1:
            raise ConfigurationError(f"stride must be >= 1, got {self.stride}")
        if self.k_neighbors < 1:
            raise ConfigurationError(f"k_neighbors must be >= 1, got {self.k_neighbors}")
        if not 0 <= self.min_probability <= 1:
            raise ConfigurationError(f"min_probability must be 0-1, got {self.min_probability}")
        if self.noise_sigma <= 0:
            raise ConfigurationError(f"noise_sigma must be > 0, got {self.noise_sigma}")
        if self.max_nfa <= 0:
            raise ConfigurationError(f"max_nfa must be > 0, got {self.max_nfa}")
        if self.entropy_threshold is not None:
            ceiling = max_entropy(self.patch_size)
            if not 0 <= self.entropy_threshold < ceiling:
                raise ConfigurationError(
                    f"entropy_threshold must be in [0, {ceiling:.2f}) for {self.patch_size}x{self.patch_size} "
                    f"patches, got {self.entropy_threshold} (nothing would pass)"
                )
        if self.coherence_offset_tolerance < 0:
            raise ConfigurationError(f"coherence_offset_tolerance must be >= 0, got {self.coherence_offset_tolerance}")
        if self.coherence_min_cluster < 2:
            raise ConfigurationError(f"coherence_min_cluster must be >= 2, got {self.coherence_min_cluster}")
        if self.pyramid_enabled:
            if not self.pyramid_scales:
                raise ConfigurationError("pyramid_scales must not be empty when pyramid_enabled")
            if not all(scale > 0 for scale in self.pyramid_scales):
                raise ConfigurationError(f"All pyramid_scales must be > 0, got {self.pyramid_scales}")
            if 1.0 not in self.pyramid_scales:
                raise ConfigurationError(f"pyramid_scales must include 1.0 (original scale), got {self.pyramid_scales}")
        if self.rotation_invariant and self.pyramid_enabled:
            raise ConfigurationError("rotation_invariant and pyramid_enabled can't be combined yet")
        if self.max_targets is not None and self.max_targets < 1:
            raise ConfigurationError(f"max_targets must be >= 1 or None, got {self.max_targets}")

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **overrides) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        values.update(overrides)
        return cls(**values)


@dataclass
class MatchResult:
    """Result from matching source against indexed targets."""
    matches: List[Match]
    best_match: Optional[Match]
    heatmap: Optional[np.ndarray]
    coherent_regions: List[CoherentRegion]
    stats: Dict[str, Any]
    # Footprint of the significant regions in source coordinates.
    localization: Optional[Localization] = None

    @property
    def significant_regions(self) -> List[CoherentRegion]:
        return [r for r in self.coherent_regions if r.significant]


@dataclass
class _Target:
    """What we keep per indexed image. ``patches`` may be dropped and reloaded from ``path``."""
    shape: Tuple[int, int]
    path: Optional[str]
    patches: Optional[np.ndarray] = None   # (N, s, s, 3) float32, aligned with the index
    coords: Optional[np.ndarray] = None    # (N, 2) full-resolution (x, y)
    scales: Optional[np.ndarray] = None    # (N,) pyramid scale of each patch
    jpeg_sigma: float = 0.0                # rounding-error std of the file it came from
    thetas: Optional[np.ndarray] = None    # (N,) patch orientations (rotation-invariant mode)


@dataclass
class _Candidates:
    """Verified per-patch matches, as parallel arrays (one row per match)."""
    src_idx: np.ndarray
    tgt_idx: np.ndarray
    matches: List[Match]


class ForensicsEngine:
    """
    Main SPADE engine for image fragment forensics.

    Usage:
        engine = ForensicsEngine()
        engine.index_target(target_image, "target_id")
        result = engine.match(source_image)
        for region in result.significant_regions:
            print(region.image_id, region.offset, region.log10_nfa)
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.extractor = PatchExtractor(
            size=self.config.patch_size,
            stride=self.config.stride,
            entropy_threshold=self.config.entropy_threshold,
        )
        self.descriptor = LuminanceDescriptor(self.config.patch_size)
        self.index = ANNIndex(dim=self.descriptor.dim)
        self.verifier = AffineVerifier(noise_sigma=self.config.noise_sigma)
        self.aggregator = SpatialAggregator(smoothing_sigma=self.config.smoothing_sigma)
        self.coherence_verifier = SpatialCoherenceVerifier(
            offset_tolerance=self.config.coherence_offset_tolerance,
            min_cluster_size=self.config.coherence_min_cluster,
        )
        self.pyramid: Optional[ImagePyramid] = None
        if self.config.pyramid_enabled:
            self.pyramid = ImagePyramid(
                scales=self.config.pyramid_scales,
                min_size=self.config.pyramid_min_size,
            )
        self._targets: "OrderedDict[str, _Target]" = OrderedDict()
        # Set only on the private engine match_self builds: search the image
        # against itself, ignoring offsets shorter than this.
        self._self_min_offset: Optional[int] = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_target(
        self,
        image: np.ndarray,
        image_id: str,
        image_path: Optional[str] = None,
        jpeg_sigma: Optional[float] = None,
    ) -> int:
        """
        Index a target image for later matching.

        Args:
            image: RGB image as numpy array
            image_id: Unique identifier for this image
            image_path: Optional file path; lets a saved index reload the
                image's patches later instead of keeping them in the bundle
            jpeg_sigma: the image's JPEG rounding error (``spade.jpeg``); read
                from ``image_path`` when not given

        Returns:
            Number of patches indexed (flat patches are skipped)

        Raises:
            TargetExistsError: if ``image_id`` is already indexed
        """
        if image_id in self._targets:
            raise TargetExistsError(f"Target '{image_id}' is already indexed")
        image = self._normalize_image(image)
        patches, coords, scales, thetas, descriptors = self._describe_target(image)
        # Evict only once the new target is ready, so a failure above costs nothing.
        self._evict_lru_if_needed()
        self.index.add(image_id, descriptors)
        if jpeg_sigma is None:
            jpeg_sigma = (quantization_sigma(image_path) if image_path else None) or 0.0
        self._targets[image_id] = _Target(
            shape=(int(image.shape[0]), int(image.shape[1])),
            path=image_path,
            patches=patches,
            coords=coords,
            scales=scales,
            jpeg_sigma=float(jpeg_sigma),
            thetas=thetas,
        )
        return len(patches)

    def _describe_target(self, image: np.ndarray):
        """Patches, full-res coords, scales, orientations and descriptors for one target, flat patches removed."""
        if self.pyramid is not None:
            pyr = self.pyramid.extract_patches(image, self.extractor)
            patches, coords, scales = pyr.patches, pyr.original_coords, pyr.scales
            descriptors, keep = self.descriptor.compute_batch(patches)
            return patches[keep], coords[keep], scales[keep], np.zeros(int(keep.sum())), descriptors[keep]
        patches, coords, thetas, descriptors = self._extract(image)
        return patches, coords, np.ones(len(patches), dtype=np.float32), thetas, descriptors

    def _extract(self, image: np.ndarray):
        """Patches, coords, orientations and descriptors at full resolution, flat patches removed.

        In rotation-invariant mode each patch is resampled in its own canonical
        orientation, so descriptors and fits compare like with like.
        """
        col = self.extractor.extract(image)
        patches, coords = col.patches, col.coords
        thetas = np.zeros(len(patches))
        if self.config.rotation_invariant and len(patches):
            thetas = patch_orientations(image, coords, self.config.patch_size)
            patches = canonical_patches(image, coords, self.config.patch_size, thetas)
        descriptors, keep = self.descriptor.compute_batch(patches)
        return patches[keep], coords[keep], thetas[keep], descriptors[keep]

    def _evict_lru_if_needed(self) -> None:
        if self.config.max_targets is None:
            return
        while len(self._targets) >= self.config.max_targets:
            oldest_id = next(iter(self._targets))
            logger.warning(f"Max targets ({self.config.max_targets}) reached. Evicting oldest target: {oldest_id}")
            self.remove_target(oldest_id)

    def remove_target(self, image_id: str) -> bool:
        """Remove a target and its descriptors. Returns False if it wasn't indexed."""
        if image_id not in self._targets:
            return False
        del self._targets[image_id]
        self.index.remove(image_id)
        return True

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(
        self,
        source_image: np.ndarray,
        target_ids: Optional[List[str]] = None,
        return_heatmap: bool = True,
        jpeg_sigma: float = 0.0,
    ) -> MatchResult:
        """
        Match source image against indexed targets.

        Args:
            source_image: RGB image to search for
            target_ids: Optional list of specific targets to match against
            return_heatmap: Whether to generate a target heatmap for the best match
            jpeg_sigma: the source's JPEG rounding error; use
                ``spade.jpeg.quantization_sigma(path)`` for a file

        Returns:
            MatchResult. ``coherent_regions`` are sorted most significant first;
            use ``significant_regions`` for the ones with NFA < max_nfa.
        """
        source_image = self._normalize_image(source_image)
        source_shape = (int(source_image.shape[0]), int(source_image.shape[1]))
        src_patches, src_coords, src_thetas, descriptors = self._extract(source_image)

        if len(src_patches) == 0 or self.index.size == 0:
            return MatchResult(
                [], None, None, [], self._stats(len(src_patches), 0, []),
                localization=localize_region([], source_shape),
            )

        candidates, verified_per_target = self._verified_candidates(
            src_patches, src_coords, src_thetas, descriptors, target_ids, jpeg_sigma
        )
        matches = candidates.matches

        searched = [t for t in self._targets if target_ids is None or t in set(target_ids)]
        n_tests = self._count_tests(source_shape, len(src_patches), searched)
        regions: List[CoherentRegion] = []
        if self.config.coherence_enabled and matches:
            regions = self._significant_regions(
                candidates, src_patches, src_coords, verified_per_target, n_tests, jpeg_sigma
            )

        best_match = None
        if regions:
            best_match = max(regions[0].matches, key=lambda m: m.probability)
        elif matches:
            best_match = max(matches, key=lambda m: m.probability)

        heatmap = None
        if return_heatmap and best_match is not None:
            if regions:
                heat_matches = regions[0].matches
            else:
                heat_matches = [m for m in matches if m.image_id == best_match.image_id]
            heatmap = self.aggregator.aggregate(heat_matches, self._targets[best_match.image_id].shape)

        significant = [m for r in regions if r.significant for m in r.matches]
        localization = localize_region(significant, source_shape, largest_component=False)

        return MatchResult(
            matches=matches,
            best_match=best_match,
            heatmap=heatmap,
            coherent_regions=regions,
            stats={**self._stats(len(src_patches), len(matches), regions), "n_tests": n_tests},
            localization=localization,
        )

    def match_self(
        self, image: np.ndarray, min_offset: Optional[int] = None, jpeg_sigma: float = 0.0
    ) -> MatchResult:
        """Find regions copied within one image (copy-move).

        The image is matched against itself, ignoring offsets shorter than
        ``min_offset`` (default: twice the patch size) so a region doesn't
        match its own neighbourhood. Each duplicated pair is reported once, as
        a region whose source and target are the two copies; which one is the
        original can't be told from the image alone. ``localization`` covers
        both copies. Doesn't touch this engine's index.
        """
        engine = ForensicsEngine(dataclasses.replace(self.config, pyramid_enabled=False, max_targets=None))
        engine._self_min_offset = min_offset if min_offset is not None else 2 * self.config.patch_size
        image = self._normalize_image(image)
        engine.index_target(image, "self", jpeg_sigma=jpeg_sigma)
        result = engine.match(image, return_heatmap=False, jpeg_sigma=jpeg_sigma)
        shape = image.shape[:2]
        both = np.zeros(shape, dtype=bool)
        for region in result.significant_regions:
            for m in region.matches:
                for x, y in (m.source_coord, m.target_coord):
                    both[y:y + m.patch_size, x:x + m.patch_size] = True
        result.localization = Localization(
            mask=both,
            bbox=mask_to_bbox(both),
            area=int(both.sum()),
            area_fraction=float(both.sum()) / float(shape[0] * shape[1]),
        )
        return result

    @staticmethod
    def _stats(n_source: int, n_matches: int, regions: List[CoherentRegion]) -> Dict[str, Any]:
        return {
            "source_patches": n_source,
            "total_matches": n_matches,
            "coherent_regions": len(regions),
            "significant_regions": sum(1 for r in regions if r.significant),
        }

    def _noise_model(self, image_id: str, source_jpeg_sigma: float) -> ProbabilityModel:
        """Residual noise for a pair: base noise plus both sides' JPEG rounding error."""
        target_sigma = self._targets[image_id].jpeg_sigma
        sigma = np.sqrt(self.config.noise_sigma ** 2 + source_jpeg_sigma ** 2 + target_sigma ** 2)
        return ProbabilityModel(noise_sigma=float(sigma))

    def _verified_candidates(self, src_patches, src_coords, src_thetas, descriptors, target_ids, jpeg_sigma=0.0):
        """k-NN search plus the per-patch chi-square pre-filter, vectorized."""
        k = self.config.k_neighbors
        if self._self_min_offset is not None:
            # A patch's nearest neighbours in its own image are mostly itself and
            # its slightly shifted copies; ask for more so real duplicates survive.
            k = 4 * k
        hits = self.index.search(descriptors, k)
        valid = (hits.owner >= 0) & (hits.distances <= self.config.distance_threshold)
        if target_ids is not None:
            wanted = set(target_ids)
            allowed = np.array([i in wanted for i in hits.image_ids] + [False])
            valid &= allowed[hits.owner]  # owner == -1 lands on the trailing False
        rows, cols = np.nonzero(valid)
        owners = hits.owner[rows, cols]
        tgt_all = hits.patch_idx[rows, cols]

        src_out, tgt_out, matches = [], [], []
        verified_per_target: Dict[str, int] = {}

        for owner in np.unique(owners):
            image_id = hits.image_ids[owner]
            target = self._load_target(image_id)
            if target is None:
                continue
            sel = owners == owner
            src_idx, tgt_idx = rows[sel], tgt_all[sel]
            noise = self._noise_model(image_id, jpeg_sigma)
            if self._self_min_offset is not None:
                offset = target.coords[tgt_idx] - src_coords[src_idx]
                # Drop trivial self-matches, and keep one direction of each copy
                # pair (A->B and B->A are the same evidence).
                forward = (offset[:, 0] > 0) | ((offset[:, 0] == 0) & (offset[:, 1] > 0))
                keep = forward & (np.abs(offset).max(axis=1) >= self._self_min_offset)
                src_idx, tgt_idx = src_idx[keep], tgt_idx[keep]

            for start in range(0, len(src_idx), _FIT_CHUNK):
                s_chunk = src_idx[start:start + _FIT_CHUNK]
                t_chunk = tgt_idx[start:start + _FIT_CHUNK]
                src_chunk, tgt_chunk = src_patches[s_chunk], target.patches[t_chunk]
                usable = _usable_pixels(src_chunk, tgt_chunk)
                rss = fit_gain_offset_batch(
                    luminance(src_chunk).reshape(len(s_chunk), -1),
                    luminance(tgt_chunk).reshape(len(t_chunk), -1),
                    usable,
                )
                prob = noise.compute(rss, usable.sum(axis=1) - 2)
                ok = np.nonzero(prob >= self.config.min_probability)[0]
                for j in ok:
                    ti = t_chunk[j]
                    matches.append(Match(
                        source_coord=(int(src_coords[s_chunk[j], 0]), int(src_coords[s_chunk[j], 1])),
                        target_coord=(int(target.coords[ti, 0]), int(target.coords[ti, 1])),
                        patch_size=self.config.patch_size,
                        probability=float(prob[j]),
                        image_id=image_id,
                        target_scale=float(target.scales[ti]),
                        rotation=float(_wrap_angle(target.thetas[ti] - src_thetas[s_chunk[j]])),
                    ))
                src_out.append(s_chunk[ok])
                tgt_out.append(t_chunk[ok])
                verified_per_target[image_id] = verified_per_target.get(image_id, 0) + len(ok)

        candidates = _Candidates(
            src_idx=np.concatenate(src_out) if src_out else np.zeros(0, dtype=np.int64),
            tgt_idx=np.concatenate(tgt_out) if tgt_out else np.zeros(0, dtype=np.int64),
            matches=matches,
        )
        return candidates, verified_per_target

    # ------------------------------------------------------------------
    # Region-level verification
    # ------------------------------------------------------------------

    def _count_tests(self, source_shape, n_source: int, searched: List[str]) -> float:
        """Region hypotheses a search tries: offsets x scales x region seeds, over every target searched.

        Every target counts, hits or not; leaving out the ones without hits
        would understate how hard we looked.
        """
        cfg = self.config
        overlap = overlap_factor(cfg.patch_size, cfg.stride)
        window = (2 * cfg.coherence_offset_tolerance + 1) ** 2
        n_scales = len(cfg.pyramid_scales) if self.pyramid is not None else 1
        sh, sw = source_shape
        n_offsets = sum(
            (sw + self._targets[t].shape[1]) * (sh + self._targets[t].shape[0]) / window * n_scales
            for t in searched
        )
        n_tests = max(n_offsets, 1.0) * max(n_source / overlap, 1.0)
        if cfg.rotation_invariant:
            n_tests *= np.pi / _ANGLE_TOLERANCE  # rotation hypotheses
        return n_tests

    def _significant_regions(self, candidates, src_patches, src_coords, verified_per_target, n_tests, jpeg_sigma):
        """Cluster, fit one color transform per cluster, keep consistent patches, score by NFA."""
        cfg = self.config
        if cfg.rotation_invariant:
            clusters = find_rigid_regions(
                candidates.matches, cfg.coherence_offset_tolerance, cfg.coherence_min_cluster, _ANGLE_TOLERANCE
            )
        else:
            clusters = self.coherence_verifier.find_coherent_regions(candidates.matches)
        clusters.sort(key=lambda r: len(r.matches), reverse=True)
        row_of = {id(m): i for i, m in enumerate(candidates.matches)}

        n_source = len(src_patches)
        overlap = overlap_factor(cfg.patch_size, cfg.stride)
        window = (2 * cfg.coherence_offset_tolerance + 1) ** 2
        max_log10_nfa = float(np.log10(cfg.max_nfa))

        regions = []
        for cluster in clusters[:_MAX_REGIONS]:
            rows = np.array([row_of[id(m)] for m in cluster.matches])
            target = self._targets[cluster.image_id]
            src = src_patches[candidates.src_idx[rows]]
            tgt = target.patches[candidates.tgt_idx[rows]]

            usable = _usable_pixels(src, tgt)
            M, b = self._fit_region_transform(src, tgt, usable)
            residual = tgt.reshape(len(rows), -1, 3) - (src.reshape(len(rows), -1, 3) @ M.T + b)
            # The transform is shared by the region, so each patch keeps one
            # degree of freedom per usable pixel.
            rss = (usable * (residual @ LUMA) ** 2).sum(axis=1)
            prob = self._noise_model(cluster.image_id, jpeg_sigma).compute(rss, usable.sum(axis=1))
            consistent = prob >= cfg.min_probability
            inliers = [m for m, ok in zip(cluster.matches, consistent) if ok]
            if len(inliers) < cfg.coherence_min_cluster:
                continue

            region = make_region(cluster.offset, inliers, cluster.image_id)
            region.rotation = cluster.rotation
            region.color_M, region.color_b = M, b
            for m in inliers:
                m.color_M, m.color_b = M, b

            # k: distinct source positions that agree; n: source positions tested in that area.
            k = len({m.source_coord for m in inliers})
            x0, y0, w, h = region.source_bbox
            last_x, last_y = x0 + w - cfg.patch_size, y0 + h - cfg.patch_size
            in_box = (
                (src_coords[:, 0] >= x0) & (src_coords[:, 0] <= last_x)
                & (src_coords[:, 1] >= y0) & (src_coords[:, 1] <= last_y)
            )
            n = max(int(in_box.sum()), k)
            n_target = max(self.index.count(cluster.image_id), 1)
            p = min(1.0, verified_per_target[cluster.image_id] / (n_source * n_target) * window)
            if cfg.rotation_invariant:
                # A chance match must also agree on the angle.
                p *= _ANGLE_TOLERANCE / np.pi
            region.log10_nfa = log10_nfa(k / overlap, n / overlap, p, n_tests)
            region.max_log10_nfa = max_log10_nfa
            region.n_tests = n_tests
            regions.append(region)

        regions.sort(key=lambda r: (r.log10_nfa, -len(r.matches)))
        return regions

    def _fit_region_transform(
        self, src: np.ndarray, tgt: np.ndarray, usable: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Robust (Huber) affine color fit over the usable pixel pairs of a region."""
        src_px = src.reshape(-1, 3)
        tgt_px = tgt.reshape(-1, 3)
        usable = usable.ravel()
        if usable.sum() >= 4:
            src_px, tgt_px = src_px[usable], tgt_px[usable]
        if len(src_px) > _MAX_FIT_PIXELS:
            pick = np.random.RandomState(0).choice(len(src_px), _MAX_FIT_PIXELS, replace=False)
            src_px, tgt_px = src_px[pick], tgt_px[pick]
        fit = self.verifier.verify(src_px[:, None, :], tgt_px[:, None, :])
        return fit.M, fit.b

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the index bundle: ``{path}.index``, ``{path}.meta.json`` and
        ``{path}.engine.json`` (config + target paths and shapes).

        Patches aren't stored; ``load`` re-extracts them from each target's
        ``image_path`` when it's first needed.
        """
        from spade.search.index import _atomic_write, _write_json

        self.index.save(path)
        state = {
            "version": _BUNDLE_VERSION,
            "config": self.config.to_dict(),
            "targets": [
                [image_id, t.path, t.shape[0], t.shape[1], t.jpeg_sigma] for image_id, t in self._targets.items()
            ],
        }
        _atomic_write(f"{path}.engine.json", lambda tmp: _write_json(tmp, state))

    @classmethod
    def load(cls, path: str, **config_overrides) -> "ForensicsEngine":
        """
        Load a bundle written by ``save``.

        ``config_overrides`` may change matching settings (thresholds, k, ...);
        settings that shape the index (patch size, stride, entropy, pyramid)
        always come from the bundle so the stored descriptors stay valid.
        """
        try:
            with open(f"{path}.engine.json") as f:
                state = json.load(f)
        except FileNotFoundError as e:
            raise IndexStoreError(f"no engine bundle at {path}.engine.json") from e
        if state.get("version") != _BUNDLE_VERSION:
            raise IndexStoreError(f"unsupported bundle version {state.get('version')!r}")

        frozen = {"patch_size", "stride", "entropy_threshold", "pyramid_enabled", "pyramid_scales", "pyramid_min_size"}
        overrides = {k: v for k, v in config_overrides.items() if k not in frozen}
        engine = cls(Config.from_dict(state["config"], **overrides))
        index = ANNIndex.load(path)
        if index.dim != engine.descriptor.dim:
            raise IndexStoreError(f"index dim {index.dim} doesn't match descriptor dim {engine.descriptor.dim}")
        engine.index = index
        for image_id, image_path, h, w, jpeg_sigma in state["targets"]:
            engine._targets[image_id] = _Target(shape=(int(h), int(w)), path=image_path, jpeg_sigma=jpeg_sigma)
        return engine

    def _load_target(self, image_id: str) -> Optional[_Target]:
        """The target with patches in memory, re-extracting them from disk if needed."""
        target = self._targets.get(image_id)
        if target is None or target.patches is not None:
            return target
        if not target.path or not os.path.exists(target.path):
            logger.warning(f"Can't verify matches against '{image_id}': image file not available")
            return None
        image = self._normalize_image(load_image(target.path))
        patches, coords, scales, thetas, _ = self._describe_target(image)
        if len(patches) != self.index.count(image_id):
            logger.warning(f"'{image_id}' changed on disk since it was indexed; skipping it")
            return None
        target.patches, target.coords, target.scales, target.thetas = patches, coords, scales, thetas
        return target

    # ------------------------------------------------------------------

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to float32 in [0, 1] range."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ImageSizeError(f"Expected RGB image (H, W, 3), got shape {image.shape}")

        if image.dtype == np.uint8:
            result = image.astype(np.float32) / 255.0
        elif image.dtype == np.float32 and image.max() <= 1.0:
            result = image
        else:
            max_val = image.max()
            if not np.isfinite(max_val):
                raise InvalidImageError("Image contains NaN or infinite values")
            if max_val == 0:
                raise InvalidImageError("Image is completely black (all zeros)")
            result = image.astype(np.float32) / max_val

        if np.any(~np.isfinite(result)):
            raise InvalidImageError("Image contains NaN or infinite values")

        return result

    # Public accessors for API and external use
    @property
    def target_ids(self) -> List[str]:
        """List of indexed target image IDs."""
        return list(self._targets)

    @property
    def target_count(self) -> int:
        """Number of indexed targets."""
        return len(self._targets)

    def get_target_shape(self, image_id: str) -> Optional[Tuple[int, int]]:
        """Get (height, width) of indexed target, or None if not found."""
        target = self._targets.get(image_id)
        return target.shape if target else None

    def get_target_path(self, image_id: str) -> Optional[str]:
        """Get file path of indexed target, or None if not found."""
        target = self._targets.get(image_id)
        return target.path if target else None

    def has_target(self, image_id: str) -> bool:
        """Check if target is indexed."""
        return image_id in self._targets


def _wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _usable_pixels(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(N, P) bool: pixel pairs with no channel clipped at 0 or 1 on either side.

    A recolor that saturates breaks the affine model on exactly those pixels; a
    clipped source channel corrupts every predicted target channel, so the
    whole pixel pair goes. One 8-bit level of slack either way.
    """
    n = len(source)
    both = np.concatenate([source.reshape(n, -1, 3), target.reshape(n, -1, 3)], axis=2)
    return ((both > 1.0 / 255) & (both < 254.0 / 255)).all(axis=2)


def load_image(path: str) -> np.ndarray:
    """Load image from file path as RGB numpy array."""
    return np.array(Image.open(path).convert("RGB"))
