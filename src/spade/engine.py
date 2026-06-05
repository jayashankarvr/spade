"""Main forensics engine that orchestrates the full pipeline."""

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image

from spade.logging_config import get_logger
from spade.extraction.patches import PatchExtractor, PatchCollection
from spade.exceptions import (
    ConfigurationError,
    InvalidImageError,
    ImageSizeError,
    TargetNotFoundError,
    TargetExistsError,
)

logger = get_logger("engine")
from spade.extraction.pyramid import ImagePyramid, PyramidPatchCollection
from spade.descriptors.core import CompositeDescriptor
from spade.search.index import ANNIndex
from spade.search import LSH_AVAILABLE
from spade.verification.affine import AffineVerifier
from spade.verification.probability import ProbabilityModel
from spade.verification.coherence import SpatialCoherenceVerifier, CoherentRegion
from spade.aggregation.heatmap import SpatialAggregator, Match


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


@dataclass
class Config:
    """
    Configuration for the forensics engine.

    All settings can be overridden via environment variables with SPADE_ prefix:
    - SPADE_PATCH_SIZE, SPADE_STRIDE, SPADE_ENTROPY_THRESHOLD
    - SPADE_DESCRIPTOR_DIM, SPADE_K_NEIGHBORS, SPADE_DISTANCE_THRESHOLD
    - SPADE_MIN_PROBABILITY, SPADE_NOISE_SIGMA
    - SPADE_PYRAMID_ENABLED, SPADE_LSH_ENABLED
    """
    patch_size: int = field(default_factory=lambda: _env_int("SPADE_PATCH_SIZE", 3, min_val=3, max_val=6))
    stride: int = field(default_factory=lambda: _env_int("SPADE_STRIDE", 1, min_val=1))
    entropy_threshold: Optional[float] = field(default_factory=lambda: _env_float("SPADE_ENTROPY_THRESHOLD", 2.5, min_val=0.0))
    descriptor_dim: int = field(default_factory=lambda: _env_int("SPADE_DESCRIPTOR_DIM", 256, min_val=64, max_val=512))
    k_neighbors: int = field(default_factory=lambda: _env_int("SPADE_K_NEIGHBORS", 100, min_val=1))
    distance_threshold: float = field(default_factory=lambda: _env_float("SPADE_DISTANCE_THRESHOLD", 0.5, min_val=0.0))
    min_probability: float = field(default_factory=lambda: _env_float("SPADE_MIN_PROBABILITY", 0.5, min_val=0.0, max_val=1.0))
    noise_sigma: float = field(default_factory=lambda: _env_float("SPADE_NOISE_SIGMA", 0.02, min_val=0.0))
    smoothing_sigma: float = field(default_factory=lambda: _env_float("SPADE_SMOOTHING_SIGMA", 2.0, min_val=0.0))
    auto_train_pca: bool = field(default_factory=lambda: _env_bool("SPADE_AUTO_TRAIN_PCA", True))
    pca_train_samples: int = field(default_factory=lambda: _env_int("SPADE_PCA_TRAIN_SAMPLES", 5000, min_val=100))
    # Spatial coherence settings
    coherence_enabled: bool = field(default_factory=lambda: _env_bool("SPADE_COHERENCE_ENABLED", True))
    coherence_offset_tolerance: int = field(default_factory=lambda: _env_int("SPADE_COHERENCE_OFFSET_TOLERANCE", 2, min_val=0))
    coherence_min_cluster: int = field(default_factory=lambda: _env_int("SPADE_COHERENCE_MIN_CLUSTER", 3, min_val=2))
    scale_weights: Dict[int, float] = field(default_factory=lambda: {
        3: 1.0, 4: 1.2, 5: 1.5, 6: 2.0
    })
    # Multi-scale pyramid settings
    pyramid_enabled: bool = field(default_factory=lambda: _env_bool("SPADE_PYRAMID_ENABLED", False))
    pyramid_scales: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.125])
    pyramid_min_size: int = field(default_factory=lambda: _env_int("SPADE_PYRAMID_MIN_SIZE", 16, min_val=3))
    # LSH pre-filtering settings
    lsh_enabled: bool = field(default_factory=lambda: _env_bool("SPADE_LSH_ENABLED", False))
    lsh_threshold: float = field(default_factory=lambda: _env_float("SPADE_LSH_THRESHOLD", 0.3, min_val=0.0, max_val=1.0))
    lsh_candidates: int = field(default_factory=lambda: _env_int("SPADE_LSH_CANDIDATES", 1000, min_val=1))
    # Phase 3: GPU acceleration
    use_gpu: bool = field(default_factory=lambda: _env_bool("SPADE_USE_GPU", False))
    # Phase 3: Learned descriptors
    use_learned_descriptors: bool = field(default_factory=lambda: _env_bool("SPADE_LEARNED", False))
    learned_model: str = field(default_factory=lambda: os.environ.get("SPADE_LEARNED_MODEL", "resnet18"))
    # Phase 3: Distributed search
    distributed_enabled: bool = field(default_factory=lambda: _env_bool("SPADE_DISTRIBUTED", False))
    num_shards: int = field(default_factory=lambda: _env_int("SPADE_NUM_SHARDS", 4, min_val=1))
    # Memory management
    max_targets: Optional[int] = field(default_factory=lambda: _env_int("SPADE_MAX_TARGETS", 1000, min_val=1) if os.environ.get("SPADE_MAX_TARGETS") else None)
    # Bayes-factor pipeline (Option-A)
    scoring_mode: str = field(default_factory=lambda: os.environ.get("SPADE_SCORING_MODE", "chi2"))
    aggregation_mode: str = field(default_factory=lambda: os.environ.get("SPADE_AGGREGATION_MODE", "translation_cluster"))
    priors_dir: Optional[str] = field(default_factory=lambda: os.environ.get("SPADE_PRIORS_DIR"))

    def _validate_combinations(self):
        """Validate configuration combinations that require multiple settings."""
        # Import availability flags
        from spade.search import LSH_AVAILABLE
        from spade.descriptors import TORCH_AVAILABLE
        from spade.acceleration.backend import CUPY_AVAILABLE

        # LSH requires datasketch
        if self.lsh_enabled and not LSH_AVAILABLE:
            raise ConfigurationError(
                "LSH pre-filtering enabled but datasketch not installed. "
                "Install with: pip install datasketch"
            )

        # Learned descriptors require PyTorch
        if self.use_learned_descriptors and not TORCH_AVAILABLE:
            raise ConfigurationError(
                "Learned descriptors enabled but PyTorch not installed. "
                "Install with: pip install torch torchvision"
            )

        # GPU acceleration requires CuPy
        if self.use_gpu and not CUPY_AVAILABLE:
            raise ConfigurationError(
                "GPU acceleration enabled but CuPy not installed. "
                "Install with: pip install cupy-cuda11x (match your CUDA version)"
            )

        # Validate pyramid scales
        if self.pyramid_enabled:
            if not all(scale > 0 for scale in self.pyramid_scales):
                raise ConfigurationError(
                    f"All pyramid_scales must be > 0, got {self.pyramid_scales}"
                )
            if 1.0 not in self.pyramid_scales:
                raise ConfigurationError(
                    f"pyramid_scales must include 1.0 (original scale), got {self.pyramid_scales}"
                )

        # Descriptor dimension must be reasonable for patch size
        min_dim = self.patch_size * self.patch_size * 3  # RGB channels
        if self.descriptor_dim < min_dim:
            raise ConfigurationError(
                f"descriptor_dim ({self.descriptor_dim}) is too small for patch_size {self.patch_size}x{self.patch_size}. "
                f"Minimum: {min_dim} (patch pixels)"
            )

        # Distributed search requires reasonable shard count
        if self.distributed_enabled:
            if self.num_shards < 2:
                raise ConfigurationError(
                    f"Distributed search requires num_shards >= 2, got {self.num_shards}. "
                    f"Use single index if not distributing."
                )

        # k_neighbors should be reasonable for lsh_candidates
        if self.lsh_enabled and self.k_neighbors > self.lsh_candidates:
            raise ConfigurationError(
                f"k_neighbors ({self.k_neighbors}) cannot exceed lsh_candidates ({self.lsh_candidates}). "
                f"LSH pre-filter would eliminate valid results."
            )

    def __post_init__(self):
        # Validate patch_size
        if self.patch_size not in (3, 4, 5, 6):
            raise ConfigurationError(f"patch_size must be 3, 4, 5, or 6, got {self.patch_size}")
        # Validate stride
        if self.stride < 1:
            raise ConfigurationError(f"stride must be >= 1, got {self.stride}")
        # Validate probability thresholds
        if not 0 <= self.min_probability <= 1:
            raise ConfigurationError(f"min_probability must be 0-1, got {self.min_probability}")
        # Validate noise_sigma
        if self.noise_sigma <= 0:
            raise ConfigurationError(f"noise_sigma must be > 0, got {self.noise_sigma}")
        # Validate pyramid settings
        if self.pyramid_enabled and len(self.pyramid_scales) == 0:
            raise ConfigurationError("pyramid_scales must not be empty when pyramid_enabled")
        # Validate LSH settings
        if not 0 < self.lsh_threshold <= 1:
            raise ConfigurationError(f"lsh_threshold must be (0, 1], got {self.lsh_threshold}")
        if self.lsh_candidates < 1:
            raise ConfigurationError(f"lsh_candidates must be >= 1, got {self.lsh_candidates}")
        # Validate coherence settings
        if self.coherence_offset_tolerance < 0:
            raise ConfigurationError(f"coherence_offset_tolerance must be >= 0, got {self.coherence_offset_tolerance}")
        if self.coherence_min_cluster < 1:
            raise ConfigurationError(f"coherence_min_cluster must be >= 1, got {self.coherence_min_cluster}")
        # Validate memory management
        if self.max_targets is not None and self.max_targets < 1:
            raise ConfigurationError(f"max_targets must be >= 1 or None, got {self.max_targets}")

        # Validate configuration combinations
        self._validate_combinations()


@dataclass
class MatchResult:
    """Result from matching source against indexed targets."""
    matches: List[Match]
    best_match: Optional[Match]
    heatmap: Optional[np.ndarray]
    coherent_regions: List[CoherentRegion]
    stats: Dict[str, Any]


class ForensicsEngine:
    """
    Main SPADE engine for image fragment forensics.

    Usage:
        engine = ForensicsEngine()
        engine.index_target(target_image, "target_id")
        result = engine.match(source_image)
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

        # Patch extraction (with optional GPU acceleration)
        self.extractor = PatchExtractor(
            size=self.config.patch_size,
            stride=self.config.stride,
            entropy_threshold=self.config.entropy_threshold,
            use_gpu=self.config.use_gpu,
        )

        # Descriptor computation (learned or mathematical)
        if self.config.use_learned_descriptors:
            from spade.descriptors import LearnedDescriptor, TORCH_AVAILABLE
            if TORCH_AVAILABLE and LearnedDescriptor is not None:
                self.descriptor = LearnedDescriptor(
                    model_name=self.config.learned_model,
                    target_dim=self.config.descriptor_dim,
                    use_gpu=self.config.use_gpu,
                )
            else:
                logger.warning(
                    "Learned descriptors requested but PyTorch not available. "
                    "Install with: pip install spade-forensics[learned]. "
                    "Falling back to mathematical descriptors."
                )
                self.descriptor = CompositeDescriptor(target_dim=self.config.descriptor_dim)
        else:
            self.descriptor = CompositeDescriptor(target_dim=self.config.descriptor_dim)

        # Search index (distributed, hybrid LSH, or standard)
        if self.config.distributed_enabled:
            from spade.search.distributed import ShardedIndex
            self.index = ShardedIndex(
                dim=self.config.descriptor_dim,
                num_shards=self.config.num_shards,
                use_gpu=self.config.use_gpu,
            )
        elif self.config.lsh_enabled and LSH_AVAILABLE:
            from spade.search.lsh import HybridIndex, LSHConfig
            lsh_config = LSHConfig(threshold=self.config.lsh_threshold)
            self.index = HybridIndex(dim=self.config.descriptor_dim, lsh_config=lsh_config)
        elif self.config.lsh_enabled and not LSH_AVAILABLE:
            logger.warning(
                "LSH pre-filtering requested but datasketch not available. "
                "Install with: pip install spade-forensics[lsh]. "
                "Falling back to standard FAISS index."
            )
            self.index = ANNIndex(dim=self.config.descriptor_dim, use_gpu=self.config.use_gpu)
        else:
            self.index = ANNIndex(dim=self.config.descriptor_dim, use_gpu=self.config.use_gpu)

        self.verifier = AffineVerifier(noise_sigma=self.config.noise_sigma)
        self.probability_model = ProbabilityModel(noise_sigma=self.config.noise_sigma)
        self.aggregator = SpatialAggregator(
            scale_weights=self.config.scale_weights,
            smoothing_sigma=self.config.smoothing_sigma,
        )
        self.coherence_verifier = SpatialCoherenceVerifier(
            offset_tolerance=self.config.coherence_offset_tolerance,
            min_cluster_size=self.config.coherence_min_cluster,
            min_cluster_probability=self.config.min_probability,
        )

        # Multi-scale pyramid (Phase 2)
        self.pyramid: Optional[ImagePyramid] = None
        if self.config.pyramid_enabled:
            self.pyramid = ImagePyramid(
                scales=self.config.pyramid_scales,
                min_size=self.config.pyramid_min_size,
            )

        # Store only paths and patches, not full images (memory efficient)
        # Using OrderedDict for LRU eviction when max_targets is set
        self._target_paths: OrderedDict[str, Optional[str]] = OrderedDict()
        self._target_patches: OrderedDict[str, PatchCollection] = OrderedDict()
        self._target_pyramid_patches: OrderedDict[str, PyramidPatchCollection] = OrderedDict()
        self._target_shapes: OrderedDict[str, Tuple[int, int]] = OrderedDict()
        self._pca_trained = False
        self._pca_training_patches: List[np.ndarray] = []

        # Bayes-factor pipeline (Option-A) - lazy-initialized on first use
        self._bayes_engine = None

    def _get_bayes_engine(self):
        """Lazily construct the Bayes-factor engine when scoring_mode='bayes'."""
        if self._bayes_engine is None:
            from spade.engine_bayes import BayesianForensicsEngine, BayesianConfig
            from pathlib import Path
            self._bayes_engine = BayesianForensicsEngine(BayesianConfig(
                stride=self.config.stride,
                entropy_threshold=None,
                k_neighbors=self.config.k_neighbors,
                noise_sigma=self.config.noise_sigma,
                priors_dir=Path(self.config.priors_dir) if self.config.priors_dir else None,
            ))
        return self._bayes_engine

    def _evict_lru_if_needed(self) -> None:
        """Evict least recently used target if max_targets limit reached."""
        if self.config.max_targets is None:
            return

        while len(self._target_shapes) >= self.config.max_targets:
            # Remove oldest (first) item from OrderedDict
            oldest_id = next(iter(self._target_shapes))
            logger.warning(
                f"Max targets ({self.config.max_targets}) reached. "
                f"Evicting oldest target: {oldest_id}"
            )
            self.remove_target(oldest_id)

    def index_target(
        self,
        image: np.ndarray,
        image_id: str,
        image_path: Optional[str] = None,
    ) -> int:
        """
        Index a target image for later matching.

        Args:
            image: RGB image as numpy array
            image_id: Unique identifier for this image
            image_path: Optional file path for lazy loading during verification

        Returns:
            Number of patches indexed

        Note:
            If max_targets is set and limit is reached, the least recently
            indexed target will be automatically evicted.
        """
        # Bayes-factor pipeline indexes through its own multi-scale machinery
        if self.config.scoring_mode == "bayes":
            engine = self._get_bayes_engine()
            engine.index_target(image, image_id)
            self._target_shapes[image_id] = (image.shape[0], image.shape[1])
            self._target_paths[image_id] = image_path
            return sum(
                len(engine._target_patches[s].get(image_id, []))
                for s in engine.cfg.patch_sizes
            )

        # Evict LRU target if memory limit reached
        self._evict_lru_if_needed()

        image = self._normalize_image(image)

        # Use pyramid-based extraction if enabled
        if self.pyramid is not None:
            return self._index_target_pyramid(image, image_id, image_path)

        patches = self.extractor.extract(image)

        if len(patches.patches) == 0:
            return 0

        # Collect patches for PCA training
        if self.config.auto_train_pca and not self._pca_trained:
            self._pca_training_patches.append(patches.patches)
            total_patches = sum(len(p) for p in self._pca_training_patches)
            if total_patches >= self.config.pca_train_samples:
                self._train_pca()

        descriptors = self.descriptor.compute_batch(patches.patches)

        metadata = [
            {"image_id": image_id, "coord": tuple(coord), "patch_idx": i}
            for i, coord in enumerate(patches.coords)
        ]

        self.index.add(descriptors, metadata)
        self._target_paths[image_id] = image_path
        self._target_patches[image_id] = patches
        self._target_shapes[image_id] = image.shape[:2]

        return len(patches.patches)

    def _index_target_pyramid(
        self,
        image: np.ndarray,
        image_id: str,
        image_path: Optional[str] = None,
    ) -> int:
        """Index target using multi-scale pyramid."""
        pyramid_patches = self.pyramid.extract_patches(image, self.extractor)

        if len(pyramid_patches.patches) == 0:
            return 0

        # Collect patches for PCA training
        if self.config.auto_train_pca and not self._pca_trained:
            self._pca_training_patches.append(pyramid_patches.patches)
            total_patches = sum(len(p) for p in self._pca_training_patches)
            if total_patches >= self.config.pca_train_samples:
                self._train_pca()

        descriptors = self.descriptor.compute_batch(pyramid_patches.patches)

        # Metadata includes pyramid level and scale for coordinate mapping
        metadata = [
            {
                "image_id": image_id,
                "coord": tuple(pyramid_patches.coords[i]),
                "original_coord": tuple(pyramid_patches.original_coords[i]),
                "patch_idx": i,
                "level": int(pyramid_patches.levels[i]),
                "scale": float(pyramid_patches.scales[i]),
            }
            for i in range(len(pyramid_patches.patches))
        ]

        self.index.add(descriptors, metadata)
        self._target_paths[image_id] = image_path
        self._target_pyramid_patches[image_id] = pyramid_patches
        self._target_shapes[image_id] = image.shape[:2]

        return len(pyramid_patches.patches)

    def _train_pca(self) -> None:
        """Train PCA on collected patches."""
        all_patches = np.concatenate(self._pca_training_patches, axis=0)
        self.descriptor.train_pca(all_patches, self.config.pca_train_samples)
        self._pca_trained = True
        self._pca_training_patches = []  # Free memory

    def match(
        self,
        source_image: np.ndarray,
        target_ids: Optional[List[str]] = None,
        return_heatmap: bool = True,
    ) -> MatchResult:
        """
        Match source image against indexed targets.

        Args:
            source_image: RGB image to search for
            target_ids: Optional list of specific targets to match against
            return_heatmap: Whether to generate aggregated heatmap

        Returns:
            MatchResult with matches and optional heatmap
        """
        # Dispatch to Bayes-factor pipeline when configured
        if self.config.scoring_mode == "bayes":
            engine = self._get_bayes_engine()
            br = engine.match(source_image, return_heatmap=return_heatmap)
            return MatchResult(
                matches=br.matches,
                best_match=br.best_match,
                heatmap=br.heatmap,
                coherent_regions=[],
                stats={**br.stats, "log_z": br.log_z},
            )

        source_image = self._normalize_image(source_image)
        source_patches = self.extractor.extract(source_image)

        if len(source_patches.patches) == 0:
            return MatchResult(
                matches=[],
                best_match=None,
                heatmap=None,
                coherent_regions=[],
                stats={"source_patches": 0, "total_matches": 0, "coherent_regions": 0},
            )

        source_descriptors = self.descriptor.compute_batch(source_patches.patches)
        dof = self.probability_model.compute_dof(self.config.patch_size)

        all_matches = []

        for i, (descriptor, source_patch) in enumerate(
            zip(source_descriptors, source_patches.patches)
        ):
            source_coord = tuple(source_patches.coords[i])
            search_results = self.index.search(
                descriptor, self.config.k_neighbors, lsh_candidates=self.config.lsh_candidates
            )

            for result in search_results:
                image_id = result.metadata.get("image_id", "")

                if target_ids is not None and image_id not in target_ids:
                    continue
                if result.distance > self.config.distance_threshold:
                    continue

                # Check if target was indexed with pyramid or regular extraction
                is_pyramid = image_id in self._target_pyramid_patches
                is_regular = image_id in self._target_patches

                if not is_pyramid and not is_regular:
                    continue

                patch_idx = result.metadata.get("patch_idx", 0)

                if is_pyramid:
                    # Pyramid-indexed target
                    target_pyr = self._target_pyramid_patches[image_id]
                    if patch_idx >= len(target_pyr.patches):
                        continue

                    target_patch = target_pyr.patches[patch_idx]
                    # Use original coordinates (mapped back to full-res image)
                    target_coord = tuple(target_pyr.original_coords[patch_idx])
                    target_scale = target_pyr.scales[patch_idx]
                else:
                    # Regular target
                    target_patches = self._target_patches[image_id]
                    if patch_idx >= len(target_patches.patches):
                        continue

                    target_patch = target_patches.patches[patch_idx]
                    target_coord = tuple(target_patches.coords[patch_idx])
                    target_scale = 1.0

                verification = self.verifier.verify(source_patch, target_patch)
                if not verification.success:
                    continue

                probability = self.probability_model.compute(verification.rss, dof)
                if probability < self.config.min_probability:
                    continue

                # Compute effective patch size (larger at lower pyramid levels)
                effective_size = int(self.config.patch_size / target_scale)

                all_matches.append(Match(
                    source_coord=source_coord,
                    target_coord=target_coord,
                    patch_size=effective_size,
                    probability=probability,
                    image_id=image_id,
                    color_M=verification.M,
                    color_b=verification.b,
                ))

        best_match = None
        if all_matches:
            best_match = max(all_matches, key=lambda m: m.probability)

        # Find spatially coherent regions (dramatically reduces false positives)
        coherent_regions = []
        if self.config.coherence_enabled and all_matches:
            coherent_regions = self.coherence_verifier.find_coherent_regions(all_matches)

        heatmap = None
        if return_heatmap and best_match is not None:
            target_shape = self._target_shapes.get(best_match.image_id)
            if target_shape is not None:
                # Use coherent matches if available, otherwise all matches
                if coherent_regions:
                    # Use matches from the best coherent region
                    target_matches = coherent_regions[0].matches
                else:
                    target_matches = [m for m in all_matches if m.image_id == best_match.image_id]
                heatmap = self.aggregator.aggregate(target_matches, target_shape)

        return MatchResult(
            matches=all_matches,
            best_match=best_match,
            heatmap=heatmap,
            coherent_regions=coherent_regions,
            stats={
                "source_patches": len(source_patches.patches),
                "total_matches": len(all_matches),
                "coherent_regions": len(coherent_regions),
            },
        )

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
            # Check for invalid values before division
            if not np.isfinite(max_val) or max_val == 0:
                if max_val == 0:
                    raise InvalidImageError("Image is completely black (all zeros)")
                else:
                    raise InvalidImageError("Image contains NaN or infinite values")
            result = image.astype(np.float32) / max_val

        # Final check for invalid values
        if np.any(~np.isfinite(result)):
            raise InvalidImageError("Image contains NaN or infinite values")

        return result

    # Public accessors for API and external use
    @property
    def target_ids(self) -> List[str]:
        """List of indexed target image IDs."""
        return list(self._target_shapes.keys())

    @property
    def target_count(self) -> int:
        """Number of indexed targets."""
        return len(self._target_shapes)

    def get_target_shape(self, image_id: str) -> Optional[Tuple[int, int]]:
        """Get (height, width) of indexed target, or None if not found."""
        return self._target_shapes.get(image_id)

    def get_target_path(self, image_id: str) -> Optional[str]:
        """Get file path of indexed target, or None if not found."""
        return self._target_paths.get(image_id)

    def has_target(self, image_id: str) -> bool:
        """Check if target is indexed."""
        return image_id in self._target_shapes

    def remove_target(self, image_id: str) -> bool:
        """
        Remove target metadata (patches, shapes, paths).

        Note: Descriptors remain in FAISS index but won't match due to
        metadata filtering. Returns True if removed, False if not found.
        """
        if image_id not in self._target_shapes:
            return False

        del self._target_shapes[image_id]
        self._target_paths.pop(image_id, None)
        self._target_patches.pop(image_id, None)
        self._target_pyramid_patches.pop(image_id, None)
        return True


def load_image(path: str) -> np.ndarray:
    """Load image from file path as RGB numpy array."""
    return np.array(Image.open(path).convert("RGB"))
