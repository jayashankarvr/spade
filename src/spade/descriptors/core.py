"""Photometric-invariant descriptor computation."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class DescriptorStrategy(ABC):
    """Base class for descriptor computation strategies."""

    @abstractmethod
    def compute(self, patch: np.ndarray) -> np.ndarray:
        """Compute descriptor for a single patch."""
        pass

    def compute_batch(self, patches: np.ndarray) -> np.ndarray:
        """Compute descriptors for multiple patches (vectorized where possible)."""
        return np.array([self.compute(p) for p in patches], dtype=np.float32)


def _to_gray(patch: np.ndarray) -> np.ndarray:
    """Convert RGB patch to grayscale using ITU-R BT.601."""
    return 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]


def _normalize(descriptor: np.ndarray) -> np.ndarray:
    """L2 normalize descriptor, handling zero vectors."""
    norm = np.linalg.norm(descriptor)
    if norm < 1e-8:
        return np.ones_like(descriptor) / np.sqrt(len(descriptor))
    return descriptor / norm


class DifferenceVectorDescriptor(DescriptorStrategy):
    """
    Computes difference vectors between all pixels and patch mean.
    Works for any patch size.
    """

    def compute(self, patch: np.ndarray) -> np.ndarray:
        h, w, c = patch.shape
        pixels = patch.reshape(-1, c)
        center = pixels.mean(axis=0)

        # Differences from mean for all pixels
        differences = pixels - center
        descriptor = differences.ravel()
        return _normalize(descriptor)


class ChromaticityDescriptor(DescriptorStrategy):
    """
    Computes chromaticity (color ratio) features.
    Invariant to brightness scaling: c = RGB / (R+G+B).
    """

    MIN_INTENSITY = 0.03  # Minimum intensity to avoid instability

    def compute(self, patch: np.ndarray) -> np.ndarray:
        h, w, c = patch.shape
        pixels = patch.reshape(-1, c)

        # Clamp intensity to avoid division issues with dark pixels
        intensity = pixels.sum(axis=1, keepdims=True)
        intensity = np.maximum(intensity, self.MIN_INTENSITY)
        chromaticity = pixels / intensity

        # Differences from mean chromaticity
        mean_chroma = chromaticity.mean(axis=0)
        differences = chromaticity - mean_chroma
        descriptor = differences.ravel()
        return _normalize(descriptor)


class LBPDescriptor(DescriptorStrategy):
    """
    Local Binary Pattern descriptor with uniform patterns.
    Compares all pixels to local neighborhood - works for any size.
    """

    def compute(self, patch: np.ndarray) -> np.ndarray:
        gray = _to_gray(patch)
        h, w = gray.shape

        # For each pixel (except border), compute 8-neighbor LBP
        patterns = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                center = gray[y, x]
                # 8-connected neighbors in clockwise order
                neighbors = [
                    gray[y-1, x-1], gray[y-1, x], gray[y-1, x+1],
                    gray[y, x+1], gray[y+1, x+1], gray[y+1, x],
                    gray[y+1, x-1], gray[y, x-1]
                ]
                pattern = [1.0 if n >= center else 0.0 for n in neighbors]
                patterns.extend(pattern)

        if not patterns:
            # Fallback for very small patches
            return _normalize(np.ones(8, dtype=np.float32))

        descriptor = np.array(patterns, dtype=np.float32)
        return _normalize(descriptor)


class RankOrderDescriptor(DescriptorStrategy):
    """
    Rank-order descriptor based on pairwise brightness comparisons.
    Highly invariant to photometric transformations.
    Limited to first 16 pixels to keep descriptor size reasonable.
    """

    MAX_PIXELS = 16  # Limit comparisons for larger patches

    def compute(self, patch: np.ndarray) -> np.ndarray:
        gray = _to_gray(patch)
        pixels = gray.ravel()

        # Limit number of pixels for tractable descriptor size
        if len(pixels) > self.MAX_PIXELS:
            # Sample evenly spaced pixels
            indices = np.linspace(0, len(pixels) - 1, self.MAX_PIXELS, dtype=int)
            pixels = pixels[indices]

        # Pairwise comparisons
        n = len(pixels)
        comparisons = []
        for i in range(n):
            for j in range(i + 1, n):
                comparisons.append(1.0 if pixels[i] > pixels[j] else 0.0)

        descriptor = np.array(comparisons, dtype=np.float32)
        return _normalize(descriptor)


class GradientHistogramDescriptor(DescriptorStrategy):
    """
    Histogram of gradient orientations (simplified HOG).
    Captures edge structure regardless of patch size.
    """

    NUM_BINS = 8

    def compute(self, patch: np.ndarray) -> np.ndarray:
        gray = _to_gray(patch)

        # Compute gradients
        gy, gx = np.gradient(gray)
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)  # -π to π

        # Bin orientations weighted by magnitude
        bins = np.linspace(-np.pi, np.pi, self.NUM_BINS + 1)
        hist = np.zeros(self.NUM_BINS, dtype=np.float32)

        for i in range(self.NUM_BINS):
            mask = (orientation >= bins[i]) & (orientation < bins[i + 1])
            hist[i] = magnitude[mask].sum()

        return _normalize(hist)


class CompositeDescriptor(DescriptorStrategy):
    """
    Combines multiple descriptor strategies and projects to target dimension.
    For patches larger than 3x3, uses spatial pooling of 3x3 subpatches.
    """

    def __init__(
        self,
        strategies: Optional[List[DescriptorStrategy]] = None,
        target_dim: int = 128,
        pca_matrix: Optional[np.ndarray] = None,
        spatial_pooling: bool = True,
    ):
        self.strategies = strategies or [
            DifferenceVectorDescriptor(),
            ChromaticityDescriptor(),
            LBPDescriptor(),
            RankOrderDescriptor(),
            GradientHistogramDescriptor(),
        ]
        self.target_dim = target_dim
        self.pca_matrix = pca_matrix
        # When False, descriptor strategies are computed directly on the whole
        # patch instead of pooling (w-2)*(h-2) 3x3 sub-patches. Pooling costs
        # ~(s-2)^2x more per patch for no measurable accuracy gain on larger
        # patches (benchmarked: 8x8 is 19x faster with identical IoU/precision),
        # so larger support becomes both more accurate and faster.
        self.spatial_pooling = spatial_pooling
        self._projection: Optional[np.ndarray] = None
        self._input_dim: Optional[int] = None

    def _raw_descriptor(self, patch: np.ndarray) -> np.ndarray:
        """Concatenated strategy outputs, optionally over pooled 3x3 sub-patches."""
        h, w, _ = patch.shape
        if not self.spatial_pooling or (h == 3 and w == 3):
            return np.concatenate([s.compute(patch) for s in self.strategies])
        subpatch_descriptors = []
        for y in range(h - 2):
            for x in range(w - 2):
                subpatch = patch[y:y+3, x:x+3, :]
                subpatch_descriptors.append(
                    np.concatenate([s.compute(subpatch) for s in self.strategies])
                )
        return np.concatenate(subpatch_descriptors)

    def compute(self, patch: np.ndarray) -> np.ndarray:
        concatenated = self._raw_descriptor(patch)
        projected = self._project(concatenated)
        return _normalize(projected)

    def _project(self, descriptor: np.ndarray) -> np.ndarray:
        if self.pca_matrix is not None:
            # Trained PCA projection
            if len(descriptor) != self.pca_matrix.shape[0]:
                # Dimension mismatch - pad or truncate input
                if len(descriptor) < self.pca_matrix.shape[0]:
                    descriptor = np.pad(descriptor, (0, self.pca_matrix.shape[0] - len(descriptor)))
                else:
                    descriptor = descriptor[:self.pca_matrix.shape[0]]
            result = descriptor @ self.pca_matrix
            # Pad output if PCA produced fewer dimensions than target
            if len(result) < self.target_dim:
                result = np.pad(result, (0, self.target_dim - len(result)))
            return result[:self.target_dim]

        # Deterministic random projection (fallback)
        if self._projection is None or self._input_dim != len(descriptor):
            self._input_dim = len(descriptor)
            rng = np.random.RandomState(42)
            self._projection = rng.randn(len(descriptor), self.target_dim).astype(np.float32)
            # Normalize columns for variance preservation
            self._projection /= np.linalg.norm(self._projection, axis=0, keepdims=True)

        return descriptor @ self._projection

    def compute_batch(self, patches: np.ndarray) -> np.ndarray:
        """Compute descriptors for batch of patches."""
        results = []
        for patch in patches:
            results.append(self.compute(patch))
        return np.array(results, dtype=np.float32)

    def train_pca(self, patches: np.ndarray, n_samples: int = 5000, random_state: int = 42) -> None:
        """
        Train PCA projection matrix from sample patches.

        Args:
            patches: Training patches, shape (N, H, W, 3)
            n_samples: Max samples to use (for memory efficiency)
            random_state: Random seed for reproducibility
        """
        from sklearn.decomposition import PCA

        # Subsample if too many patches
        if len(patches) > n_samples:
            rng = np.random.RandomState(random_state)
            indices = rng.choice(len(patches), n_samples, replace=False)
            patches = patches[indices]

        # Collect raw descriptors (honors the spatial_pooling setting)
        descriptors = [self._raw_descriptor(patch) for patch in patches]

        # All descriptors must have same dimension for PCA
        # Group by dimension and train on largest group
        by_dim = {}
        for d in descriptors:
            dim = len(d)
            if dim not in by_dim:
                by_dim[dim] = []
            by_dim[dim].append(d)

        # Use most common dimension
        most_common_dim = max(by_dim.keys(), key=lambda k: len(by_dim[k]))
        train_data = np.array(by_dim[most_common_dim])

        # Fit PCA
        n_components = min(self.target_dim, train_data.shape[1], len(train_data))
        pca = PCA(n_components=n_components)
        pca.fit(train_data)

        self.pca_matrix = pca.components_.T.astype(np.float32)
        self._input_dim = most_common_dim
