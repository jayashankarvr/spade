"""Patch extraction with entropy-based filtering."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from spade.logging_config import get_logger

logger = get_logger("extraction.patches")

try:
    from spade.acceleration import CUPY_AVAILABLE
    from spade.acceleration.cupy_ops import compute_entropy_batch_gpu
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


@dataclass
class PatchCollection:
    """Collection of extracted patches with metadata."""
    size: int
    patches: np.ndarray      # shape: (N, size, size, 3)
    coords: np.ndarray       # shape: (N, 2) as (x, y)
    entropy: Optional[np.ndarray] = None  # shape: (N,)


class PatchExtractor:
    """
    Extracts patches from images with optional entropy filtering.

    Uses numpy stride tricks for efficient extraction (10-100x faster).
    Entropy filtering removes low-information patches (uniform regions like
    sky or walls) which reduces false positives and improves performance.
    """

    def __init__(
        self,
        size: int = 3,
        stride: int = 1,
        entropy_threshold: Optional[float] = 2.5,
        use_gpu: bool = False,
    ):
        """
        Args:
            size: Patch size (e.g., 3 for 3x3)
            stride: Extraction stride
            entropy_threshold: Minimum entropy to keep patch (None to disable)
            use_gpu: Use GPU acceleration if available (requires CuPy)
        """
        self.size = size
        self.stride = stride
        self.entropy_threshold = entropy_threshold
        # Adaptive bins based on patch size (fewer bins for smaller patches)
        self._entropy_bins = min(8, size * size)

        # GPU support with graceful fallback
        self.use_gpu = use_gpu
        if use_gpu and not CUPY_AVAILABLE:
            logger.warning("CuPy not available, falling back to CPU")
            self.use_gpu = False

    def extract(self, image: np.ndarray) -> PatchCollection:
        """
        Extract patches from an RGB image.

        Args:
            image: RGB image, shape (H, W, 3), uint8 or float32

        Returns:
            PatchCollection with filtered patches
        """
        image = self._normalize(image)
        height, width, _ = image.shape

        # Check if image is too small
        if height < self.size or width < self.size:
            return self._empty_collection()

        # Use stride tricks for efficient extraction
        # This creates a view, not a copy - very memory efficient
        windows = sliding_window_view(image, (self.size, self.size, 3))
        # windows shape: (H-size+1, W-size+1, size, size, 3)

        # Apply stride
        windows = windows[::self.stride, ::self.stride]
        n_rows, n_cols = windows.shape[:2]

        # Reshape to (N, size, size, 3)
        all_patches = windows.reshape(-1, self.size, self.size, 3)

        # Generate coordinates
        ys = np.arange(0, height - self.size + 1, self.stride)
        xs = np.arange(0, width - self.size + 1, self.stride)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        all_coords = np.stack([xx.ravel(), yy.ravel()], axis=1)

        # Compute entropy for all patches (vectorized)
        all_entropy = self._compute_entropy_batch(all_patches)

        # Filter by entropy if threshold set
        if self.entropy_threshold is not None:
            mask = all_entropy > self.entropy_threshold
            patches = np.ascontiguousarray(all_patches[mask])
            coords = all_coords[mask]
            entropy = all_entropy[mask]
        else:
            patches = np.ascontiguousarray(all_patches)
            coords = all_coords
            entropy = all_entropy

        if len(patches) == 0:
            return self._empty_collection()

        return PatchCollection(
            size=self.size,
            patches=patches.astype(np.float32),
            coords=coords.astype(np.int32),
            entropy=entropy.astype(np.float32),
        )

    def _empty_collection(self) -> PatchCollection:
        """Return empty patch collection."""
        return PatchCollection(
            size=self.size,
            patches=np.zeros((0, self.size, self.size, 3), dtype=np.float32),
            coords=np.zeros((0, 2), dtype=np.int32),
            entropy=np.array([], dtype=np.float32),
        )

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Convert image to float32 in [0, 1] range."""
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return image

    def _compute_entropy_batch(self, patches: np.ndarray) -> np.ndarray:
        """Compute Shannon entropy for batch of patches."""
        n = len(patches)
        if n == 0:
            return np.array([], dtype=np.float32)

        # Use GPU if enabled
        if self.use_gpu and CUPY_AVAILABLE:
            patches_gpu = cp.asarray(patches)
            entropy_gpu = compute_entropy_batch_gpu(patches_gpu, self._entropy_bins)
            return cp.asnumpy(entropy_gpu)

        # CPU implementation
        # Convert to grayscale
        gray = (
            0.299 * patches[:, :, :, 0]
            + 0.587 * patches[:, :, :, 1]
            + 0.114 * patches[:, :, :, 2]
        )
        # gray shape: (N, size, size)

        # Flatten spatial dimensions
        gray_flat = gray.reshape(n, -1)

        # Compute histograms for all patches
        entropies = np.zeros(n, dtype=np.float32)
        for i in range(n):
            hist, _ = np.histogram(gray_flat[i], bins=self._entropy_bins, range=(0, 1))
            hist = hist / (hist.sum() + 1e-10)
            # Shannon entropy with proper handling of zero bins
            nonzero = hist > 0
            entropies[i] = -np.sum(hist[nonzero] * np.log2(hist[nonzero]))

        return entropies
