"""Image pyramid generation for multi-scale analysis."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

from spade.extraction.patches import PatchExtractor, PatchCollection


@dataclass
class PyramidLevel:
    """A single level of the image pyramid."""
    level: int                    # 0 = original, 1 = half, etc.
    scale: float                  # Scale factor (1.0, 0.5, 0.25, ...)
    image: np.ndarray             # Scaled image
    shape: Tuple[int, int]        # (height, width)


@dataclass
class PyramidPatchCollection:
    """Patches extracted from a pyramid with level metadata."""
    patches: np.ndarray           # shape: (N, size, size, 3)
    coords: np.ndarray            # shape: (N, 2) - coords at pyramid level
    original_coords: np.ndarray   # shape: (N, 2) - coords in original image
    levels: np.ndarray            # shape: (N,) - which level each patch came from
    scales: np.ndarray            # shape: (N,) - scale factor for each patch
    entropy: Optional[np.ndarray] = None


class ImagePyramid:
    """
    Multi-resolution image pyramid for scale-invariant matching.

    Creates pyramid levels at powers of 2 downscaling (1x, 0.5x, 0.25x, 0.125x).
    Patches extracted at lower levels correspond to larger regions in the original.
    """

    DEFAULT_SCALES = [1.0, 0.5, 0.25, 0.125]

    def __init__(
        self,
        scales: Optional[List[float]] = None,
        min_size: int = 16,
    ):
        """
        Args:
            scales: Scale factors for each level (default: [1.0, 0.5, 0.25, 0.125])
            min_size: Minimum dimension to create a level (skip if smaller)
        """
        self.scales = scales or self.DEFAULT_SCALES
        self.min_size = min_size

    def build(self, image: np.ndarray) -> List[PyramidLevel]:
        """
        Build image pyramid from source image.

        Args:
            image: RGB image, shape (H, W, 3), float32 in [0, 1]

        Returns:
            List of PyramidLevel from finest to coarsest
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        height, width = image.shape[:2]
        levels = []

        for i, scale in enumerate(self.scales):
            new_h = int(height * scale)
            new_w = int(width * scale)

            # Skip if too small
            if new_h < self.min_size or new_w < self.min_size:
                continue

            if scale == 1.0:
                scaled = image
            else:
                # Use PIL for high-quality downsampling (Lanczos)
                pil_img = Image.fromarray((image * 255).astype(np.uint8))
                pil_scaled = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                scaled = np.array(pil_scaled).astype(np.float32) / 255.0

            levels.append(PyramidLevel(
                level=i,
                scale=scale,
                image=scaled,
                shape=(new_h, new_w),
            ))

        return levels

    def extract_patches(
        self,
        image: np.ndarray,
        extractor: PatchExtractor,
    ) -> PyramidPatchCollection:
        """
        Extract patches from all pyramid levels.

        Args:
            image: Source RGB image
            extractor: PatchExtractor to use

        Returns:
            PyramidPatchCollection with patches from all levels
        """
        levels = self.build(image)

        all_patches = []
        all_coords = []
        all_original_coords = []
        all_levels = []
        all_scales = []
        all_entropy = []

        for pyr_level in levels:
            collection = extractor.extract(pyr_level.image)

            if len(collection.patches) == 0:
                continue

            # Transform coordinates back to original image space
            # coord_original = coord_level / scale
            # Use round instead of truncation to avoid off-by-one errors
            original_coords = np.round(collection.coords / pyr_level.scale).astype(np.int32)

            all_patches.append(collection.patches)
            all_coords.append(collection.coords)
            all_original_coords.append(original_coords)
            all_levels.append(np.full(len(collection.patches), pyr_level.level, dtype=np.int32))
            all_scales.append(np.full(len(collection.patches), pyr_level.scale, dtype=np.float32))
            if collection.entropy is not None:
                all_entropy.append(collection.entropy)

        if not all_patches:
            return PyramidPatchCollection(
                patches=np.zeros((0, extractor.size, extractor.size, 3), dtype=np.float32),
                coords=np.zeros((0, 2), dtype=np.int32),
                original_coords=np.zeros((0, 2), dtype=np.int32),
                levels=np.array([], dtype=np.int32),
                scales=np.array([], dtype=np.float32),
                entropy=np.array([], dtype=np.float32),
            )

        return PyramidPatchCollection(
            patches=np.concatenate(all_patches),
            coords=np.concatenate(all_coords),
            original_coords=np.concatenate(all_original_coords),
            levels=np.concatenate(all_levels),
            scales=np.concatenate(all_scales),
            entropy=np.concatenate(all_entropy) if all_entropy else None,
        )


def compute_effective_patch_size(patch_size: int, scale: float) -> float:
    """
    Compute effective patch size in original image coordinates.

    A 3x3 patch at scale 0.5 covers a 6x6 region in the original image.

    Args:
        patch_size: Base patch size
        scale: Pyramid scale factor

    Returns:
        Effective size in original pixels
    """
    return patch_size / scale
