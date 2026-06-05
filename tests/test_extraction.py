"""Tests for patch extraction."""

import numpy as np
import pytest
from spade.extraction.patches import PatchExtractor, PatchCollection


class TestPatchExtractor:
    def test_basic_extraction(self):
        extractor = PatchExtractor(size=3, entropy_threshold=None)
        image = np.random.rand(10, 10, 3).astype(np.float32)

        result = extractor.extract(image)

        assert isinstance(result, PatchCollection)
        assert result.size == 3
        assert len(result.patches) == 64  # (10-3+1)^2
        assert result.patches.shape == (64, 3, 3, 3)

    def test_entropy_filtering_reduces_patches(self):
        # Create image with uniform region (low entropy)
        image = np.zeros((20, 20, 3), dtype=np.float32)
        image[10:, :, :] = np.random.rand(10, 20, 3)  # Add texture to half

        extractor_unfiltered = PatchExtractor(size=3, entropy_threshold=None)
        extractor_filtered = PatchExtractor(size=3, entropy_threshold=2.5)

        result_unfiltered = extractor_unfiltered.extract(image)
        result_filtered = extractor_filtered.extract(image)

        assert len(result_filtered.patches) < len(result_unfiltered.patches)

    def test_entropy_values_stored(self):
        extractor = PatchExtractor(size=3, entropy_threshold=None)
        image = np.random.rand(10, 10, 3).astype(np.float32)

        result = extractor.extract(image)

        assert result.entropy is not None
        assert len(result.entropy) == len(result.patches)
        # Entropy should be in valid range [0, 4] for 16 bins
        assert all(0 <= e <= 4 for e in result.entropy)

    def test_coordinates_are_correct(self):
        extractor = PatchExtractor(size=3, stride=1, entropy_threshold=None)
        image = np.random.rand(5, 5, 3).astype(np.float32)

        result = extractor.extract(image)

        # First patch at (0, 0), last at (2, 2)
        assert tuple(result.coords[0]) == (0, 0)
        assert tuple(result.coords[-1]) == (2, 2)

    def test_handles_uint8_input(self):
        extractor = PatchExtractor(size=3, entropy_threshold=None)
        image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

        result = extractor.extract(image)

        assert result.patches.dtype == np.float32
        assert result.patches.max() <= 1.0

    def test_empty_result_for_small_image(self):
        extractor = PatchExtractor(size=5, entropy_threshold=None)
        image = np.random.rand(3, 3, 3).astype(np.float32)

        result = extractor.extract(image)

        assert len(result.patches) == 0
        assert result.patches.shape == (0, 5, 5, 3)

    def test_stride_affects_patch_count(self):
        image = np.random.rand(10, 10, 3).astype(np.float32)

        extractor_stride1 = PatchExtractor(size=3, stride=1, entropy_threshold=None)
        extractor_stride2 = PatchExtractor(size=3, stride=2, entropy_threshold=None)

        result1 = extractor_stride1.extract(image)
        result2 = extractor_stride2.extract(image)

        assert len(result1.patches) > len(result2.patches)
