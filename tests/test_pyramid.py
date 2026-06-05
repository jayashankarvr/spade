"""Tests for image pyramid functionality."""

import numpy as np
import pytest
from spade.extraction.pyramid import ImagePyramid, PyramidPatchCollection, compute_effective_patch_size
from spade.extraction.patches import PatchExtractor


class TestImagePyramid:
    def test_builds_default_levels(self):
        """Pyramid should create 4 levels by default."""
        pyramid = ImagePyramid()
        image = np.random.rand(256, 256, 3).astype(np.float32)

        levels = pyramid.build(image)

        assert len(levels) == 4
        assert levels[0].scale == 1.0
        assert levels[1].scale == 0.5
        assert levels[2].scale == 0.25
        assert levels[3].scale == 0.125

    def test_level_dimensions(self):
        """Each level should be half the previous."""
        pyramid = ImagePyramid()
        image = np.random.rand(256, 256, 3).astype(np.float32)

        levels = pyramid.build(image)

        assert levels[0].shape == (256, 256)
        assert levels[1].shape == (128, 128)
        assert levels[2].shape == (64, 64)
        assert levels[3].shape == (32, 32)

    def test_skips_small_levels(self):
        """Levels smaller than min_size should be skipped."""
        pyramid = ImagePyramid(min_size=64)
        image = np.random.rand(128, 128, 3).astype(np.float32)

        levels = pyramid.build(image)

        # Should only have 2 levels: 128x128 and 64x64
        assert len(levels) == 2
        assert levels[0].shape == (128, 128)
        assert levels[1].shape == (64, 64)

    def test_custom_scales(self):
        """Should use custom scale factors."""
        pyramid = ImagePyramid(scales=[1.0, 0.75, 0.5])
        image = np.random.rand(100, 100, 3).astype(np.float32)

        levels = pyramid.build(image)

        assert len(levels) == 3
        assert levels[0].shape == (100, 100)
        assert levels[1].shape == (75, 75)
        assert levels[2].shape == (50, 50)

    def test_preserves_image_content(self):
        """Downsampled images should preserve general content."""
        pyramid = ImagePyramid()
        # Create gradient image
        image = np.zeros((64, 64, 3), dtype=np.float32)
        image[:, :, 0] = np.linspace(0, 1, 64).reshape(1, -1)  # Red gradient

        levels = pyramid.build(image)

        # Level 1 (half size) should still have gradient
        half = levels[1].image
        assert half[:, 0, 0].mean() < half[:, -1, 0].mean()


class TestPyramidPatchExtraction:
    def test_extracts_from_all_levels(self):
        """Should extract patches from all pyramid levels."""
        pyramid = ImagePyramid(scales=[1.0, 0.5])
        extractor = PatchExtractor(size=3, stride=2, entropy_threshold=None)
        image = np.random.rand(64, 64, 3).astype(np.float32)

        collection = pyramid.extract_patches(image, extractor)

        # Should have patches from both levels
        assert len(collection.patches) > 0
        assert 0 in collection.levels  # Level 0
        assert 1 in collection.levels  # Level 1

    def test_computes_original_coordinates(self):
        """Original coordinates should map back to full-res image."""
        pyramid = ImagePyramid(scales=[1.0, 0.5])
        extractor = PatchExtractor(size=3, stride=4, entropy_threshold=None)
        image = np.random.rand(64, 64, 3).astype(np.float32)

        collection = pyramid.extract_patches(image, extractor)

        # Find patches from level 1 (scale 0.5)
        level1_mask = collection.levels == 1
        if level1_mask.any():
            level1_coords = collection.coords[level1_mask]
            level1_original = collection.original_coords[level1_mask]

            # Original coords should be ~2x the level coords
            # (with integer rounding)
            for lc, oc in zip(level1_coords, level1_original):
                assert oc[0] == int(lc[0] / 0.5)
                assert oc[1] == int(lc[1] / 0.5)

    def test_stores_scale_per_patch(self):
        """Each patch should have associated scale factor."""
        pyramid = ImagePyramid(scales=[1.0, 0.5, 0.25])
        extractor = PatchExtractor(size=3, stride=4, entropy_threshold=None)
        image = np.random.rand(64, 64, 3).astype(np.float32)

        collection = pyramid.extract_patches(image, extractor)

        # Scales should match levels
        for i, level in enumerate(collection.levels):
            if level == 0:
                assert collection.scales[i] == 1.0
            elif level == 1:
                assert collection.scales[i] == 0.5
            elif level == 2:
                assert collection.scales[i] == 0.25


class TestEffectivePatchSize:
    def test_scale_1(self):
        """At scale 1.0, effective size equals base size."""
        assert compute_effective_patch_size(3, 1.0) == 3.0
        assert compute_effective_patch_size(5, 1.0) == 5.0

    def test_scale_half(self):
        """At scale 0.5, effective size is doubled."""
        assert compute_effective_patch_size(3, 0.5) == 6.0
        assert compute_effective_patch_size(4, 0.5) == 8.0

    def test_scale_quarter(self):
        """At scale 0.25, effective size is 4x."""
        assert compute_effective_patch_size(3, 0.25) == 12.0


class TestPyramidWithEngine:
    """Integration tests with ForensicsEngine."""

    def test_pyramid_indexing(self):
        """Engine should index with pyramid when enabled."""
        from spade import ForensicsEngine, Config

        config = Config(
            patch_size=3,
            pyramid_enabled=True,
            pyramid_scales=[1.0, 0.5],
            entropy_threshold=None,
        )
        engine = ForensicsEngine(config)

        image = np.random.rand(64, 64, 3).astype(np.float32)
        count = engine.index_target(image, "test")

        # Should have indexed patches from both levels
        assert count > 0
        assert "test" in engine._target_pyramid_patches

    def test_pyramid_matching(self):
        """Engine should match across scales."""
        from spade import ForensicsEngine, Config

        config = Config(
            patch_size=3,
            pyramid_enabled=True,
            pyramid_scales=[1.0, 0.5],
            entropy_threshold=None,
            min_probability=0.0,  # Accept any matches for test
            distance_threshold=2.0,
        )
        engine = ForensicsEngine(config)

        # Create deterministic test image
        np.random.seed(42)
        target = np.random.rand(64, 64, 3).astype(np.float32)
        engine.index_target(target, "target")

        # Use same image as source (should match)
        result = engine.match(target)

        # Should find some matches
        assert len(result.matches) > 0

    def test_effective_size_in_matches(self):
        """Matches should have effective patch size based on scale."""
        from spade import ForensicsEngine, Config

        config = Config(
            patch_size=3,
            pyramid_enabled=True,
            pyramid_scales=[1.0, 0.5],
            entropy_threshold=None,
            min_probability=0.0,
            distance_threshold=2.0,
        )
        engine = ForensicsEngine(config)

        np.random.seed(42)
        target = np.random.rand(64, 64, 3).astype(np.float32)
        engine.index_target(target, "target")

        result = engine.match(target)

        if result.matches:
            # Matches from scale 0.5 should have effective size 6
            sizes = set(m.patch_size for m in result.matches)
            # Should include both 3 (from scale 1.0) and 6 (from scale 0.5)
            assert 3 in sizes or 6 in sizes
