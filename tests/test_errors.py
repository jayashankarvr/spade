"""Tests for error handling and edge cases."""

import numpy as np
import pytest

from spade import ForensicsEngine, Config


class TestConfigValidation:
    def test_invalid_patch_size(self):
        with pytest.raises(ValueError, match="patch_size must be 3, 4, 5, or 6"):
            Config(patch_size=7)

    def test_invalid_stride(self):
        with pytest.raises(ValueError, match="stride must be >= 1"):
            Config(stride=0)

    def test_invalid_probability(self):
        with pytest.raises(ValueError, match="min_probability must be 0-1"):
            Config(min_probability=1.5)

    def test_invalid_noise_sigma(self):
        with pytest.raises(ValueError, match="noise_sigma must be > 0"):
            Config(noise_sigma=0)

    def test_valid_config(self):
        config = Config(patch_size=4, stride=2, min_probability=0.7)
        assert config.patch_size == 4
        assert config.stride == 2


class TestImageValidation:
    def test_grayscale_rejected(self):
        engine = ForensicsEngine()
        grayscale = np.random.rand(50, 50).astype(np.float32)

        with pytest.raises(ValueError, match="Expected RGB image"):
            engine.index_target(grayscale, "test")

    def test_rgba_rejected(self):
        engine = ForensicsEngine()
        rgba = np.random.rand(50, 50, 4).astype(np.float32)

        with pytest.raises(ValueError, match="Expected RGB image"):
            engine.index_target(rgba, "test")

    def test_nan_image_rejected(self):
        engine = ForensicsEngine()
        image = np.random.rand(50, 50, 3).astype(np.float32)
        image[10, 10, 0] = np.nan

        with pytest.raises(ValueError, match="NaN or infinite"):
            engine.index_target(image, "test")

    def test_inf_image_rejected(self):
        engine = ForensicsEngine()
        image = np.random.rand(50, 50, 3).astype(np.float32)
        image[10, 10, 0] = np.inf

        with pytest.raises(ValueError, match="NaN or infinite"):
            engine.index_target(image, "test")

    def test_all_black_rejected(self):
        engine = ForensicsEngine()
        image = np.zeros((50, 50, 3), dtype=np.float64)

        with pytest.raises(ValueError, match="completely black"):
            engine.index_target(image, "test")


class TestEdgeCases:
    def test_empty_index_search(self):
        engine = ForensicsEngine()
        source = np.random.rand(20, 20, 3).astype(np.float32)

        result = engine.match(source)

        assert result.stats["total_matches"] == 0
        assert result.best_match is None

    def test_very_small_source(self):
        engine = ForensicsEngine(Config(entropy_threshold=None))
        target = np.random.rand(50, 50, 3).astype(np.float32)
        source = np.random.rand(3, 3, 3).astype(np.float32)  # Minimum size

        engine.index_target(target, "target")
        result = engine.match(source)

        assert result.stats["source_patches"] == 1  # Exactly one patch

    def test_source_smaller_than_patch(self):
        engine = ForensicsEngine(Config(patch_size=5, entropy_threshold=None))
        target = np.random.rand(50, 50, 3).astype(np.float32)
        source = np.random.rand(3, 3, 3).astype(np.float32)  # Too small for 5x5

        engine.index_target(target, "target")
        result = engine.match(source)

        assert result.stats["source_patches"] == 0
