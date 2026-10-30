"""Tests for error handling and edge cases."""

import numpy as np
import pytest

from spade import ForensicsEngine, Config


class TestConfigValidation:
    def test_invalid_patch_size(self):
        with pytest.raises(ValueError, match="patch_size must be between 3 and 16"):
            Config(patch_size=17)
        with pytest.raises(ValueError, match="patch_size must be between 3 and 16"):
            Config(patch_size=2)

    def test_invalid_stride(self):
        with pytest.raises(ValueError, match="stride must be >= 1"):
            Config(stride=0)

    def test_invalid_probability(self):
        with pytest.raises(ValueError, match="min_probability must be 0-1"):
            Config(min_probability=1.5)

    def test_invalid_noise_sigma(self):
        with pytest.raises(ValueError, match="noise_sigma must be > 0"):
            Config(noise_sigma=0)

    def test_unreachable_entropy_threshold_rejected(self):
        # The old README example (3.0 bits on 3x3) silently dropped every patch.
        with pytest.raises(ValueError, match="nothing would pass"):
            Config(patch_size=3, entropy_threshold=3.0)
        Config(patch_size=3, entropy_threshold=2.0)

    def test_invalid_max_nfa(self):
        with pytest.raises(ValueError, match="max_nfa"):
            Config(max_nfa=0)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("SPADE_PATCH_SIZE", "6")
        monkeypatch.setenv("SPADE_ENTROPY_THRESHOLD", "1.5")
        config = Config()
        assert config.patch_size == 6 and config.entropy_threshold == 1.5

    def test_config_dict_roundtrip(self):
        config = Config(patch_size=4, max_nfa=0.1)
        assert Config.from_dict(config.to_dict()) == config

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
        engine = ForensicsEngine(Config(patch_size=3, entropy_threshold=None))
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
