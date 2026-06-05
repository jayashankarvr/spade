"""Integration tests for the full SPADE pipeline."""

import tempfile
import numpy as np
import pytest
from PIL import Image

from spade import ForensicsEngine, Config
from spade.engine import load_image


class TestFullPipeline:
    def test_exact_fragment_detection(self):
        """Detect an exact copy of a fragment in a target image."""
        np.random.seed(42)
        # Create target image with distinct regions
        target = np.zeros((100, 100, 3), dtype=np.float32)
        target[20:40, 20:40, 0] = 0.8  # Red square
        target[60:80, 60:80, 2] = 0.7  # Blue square
        target += np.random.rand(100, 100, 3).astype(np.float32) * 0.1  # Noise

        # Extract fragment from target
        fragment = target[25:35, 25:35, :].copy()

        engine = ForensicsEngine(Config(
            patch_size=3,
            entropy_threshold=None,  # Keep all patches for small test
            min_probability=0.3,
            auto_train_pca=False,  # Disable PCA for small test
        ))

        engine.index_target(target, "target")
        result = engine.match(fragment)

        assert result.stats["source_patches"] > 0
        assert result.stats["total_matches"] > 0
        assert result.best_match is not None
        # Best match should be near the fragment location
        x, y = result.best_match.target_coord
        assert 15 <= x <= 45 and 15 <= y <= 45

    def test_brightness_shifted_fragment(self):
        """Detect fragment after brightness adjustment."""
        np.random.seed(123)
        target = np.random.rand(80, 80, 3).astype(np.float32) * 0.5 + 0.2
        fragment = target[30:45, 30:45, :].copy()
        fragment_bright = np.clip(fragment + 0.15, 0, 1)  # Brighten

        engine = ForensicsEngine(Config(
            patch_size=3,
            entropy_threshold=None,
            min_probability=0.3,
            auto_train_pca=False,
        ))

        engine.index_target(target, "target")
        result = engine.match(fragment_bright)

        assert result.best_match is not None
        x, y = result.best_match.target_coord
        assert 20 <= x <= 55 and 20 <= y <= 55

    def test_no_match_for_unrelated_images(self):
        """No matches for completely different images."""
        np.random.seed(42)
        target = np.random.rand(50, 50, 3).astype(np.float32)
        np.random.seed(123)
        source = np.random.rand(20, 20, 3).astype(np.float32)

        engine = ForensicsEngine(Config(
            patch_size=3,
            entropy_threshold=None,
            min_probability=0.8,  # High threshold
            auto_train_pca=False,
        ))

        engine.index_target(target, "target")
        result = engine.match(source)

        # Should have very few or no high-confidence matches
        assert result.stats["total_matches"] < 5

    def test_heatmap_generation(self):
        """Heatmap is generated with correct shape."""
        np.random.seed(42)
        target = np.random.rand(60, 80, 3).astype(np.float32)
        fragment = target[20:35, 30:50, :].copy()

        engine = ForensicsEngine(Config(
            patch_size=3,
            entropy_threshold=None,
            min_probability=0.3,
            auto_train_pca=False,
        ))

        engine.index_target(target, "target")
        result = engine.match(fragment, return_heatmap=True)

        if result.best_match:
            assert result.heatmap is not None
            assert result.heatmap.shape == (60, 80)
            assert result.heatmap.min() >= 0
            assert result.heatmap.max() <= 1

    def test_multiple_targets(self):
        """Match against multiple indexed targets."""
        np.random.seed(42)
        target1 = np.random.rand(50, 50, 3).astype(np.float32)
        np.random.seed(123)
        target2 = np.random.rand(50, 50, 3).astype(np.float32)
        fragment = target2[20:30, 20:30, :].copy()

        engine = ForensicsEngine(Config(
            patch_size=3,
            entropy_threshold=None,
            min_probability=0.3,
            auto_train_pca=False,
        ))

        engine.index_target(target1, "target1")
        engine.index_target(target2, "target2")
        result = engine.match(fragment)

        assert result.best_match is not None
        assert result.best_match.image_id == "target2"


class TestImageIO:
    def test_load_and_match_from_files(self):
        """Load images from files and run matching."""
        np.random.seed(42)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save test images
            target = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
            source = target[40:60, 40:60, :].copy()

            target_path = f"{tmpdir}/target.png"
            source_path = f"{tmpdir}/source.png"

            Image.fromarray(target).save(target_path)
            Image.fromarray(source).save(source_path)

            # Load and match
            target_img = load_image(target_path)
            source_img = load_image(source_path)

            engine = ForensicsEngine(Config(
                patch_size=3,
                entropy_threshold=None,
                min_probability=0.3,
                auto_train_pca=False,
            ))

            engine.index_target(target_img, "target")
            result = engine.match(source_img)

            assert result.stats["source_patches"] > 0

    def test_index_save_and_load(self):
        """Save and load index from disk."""
        np.random.seed(42)
        with tempfile.TemporaryDirectory() as tmpdir:
            target = np.random.rand(50, 50, 3).astype(np.float32)

            engine = ForensicsEngine(Config(
                patch_size=3,
                entropy_threshold=None,
                auto_train_pca=False,
            ))
            engine.index_target(target, "target")

            index_path = f"{tmpdir}/test_index"
            engine.index.save(index_path)

            # Load in new engine
            engine2 = ForensicsEngine(Config(
                patch_size=3,
                entropy_threshold=None,
                auto_train_pca=False,
            ))
            engine2.index.load(index_path)

            assert engine2.index.size == engine.index.size
