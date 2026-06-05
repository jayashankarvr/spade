"""Tests for descriptor computation."""

import numpy as np
import pytest
from spade.descriptors.core import (
    DifferenceVectorDescriptor,
    ChromaticityDescriptor,
    LBPDescriptor,
    RankOrderDescriptor,
    GradientHistogramDescriptor,
    CompositeDescriptor,
)


class TestDescriptors:
    def test_all_descriptors_are_normalized(self):
        np.random.seed(42)
        patch = np.random.rand(3, 3, 3).astype(np.float32) * 0.5 + 0.25

        descriptors = [
            DifferenceVectorDescriptor(),
            ChromaticityDescriptor(),
            LBPDescriptor(),
            RankOrderDescriptor(),
            GradientHistogramDescriptor(),
        ]

        for desc in descriptors:
            result = desc.compute(patch)
            norm = np.linalg.norm(result)
            assert abs(norm - 1.0) < 1e-5, f"{desc.__class__.__name__} not normalized: {norm}"

    def test_composite_output_dimension(self):
        descriptor = CompositeDescriptor(target_dim=128)
        patch = np.random.rand(3, 3, 3).astype(np.float32)

        result = descriptor.compute(patch)

        assert result.shape == (128,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_composite_is_deterministic(self):
        descriptor = CompositeDescriptor(target_dim=128)
        patch = np.random.rand(3, 3, 3).astype(np.float32)

        result1 = descriptor.compute(patch)
        result2 = descriptor.compute(patch)

        np.testing.assert_array_equal(result1, result2)

    def test_batch_processing(self):
        descriptor = CompositeDescriptor(target_dim=128)
        patches = np.random.rand(10, 3, 3, 3).astype(np.float32)

        results = descriptor.compute_batch(patches)

        assert results.shape == (10, 128)

    def test_larger_patch_sizes(self):
        """Test that descriptors work for 4x4, 5x5, 6x6 patches."""
        descriptor = CompositeDescriptor(target_dim=128)

        for size in [4, 5, 6]:
            patch = np.random.rand(size, size, 3).astype(np.float32)
            result = descriptor.compute(patch)

            assert result.shape == (128,)
            assert abs(np.linalg.norm(result) - 1.0) < 1e-5


class TestPhotometricInvariance:
    def test_lbp_invariant_to_monotonic_transform(self):
        """LBP should be invariant to monotonic transforms on grayscale-like patches."""
        descriptor = LBPDescriptor()
        # Use grayscale-like patch where all channels have same value
        gray_values = np.random.rand(3, 3).astype(np.float32) * 0.5 + 0.25
        patch = np.stack([gray_values] * 3, axis=-1)

        result_original = descriptor.compute(patch)
        result_gamma = descriptor.compute(patch ** 0.5)

        np.testing.assert_array_almost_equal(result_original, result_gamma, decimal=5)

    def test_chromaticity_invariant_to_brightness(self):
        """Chromaticity should be similar under brightness changes."""
        descriptor = ChromaticityDescriptor()
        patch = np.random.rand(3, 3, 3).astype(np.float32) * 0.3 + 0.2

        result_original = descriptor.compute(patch)
        result_brighter = descriptor.compute(patch * 1.5)

        # Should be similar (high dot product)
        similarity = np.dot(result_original, result_brighter)
        assert similarity > 0.9

    def test_different_patches_produce_different_descriptors(self):
        np.random.seed(42)
        descriptor = CompositeDescriptor(target_dim=128)

        patch1 = np.random.rand(3, 3, 3).astype(np.float32)
        patch2 = np.random.rand(3, 3, 3).astype(np.float32)

        result1 = descriptor.compute(patch1)
        result2 = descriptor.compute(patch2)

        # Should not be identical
        assert not np.allclose(result1, result2)


class TestPCATraining:
    def test_pca_training(self):
        """Test that PCA training works and improves consistency."""
        descriptor = CompositeDescriptor(target_dim=64)
        patches = np.random.rand(100, 3, 3, 3).astype(np.float32)

        # Train PCA
        descriptor.train_pca(patches)

        assert descriptor.pca_matrix is not None
        assert descriptor.pca_matrix.shape[1] == 64

        # Compute descriptor with trained PCA
        result = descriptor.compute(patches[0])
        assert result.shape == (64,)
