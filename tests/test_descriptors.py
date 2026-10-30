"""Tests for the luminance-pattern descriptor."""

import numpy as np
import pytest

from spade.descriptors.core import LuminanceDescriptor, luminance


def _patches(n=20, size=5, seed=0):
    return np.random.RandomState(seed).rand(n, size, size, 3).astype(np.float32)


class TestLuminanceDescriptor:
    def test_shape_and_unit_norm(self):
        desc, textured = LuminanceDescriptor(5).compute_batch(_patches())
        assert desc.shape == (20, 25)
        assert textured.all()
        np.testing.assert_allclose(np.linalg.norm(desc, axis=1), 1.0, atol=1e-5)

    def test_invariant_to_gain_and_offset(self):
        p = _patches(size=4) * 0.6
        d = LuminanceDescriptor(4)
        a, _ = d.compute_batch(p)
        b, _ = d.compute_batch(p * 1.4 + 0.1)
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_close_under_mild_color_grade(self):
        p = _patches(size=5) * 0.8 + 0.1
        M = np.eye(3) + 0.1 * np.random.RandomState(1).uniform(-1, 1, (3, 3))
        graded = np.clip(p @ M.T.astype(np.float32) + 0.02, 0, 1)
        d = LuminanceDescriptor(5)
        a, _ = d.compute_batch(p)
        b, _ = d.compute_batch(graded)
        unrelated, _ = d.compute_batch(_patches(size=5, seed=9))
        assert np.median(((a - b) ** 2).sum(1)) < np.median(((a - unrelated) ** 2).sum(1)) / 5

    def test_flat_patches_flagged(self):
        p = _patches(4, size=3)
        p[1] = 0.4
        p[2] = np.array([0.9, 0.1, 0.2])  # flat but colored
        desc, textured = LuminanceDescriptor(3).compute_batch(p)
        assert textured.tolist() == [True, False, False, True]
        assert not desc[1].any() and not desc[2].any()

    def test_wrong_patch_size_rejected(self):
        with pytest.raises(ValueError):
            LuminanceDescriptor(3).compute_batch(_patches(size=4))

    def test_empty_batch(self):
        desc, textured = LuminanceDescriptor(3).compute_batch(np.zeros((0, 3, 3, 3), np.float32))
        assert desc.shape == (0, 9) and textured.shape == (0,)

    def test_luminance_weights(self):
        assert luminance(np.ones((1, 3), np.float32))[0] == pytest.approx(1.0)
