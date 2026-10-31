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


class TestLearnedDescriptor:
    def test_shipped_sizes_and_shape(self):
        from spade.descriptors.learned import LearnedDescriptor, available_sizes

        assert {4, 5, 6, 8} <= set(available_sizes())
        d = LearnedDescriptor(5)
        desc, textured = d.compute_batch(_patches(size=5))
        assert desc.shape == (20, d.dim) and d.dim == 8
        np.testing.assert_allclose(np.linalg.norm(desc[textured], axis=1), 1.0, atol=1e-5)
        assert 0 < d.distance_threshold < 2

    def test_keeps_gain_offset_invariance(self):
        from spade.descriptors.learned import LearnedDescriptor

        p = _patches(size=5) * 0.6
        d = LearnedDescriptor(5)
        np.testing.assert_allclose(d.compute_batch(p)[0], d.compute_batch(p * 1.3 + 0.1)[0], atol=1e-5)

    def test_unknown_size_explains_how_to_train(self):
        from spade.descriptors.learned import LearnedDescriptor

        with pytest.raises(ValueError, match="train_descriptor"):
            LearnedDescriptor(7)

    def test_engine_uses_its_threshold(self):
        from spade import Config, ForensicsEngine

        engine = ForensicsEngine(Config(descriptor="learned"))
        assert engine.index.dim == 8
        assert engine.distance_threshold == pytest.approx(engine.descriptor.distance_threshold)
        assert ForensicsEngine(Config(descriptor="learned", distance_threshold=0.2)).distance_threshold == 0.2
