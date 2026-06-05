"""Tests for affine verification and probability model."""

import numpy as np
import pytest
from spade.verification.affine import AffineVerifier, VerificationResult
from spade.verification.probability import ProbabilityModel


class TestAffineVerifier:
    def test_identical_patches_have_low_rss(self):
        verifier = AffineVerifier(loss="l2")
        patch = np.random.rand(3, 3, 3).astype(np.float32)

        result = verifier.verify(patch, patch)

        assert result.success
        assert result.rss < 0.001  # RSS should be near zero
        np.testing.assert_array_almost_equal(result.M, np.eye(3), decimal=2)
        np.testing.assert_array_almost_equal(result.b, np.zeros(3), decimal=2)

    def test_recovers_brightness_shift(self):
        verifier = AffineVerifier(loss="l2")
        source = np.random.rand(3, 3, 3).astype(np.float32) * 0.5
        target = source + 0.2  # Constant brightness shift

        result = verifier.verify(source, target)

        assert result.success
        assert result.rss < 0.01
        np.testing.assert_array_almost_equal(result.b, [0.2, 0.2, 0.2], decimal=2)

    def test_recovers_color_scaling(self):
        verifier = AffineVerifier(loss="l2")
        source = np.random.rand(3, 3, 3).astype(np.float32) * 0.5

        target = source.copy()
        target[:, :, 0] *= 1.2  # Scale red channel
        target[:, :, 1] *= 0.9  # Scale green
        target[:, :, 2] *= 1.1  # Scale blue

        result = verifier.verify(source, target)

        assert result.success
        assert result.rss < 0.5  # RSS scaled by number of pixels

    def test_huber_more_robust_to_outliers(self):
        np.random.seed(42)
        source = np.random.rand(5, 5, 3).astype(np.float32) * 0.5
        target = source + 0.1  # Simple shift
        target[2, 2, :] = 0.9  # Add outlier

        verifier_l2 = AffineVerifier(loss="l2")
        verifier_huber = AffineVerifier(loss="huber")

        result_l2 = verifier_l2.verify(source, target)
        result_huber = verifier_huber.verify(source, target)

        # Huber should estimate bias closer to true value of 0.1
        huber_error = abs(result_huber.b.mean() - 0.1)
        l2_error = abs(result_l2.b.mean() - 0.1)
        assert huber_error <= l2_error + 1e-6

    def test_dof_calculation(self):
        verifier = AffineVerifier()

        # 3x3: 9*3 - 12 = 15
        result_3x3 = verifier.verify(
            np.random.rand(3, 3, 3).astype(np.float32),
            np.random.rand(3, 3, 3).astype(np.float32),
        )
        assert result_3x3.dof == 15

        # 5x5: 25*3 - 12 = 63
        result_5x5 = verifier.verify(
            np.random.rand(5, 5, 3).astype(np.float32),
            np.random.rand(5, 5, 3).astype(np.float32),
        )
        assert result_5x5.dof == 63

    def test_huber_degenerate_input_does_not_crash(self):
        """Regression: a singular system (constant patch, no regularization)
        used to raise UnboundLocalError in the Huber IRLS path because `rss`
        was only assigned after the matrix inversion. It must fail gracefully."""
        verifier = AffineVerifier(loss="huber", regularization=0.0)
        constant = np.full((3, 3, 3), 0.5, dtype=np.float32)

        result = verifier.verify(constant, constant)

        # Must return a result object, not raise
        assert isinstance(result, VerificationResult)
        assert not result.success
        assert result.rss == float("inf")

    def test_mismatched_shapes_fail(self):
        verifier = AffineVerifier()

        result = verifier.verify(
            np.random.rand(3, 3, 3).astype(np.float32),
            np.random.rand(4, 4, 3).astype(np.float32),
        )

        assert not result.success
        assert result.rss == float('inf')


class TestProbabilityModel:
    def test_low_rss_gives_high_probability(self):
        model = ProbabilityModel(noise_sigma=0.02)

        # Low RSS for a 3x3 patch (27 observations)
        probability = model.compute(rss=0.001, dof=15)

        assert probability > 0.5

    def test_high_rss_gives_low_probability(self):
        model = ProbabilityModel(noise_sigma=0.02)

        # High RSS indicates poor match
        probability = model.compute(rss=1.0, dof=15)

        assert probability < 0.1

    def test_probability_in_valid_range(self):
        model = ProbabilityModel()

        for rss in [0.001, 0.01, 0.1, 1.0]:
            for dof in [15, 36, 63, 96]:
                p = model.compute(rss, dof)
                assert 0.0 <= p <= 1.0

    def test_dof_formula(self):
        assert ProbabilityModel.compute_dof(3) == 15   # 27 - 12
        assert ProbabilityModel.compute_dof(4) == 36   # 48 - 12
        assert ProbabilityModel.compute_dof(5) == 63   # 75 - 12
        assert ProbabilityModel.compute_dof(6) == 96   # 108 - 12

    def test_invalid_inputs_return_zero(self):
        model = ProbabilityModel()

        assert model.compute(rss=-1, dof=15) == 0.0
        assert model.compute(rss=0.1, dof=0) == 0.0
