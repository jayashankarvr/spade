"""Chi-square probability model for match scoring."""

import numpy as np
from scipy import stats


class ProbabilityModel:
    """
    Converts verification residuals to match probabilities.

    Under the null hypothesis (patches match with affine color transform),
    the sum of squared residuals follows a chi-square distribution.
    """

    def __init__(self, noise_sigma: float = 0.02):
        """
        Args:
            noise_sigma: Expected per-channel noise std in [0, 1] range.
                0.01 = very clean images
                0.02 = standard quality (default)
                0.05 = noisy or heavily compressed
        """
        self.noise_sigma = noise_sigma

    def compute(self, rss: float, dof: int) -> float:
        """
        Compute match probability from residual sum of squares.

        Args:
            rss: Residual sum of squares from affine verification
            dof: Degrees of freedom (n_pixels * 3 - 12)

        Returns:
            Probability in [0, 1], higher = more likely a match
        """
        if dof <= 0 or rss < 0:
            return 0.0

        # Under null hypothesis: RSS/σ^2 ~ χ^2(dof)
        variance = self.noise_sigma ** 2
        test_stat = rss / variance

        # P-value: probability of seeing this RSS or larger if match is true
        p_value = 1.0 - stats.chi2.cdf(test_stat, dof)

        return float(np.clip(p_value, 0.0, 1.0))

    @staticmethod
    def compute_dof(patch_size: int) -> int:
        """
        Compute degrees of freedom for a patch size.

        Formula: dof = (patch_size^2 x 3 channels) - 12 affine params
        The 12 params are: 3x3 matrix M + 3x1 bias b

        Examples:
            3x3 -> 15, 4x4 -> 36, 5x5 -> 63, 6x6 -> 96
        """
        return max(1, patch_size ** 2 * 3 - 12)
