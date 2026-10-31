"""Chi-square goodness-of-fit test for a single patch correspondence."""

import numpy as np
from scipy import stats


class ProbabilityModel:
    """
    Turns an affine-fit residual into a goodness-of-fit p-value.

    If the target patch really is the source patch after an affine color change
    plus Gaussian noise of std ``noise_sigma``, then RSS / sigma^2 follows a
    chi-square distribution with ``dof`` degrees of freedom. The value returned
    is the upper-tail p-value of that test.

    Read it as "the fit is consistent with a copy", not as the probability that
    the patches match: unrelated tiny patches pass this test often, because a
    12-parameter color fit has a lot of freedom on 27 numbers. Whether a set of
    matches is significant is decided at region level (see ``verification.nfa``).
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

    def compute(self, rss, dof):
        """p-value for one RSS or an array of them; ``dof`` may be a scalar or per-element."""
        rss_arr = np.asarray(rss, dtype=np.float64)
        dof_arr = np.asarray(dof, dtype=np.float64)
        p = stats.chi2.sf(rss_arr / self.noise_sigma ** 2, np.maximum(dof_arr, 1))
        p = np.where((rss_arr < 0) | (dof_arr <= 0) | ~np.isfinite(rss_arr), 0.0, p)
        return float(p) if p.ndim == 0 else p
