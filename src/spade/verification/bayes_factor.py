"""Bayes-factor scorer for forensic patch correspondence.

For a pair (source patch s, target patch t), compare two generative models:

    M_1: s was copied from t and underwent an affine color transform plus
         iid Gaussian noise sigma. Likelihood from the existing affine
         verifier's residual sum of squares.

    M_0: s was drawn from the natural-image prior P_0 (a GMM on the
         diffusion-map manifold). Likelihood from prior.log_density.

Bayes factor:

    log BF = log P(s | M_1) - log P(s | M_0)

This is the calibrated forensic confidence in nats. Unlike the chi-square
p-value it replaces, it correctly downweights "easy" matches between
ubiquitous patches (high P_0 -> low BF) and amplifies matches between
unusual patches (low P_0 -> high BF) - automatic, parameter-free.

Implementation uses the residuals already computed by AffineVerifier, so
the additional cost is one prior log-density lookup per source patch
(amortized over all candidate targets it's compared against).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math
import numpy as np

from spade.priors.natural import NaturalImagePrior
from spade.verification.affine import AffineVerifier, VerificationResult


@dataclass
class BayesFactorResult:
    log_bf: float        # log Bayes factor in nats (positive = favors M_1)
    log_p_m1: float      # log likelihood under match model
    log_p_m0: float      # log likelihood under natural-image model
    rss: float           # residual sum of squares from the affine fit
    dof: int


class BayesFactorScorer:
    """Replacement for ProbabilityModel that returns calibrated log Bayes factors."""

    def __init__(
        self,
        prior: Optional[NaturalImagePrior] = None,
        verifier: Optional[AffineVerifier] = None,
        noise_sigma: float = 0.02,
    ):
        self.prior = prior
        self.verifier = verifier or AffineVerifier(noise_sigma=noise_sigma)
        self.noise_sigma = noise_sigma
        # log P(s | M_1) for a Gaussian iid noise model with diag covariance
        # = -0.5 * RSS / sigma^2  + const(dim)
        #   const = -0.5 * dim * log(2 pi sigma^2)
        self._inv_2sigma2 = 1.0 / (2.0 * noise_sigma * noise_sigma)
        self._log_norm_per_dim = -0.5 * math.log(2.0 * math.pi * noise_sigma * noise_sigma)

    def score(
        self,
        source_patch: np.ndarray,
        target_patch: np.ndarray,
        prior_log_density: Optional[float] = None,
    ) -> BayesFactorResult:
        """Score a single (source, target) pair. Source patch shape (H, W, 3)."""
        verification = self.verifier.verify(source_patch, target_patch)
        if not verification.success:
            return BayesFactorResult(
                log_bf=-float("inf"),
                log_p_m1=-float("inf"),
                log_p_m0=0.0,
                rss=float(verification.rss),
                dof=int(verification.dof),
            )
        return self._score_from_verification(verification, source_patch, prior_log_density)

    def score_with_prefit(
        self,
        verification: VerificationResult,
        source_patch: np.ndarray,
        prior_log_density: Optional[float] = None,
    ) -> BayesFactorResult:
        """Score using an already-computed AffineVerifier result (avoid recomputation)."""
        if not verification.success:
            return BayesFactorResult(
                log_bf=-float("inf"),
                log_p_m1=-float("inf"),
                log_p_m0=0.0,
                rss=float(verification.rss),
                dof=int(verification.dof),
            )
        return self._score_from_verification(verification, source_patch, prior_log_density)

    def _score_from_verification(
        self,
        v: VerificationResult,
        source_patch: np.ndarray,
        prior_log_density: Optional[float],
    ) -> BayesFactorResult:
        n_dims = int(np.prod(source_patch.shape))
        log_p_m1 = -float(v.rss) * self._inv_2sigma2 + n_dims * self._log_norm_per_dim

        if prior_log_density is None:
            if self.prior is not None:
                prior_log_density = float(
                    self.prior.log_density(source_patch[np.newaxis, ...])[0]
                )
            else:
                # No prior -> assume uniform over the unit cube; const cancels in BF
                # comparisons but absolute log_bf is then meaningless.
                prior_log_density = 0.0

        log_bf = log_p_m1 - prior_log_density
        return BayesFactorResult(
            log_bf=float(log_bf),
            log_p_m1=float(log_p_m1),
            log_p_m0=float(prior_log_density),
            rss=float(v.rss),
            dof=int(v.dof),
        )

    def score_batch(
        self,
        source_patches: np.ndarray,
        target_patches: np.ndarray,
    ) -> np.ndarray:
        """Vectorized batch scoring. source_patches and target_patches must align 1-to-1.

        Returns (N,) array of log BF values.
        """
        assert source_patches.shape == target_patches.shape, "shape mismatch"
        if self.prior is not None:
            prior_logs = self.prior.log_density(source_patches)
        else:
            prior_logs = np.zeros(len(source_patches), dtype=np.float32)

        out = np.empty(len(source_patches), dtype=np.float32)
        for i in range(len(source_patches)):
            v = self.verifier.verify(source_patches[i], target_patches[i])
            if not v.success:
                out[i] = -np.inf
                continue
            r = self._score_from_verification(v, source_patches[i], float(prior_logs[i]))
            out[i] = r.log_bf
        return out
