"""Natural-image priors for the Bayes-factor forensic scorer.

Public API:
    photometric_normalize  - Schur-Weyl projection onto the photometric-invariant subspace
    NaturalImagePrior      - abstract interface for P_0(patch)
    GMMDiffusionPrior      - classical GMM-on-diffusion-map prior loaded from disk
"""

from spade.priors.normalize import photometric_normalize
from spade.priors.natural import NaturalImagePrior, GMMDiffusionPrior

__all__ = ["photometric_normalize", "NaturalImagePrior", "GMMDiffusionPrior"]
