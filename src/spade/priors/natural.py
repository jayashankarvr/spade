"""Natural-image prior P_0(patch) for the Bayes-factor forensic scorer.

The forensic Bayes factor is

    log BF(Y) = log P(source | M_1: copied from target Y)
              - log P(source | M_0: drawn from natural images)

This module supplies log P(. | M_0). The interface is deliberately swappable:
any object exposing log_density(patches) -> (N,) can serve as the prior. The
default classical implementation is GMMDiffusionPrior, loaded from a joblib
file produced by scripts/train_priors.py.

A future neural normalizing-flow prior can be slotted in by subclassing
NaturalImagePrior - no changes elsewhere required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import numpy as np

from spade.priors.normalize import photometric_normalize


class NaturalImagePrior(ABC):
    """Abstract interface: log P_0(patch) per patch, in nats."""

    @property
    @abstractmethod
    def patch_size(self) -> int:
        """The N for which this prior accepts (B, N, N, 3) patches."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Dimensionality of the latent embedding the GMM lives in."""

    @abstractmethod
    def log_density(self, patches: np.ndarray) -> np.ndarray:
        """Return (B,) log-densities in nats for a batch of patches."""

    @abstractmethod
    def embed(self, patches: np.ndarray) -> np.ndarray:
        """Return (B, embed_dim) manifold coordinates for a batch of patches.

        Useful as a feature for downstream descriptors and as a debug surface.
        """


class GMMDiffusionPrior(NaturalImagePrior):
    """GMM density on a diffusion-map embedding of natural patches.

    Loaded from a joblib file produced by scripts/train_priors.py. Inference
    path: photometric_normalize -> Nystrom-extend to the trained embedding ->
    GaussianMixture.score_samples. CPU-only, ~1-2 microseconds per patch
    after the one-time Nystrom NN-index build.
    """

    def __init__(self, prior_path: Union[str, Path], k_neighbors: int = 25):
        import joblib
        from sklearn.neighbors import NearestNeighbors

        data = joblib.load(prior_path)
        if data.get("version") != 1:
            raise ValueError(
                f"unsupported prior version: {data.get('version')!r}"
            )

        self._size: int = int(data["patch_size"])
        self._embed_dim: int = int(data["embed_dim"])
        self._anchor_features: np.ndarray = data["anchor_features"]
        self._anchor_embedding: np.ndarray = data["anchor_embedding"]
        self._gmm = data["gmm"]
        self._k = k_neighbors
        self._nn = NearestNeighbors(n_neighbors=k_neighbors, n_jobs=-1).fit(
            self._anchor_features
        )

    @property
    def patch_size(self) -> int:
        return self._size

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def log_density(self, patches: np.ndarray) -> np.ndarray:
        embeds = self.embed(patches)
        return self._gmm.score_samples(embeds).astype(np.float32)

    def embed(self, patches: np.ndarray) -> np.ndarray:
        feats = photometric_normalize(patches)
        return self._nystrom(feats)

    def _nystrom(self, feats: np.ndarray) -> np.ndarray:
        """k-NN-weighted Nystrom extension into the trained embedding."""
        distances, indices = self._nn.kneighbors(feats)
        sigma = float(np.median(distances)) + 1e-8
        weights = np.exp(-(distances ** 2) / (2.0 * sigma * sigma))
        weights /= weights.sum(axis=1, keepdims=True)
        # (n, k) x (n, k, d) -> (n, d)
        return np.einsum("nk,nke->ne", weights, self._anchor_embedding[indices])
