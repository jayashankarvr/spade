"""Composite multi-invariant descriptor for the Bayes-factor pipeline.

Concatenates four complementary signal sources per patch:

  1. Persistence-image vector       (monotone-tone invariant; topology)
  2. Zernike magnitude vector       (rotation-invariant; algebraic moments)
  3. Photometric-residual vector    (Schur-Weyl; photometric-orbit invariant)
  4. Manifold coordinates           (diffusion-map embedding from prior, optional)

Total dimension is matched to the patch's actual rank rather than padded to
a uniform 256-D - so 3x3 patches produce ~30 dims, 6x6 produce ~100 dims.
This avoids the "256-D from 27 raw scalars" inflation in the legacy
CompositeDescriptor and gives FAISS a cleaner index to work on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from spade.priors.normalize import photometric_normalize
from spade.priors.natural import NaturalImagePrior
from spade.descriptors.persistence import PersistenceDescriptor, PersistenceImageConfig
from spade.descriptors.zernike import ZernikeDescriptor, feature_dim_for_size


@dataclass
class BayesianDescriptorConfig:
    persistence: PersistenceImageConfig = PersistenceImageConfig()
    include_persistence: bool = True
    include_zernike: bool = True
    include_photometric: bool = True
    include_manifold: bool = True


class BayesianDescriptor:
    """Composite descriptor used by the Bayes-factor scorer."""

    def __init__(
        self,
        cfg: Optional[BayesianDescriptorConfig] = None,
        prior: Optional[NaturalImagePrior] = None,
    ):
        self.cfg = cfg or BayesianDescriptorConfig()
        self.prior = prior
        self._persistence = PersistenceDescriptor(self.cfg.persistence) if self.cfg.include_persistence else None
        self._zernike = ZernikeDescriptor() if self.cfg.include_zernike else None

    def feature_dim(self, size: int) -> int:
        d = 0
        if self.cfg.include_persistence:
            d += self.cfg.persistence.resolution ** 2
        if self.cfg.include_zernike:
            d += feature_dim_for_size(size)
        if self.cfg.include_photometric:
            d += size * size * 3
        if self.cfg.include_manifold and self.prior is not None and self.prior.patch_size == size:
            d += self.prior.embed_dim
        return d

    def compute_batch(self, patches: np.ndarray) -> np.ndarray:
        """patches: (N, H, W, 3) float in [0, 1]."""
        if patches.size == 0:
            n, h, w, _ = patches.shape
            return np.zeros((n, self.feature_dim(h)), dtype=np.float32)

        n, h, w, _ = patches.shape
        parts = []

        if self._persistence is not None:
            parts.append(np.stack([self._persistence.compute(p) for p in patches]))

        if self._zernike is not None:
            parts.append(np.stack([self._zernike.compute(p) for p in patches]))

        if self.cfg.include_photometric:
            parts.append(photometric_normalize(patches))

        if (
            self.cfg.include_manifold
            and self.prior is not None
            and self.prior.patch_size == h
        ):
            parts.append(self.prior.embed(patches).astype(np.float32))

        out = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        # L2-normalize the final descriptor for cosine/IP search compatibility.
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.maximum(norms, 1e-8, out=norms)
        return out / norms
