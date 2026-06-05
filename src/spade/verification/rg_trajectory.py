"""Cross-scale RG-trajectory consistency term.

For natural images, descriptors at consecutive patch sizes follow a
predictable scaling relationship driven by 1/f power-law statistics. The
local Holder exponent alpha governs how a patch's energy decays as you
zoom in:

    descriptor_energy(s) ~ s^(2*alpha + d)

For a true forensic copy, source and target patches at corresponding
locations share the same alpha (point-by-point). For smoothed or blended
forgeries, the source's alpha drifts away from the target's because the
attacker disturbs the high-frequency end of the spectrum.

We compute alpha from the descriptor energy at sizes {3, 4, 5, 6} via
log-log regression and add a likelihood term

    log P(consistent | alpha_src, alpha_tgt) = -0.5 * ((alpha_src - alpha_tgt) / sigma_alpha) ^ 2

to the Bayes factor. Effectively free since all four sizes are computed
already.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class RGConfig:
    sigma_alpha: float = 0.4         # tolerance on Holder exponent agreement


def patch_energy(patches: np.ndarray) -> np.ndarray:
    """Mean squared deviation per patch (a proxy for spectral energy)."""
    if patches.size == 0:
        return np.zeros((0,), dtype=np.float32)
    n = len(patches)
    flat = patches.reshape(n, -1).astype(np.float32)
    centered = flat - flat.mean(axis=1, keepdims=True)
    return (centered * centered).mean(axis=1)


def holder_exponent(
    sizes: Sequence[int],
    energies: Sequence[float],
) -> float:
    """Estimate the local Holder exponent from energy(size).

    Uses the regression  log E ~ (2*alpha + 2) * log s  (d=2 for 2-D images),
    so alpha = (slope - 2) / 2.
    """
    sizes_arr = np.asarray(sizes, dtype=np.float64)
    energies_arr = np.asarray(energies, dtype=np.float64)
    valid = energies_arr > 1e-12
    if valid.sum() < 2:
        return 0.0
    log_s = np.log(sizes_arr[valid])
    log_e = np.log(energies_arr[valid])
    slope, _ = np.polyfit(log_s, log_e, 1)
    alpha = (slope - 2.0) / 2.0
    return float(alpha)


def rg_log_likelihood(alpha_src: float, alpha_tgt: float, cfg: RGConfig | None = None) -> float:
    """Gaussian log-likelihood that two alphas agree."""
    cfg = cfg or RGConfig()
    delta = (alpha_src - alpha_tgt) / cfg.sigma_alpha
    return -0.5 * float(delta * delta)


class RGConsistencyScorer:
    """Compute and cache Holder exponents for a stack of multi-scale patches."""

    def __init__(self, cfg: RGConfig | None = None):
        self.cfg = cfg or RGConfig()

    def alphas_from_pyramid(self, patches_per_size: dict[int, np.ndarray]) -> np.ndarray:
        """Estimate one alpha per source location.

        Args:
            patches_per_size: {size: (N, size, size, 3) patches} - one patch
                per source location at each size. All size keys must contain
                the same N patches in the same order.
        """
        sizes = sorted(patches_per_size.keys())
        if not sizes:
            return np.zeros((0,), dtype=np.float32)
        n = len(patches_per_size[sizes[0]])
        energies = np.empty((n, len(sizes)), dtype=np.float32)
        for j, s in enumerate(sizes):
            energies[:, j] = patch_energy(patches_per_size[s])
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            out[i] = holder_exponent(sizes, energies[i])
        return out

    def consistency_term(self, alpha_src: np.ndarray, alpha_tgt: np.ndarray) -> np.ndarray:
        """Element-wise log-likelihood of alpha agreement between paired vectors."""
        delta = (alpha_src.astype(np.float32) - alpha_tgt.astype(np.float32)) / self.cfg.sigma_alpha
        return -0.5 * delta * delta
