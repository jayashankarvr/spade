#!/usr/bin/env python3
"""Generate a synthetic natural-image corpus for training and benchmarking.

Uses fractional Brownian motion (1/f^beta power-law noise) to produce images
that share key statistical properties of natural images:

  - 1/f power spectrum     (Field 1987)
  - approximately edge-like local structure
  - heavy-tailed gradient distribution

This is only a smoke-test substitute for real natural-image corpora (BSDS500,
Open Images, etc.) - sufficient to exercise the training and benchmarking
pipelines end-to-end. For deployment-grade priors, train on real photos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def fractional_brownian_2d(
    size: int,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a 2-D fractional-Brownian field with power spectrum ~ 1/f^beta."""
    freqs_y = np.fft.fftfreq(size).reshape(-1, 1)
    freqs_x = np.fft.fftfreq(size).reshape(1, -1)
    f = np.sqrt(freqs_x ** 2 + freqs_y ** 2)
    f[0, 0] = 1.0
    spectrum = 1.0 / (f ** (beta / 2.0))
    spectrum[0, 0] = 0.0

    re = rng.standard_normal((size, size)) * spectrum
    im = rng.standard_normal((size, size)) * spectrum
    field = np.fft.ifft2(re + 1j * im).real

    # Normalize to [0, 1]
    field -= field.min()
    field /= max(field.max(), 1e-9)
    return field.astype(np.float32)


def synthesize_image(
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize one natural-like RGB image."""
    # Three slightly-correlated fBm channels with beta in [1.5, 2.5] - the
    # empirical range for natural images.
    beta = float(rng.uniform(1.6, 2.4))
    base = fractional_brownian_2d(size, beta, rng)

    # Add per-channel modulation
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        local = fractional_brownian_2d(size, beta + rng.uniform(-0.2, 0.2), rng)
        # Mix base structure with channel-specific texture
        rgb[..., c] = 0.6 * base + 0.4 * local

    # Tone-curve to spread mass across [0, 1]
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = rgb ** float(rng.uniform(0.7, 1.3))

    return (rgb * 255.0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for synthetic images")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of images to generate")
    parser.add_argument("--size", type=int, default=256,
                        help="Image size (square)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.n):
        img = synthesize_image(args.size, rng)
        Image.fromarray(img).save(args.out / f"synth_{i:04d}.png")

    print(f"Generated {args.n} synthetic images at {args.out}")


if __name__ == "__main__":
    main()
