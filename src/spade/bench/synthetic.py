"""Synthetic recolored-splice generator with exact ground truth.

The forensic task SPADE targets: a region cut from a *donor* image is
recolored (affine color transform) and spliced into a different *host* image.
We generate that scenario and know exactly where the splice is and what
transform was applied - perfect ground truth for benchmarking, and full control
over the axes that matter (splice size, color-grade strength, JPEG quality).

A `SpliceSample` gives a detector two inputs - the `donor` (reference) and the
`tampered` image - and asks: where, in `tampered`, is the spliced region?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class SpliceSample:
    """One benchmark instance with ground truth."""
    donor: np.ndarray          # reference image (uint8 HxWx3), source of the fragment
    tampered: np.ndarray       # host image with the recolored fragment spliced in
    mask: np.ndarray           # bool HxW ground-truth mask (True = spliced pixels) in tampered coords
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) of the splice in tampered
    color_M: np.ndarray        # applied 3x3 color matrix (target = M @ source + b)
    color_b: np.ndarray        # applied bias (in [0,1] scale)
    params: dict               # generation parameters (size, grade, jpeg, seeds)


def _textured_image(rng: np.random.RandomState, h: int, w: int) -> np.ndarray:
    """Generate a natural-ish texture: 1/f multi-octave noise plus random blobs.

    A realistic benchmark must be fair to *both* dense (SPADE) and keypoint
    (SIFT/ORB) methods. Pure smoothed noise is too feature-poor for keypoint
    detectors; we add a 1/f spectrum (energy at all scales) and random hard-edged
    blobs (corners/edges) so detectors like SIFT find ample keypoints - making
    any remaining difficulty on small fragments a genuine result, not an artifact.
    """
    from scipy.ndimage import gaussian_filter

    img = np.zeros((h, w, 3), dtype=np.float32)
    for sigma, amp in ((8.0, 1.0), (4.0, 0.7), (2.0, 0.5), (1.0, 0.35), (0.0, 0.25)):
        octave = rng.rand(h, w, 3).astype(np.float32)
        if sigma > 0:
            octave = gaussian_filter(octave, sigma=(sigma, sigma, 0))
        img += amp * octave

    # Random hard-edged blobs create the corners/edges keypoint detectors need.
    yy, xx = np.ogrid[:h, :w]
    n_blobs = max(8, (h * w) // 350)
    for _ in range(n_blobs):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        r = rng.randint(2, max(3, h // 10))
        color = rng.rand(3).astype(np.float32)
        blob = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[blob] = color

    # Per-channel contrast stretch to [0, 1].
    for c in range(3):
        ch = img[:, :, c]
        lo, hi = ch.min(), ch.max()
        img[:, :, c] = (ch - lo) / max(hi - lo, 1e-6)
    return (img * 255).astype(np.uint8)


def _random_color_transform(
    rng: np.random.RandomState, strength: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Affine color transform near identity; `strength` scales the deviation.

    strength=0 -> identity (no recoloring). Typical values 0.05-0.3.
    """
    M = np.eye(3) + strength * rng.uniform(-1.0, 1.0, size=(3, 3))
    b = strength * rng.uniform(-0.15, 0.15, size=3)
    return M, b


def _apply_color_transform(region01: np.ndarray, M: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply target = M @ source + b in [0,1] space, clipped."""
    flat = region01.reshape(-1, 3)
    out = flat @ M.T + b
    return np.clip(out, 0.0, 1.0).reshape(region01.shape)


def _jpeg_roundtrip(image_u8: np.ndarray, quality: int) -> np.ndarray:
    """Recompress through JPEG at the given quality (simulates re-saving)."""
    import io

    buf = io.BytesIO()
    Image.fromarray(image_u8).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def make_recolored_splice(
    seed: int = 0,
    image_size: int = 96,
    fragment_size: int = 24,
    color_grade_strength: float = 0.15,
    jpeg_quality: Optional[int] = None,
) -> SpliceSample:
    """Generate one recolored-splice sample with ground truth.

    Args:
        seed: RNG seed (deterministic).
        image_size: side length of donor and host (square).
        fragment_size: side length of the spliced square region.
        color_grade_strength: magnitude of the recoloring (0 = none).
        jpeg_quality: if set, JPEG-recompress the tampered image at this quality.
    """
    rng = np.random.RandomState(seed)

    donor = _textured_image(rng, image_size, image_size)
    host = _textured_image(rng, image_size, image_size)

    fs = min(fragment_size, image_size)
    # Source region within the donor
    fx = rng.randint(0, image_size - fs + 1)
    fy = rng.randint(0, image_size - fs + 1)
    region01 = donor[fy:fy + fs, fx:fx + fs, :].astype(np.float32) / 255.0

    M, b = _random_color_transform(rng, color_grade_strength)
    recolored = _apply_color_transform(region01, M, b)
    recolored_u8 = (recolored * 255).astype(np.uint8)

    # Paste location within the host
    lx = rng.randint(0, image_size - fs + 1)
    ly = rng.randint(0, image_size - fs + 1)

    tampered = host.copy()
    tampered[ly:ly + fs, lx:lx + fs, :] = recolored_u8

    if jpeg_quality is not None:
        tampered = _jpeg_roundtrip(tampered, jpeg_quality)

    mask = np.zeros((image_size, image_size), dtype=bool)
    mask[ly:ly + fs, lx:lx + fs] = True

    return SpliceSample(
        donor=donor,
        tampered=tampered,
        mask=mask,
        bbox=(lx, ly, fs, fs),
        color_M=M,
        color_b=b,
        params={
            "seed": seed,
            "image_size": image_size,
            "fragment_size": fs,
            "color_grade_strength": color_grade_strength,
            "jpeg_quality": jpeg_quality,
            "donor_region": (fx, fy, fs, fs),
        },
    )


def make_negative_pair(seed: int = 0, image_size: int = 96) -> SpliceSample:
    """A negative sample: donor and tampered are unrelated (no splice).

    A correct detector should produce a low score and an empty mask. Used to
    measure detection (image-level) discrimination, not localization.
    """
    rng = np.random.RandomState(seed)
    donor = _textured_image(rng, image_size, image_size)
    tampered = _textured_image(rng, image_size, image_size)
    mask = np.zeros((image_size, image_size), dtype=bool)
    return SpliceSample(
        donor=donor,
        tampered=tampered,
        mask=mask,
        bbox=(0, 0, 0, 0),
        color_M=np.eye(3),
        color_b=np.zeros(3),
        params={"seed": seed, "image_size": image_size, "negative": True},
    )
