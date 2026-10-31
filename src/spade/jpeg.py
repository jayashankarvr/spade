"""Expected pixel error from a JPEG file's own quantization tables.

JPEG rounds each 8x8 DCT coefficient to a multiple of its table entry q, so
the rounding error is at most uniform on [-q/2, q/2], variance q^2 / 12. The
DCT is orthonormal, so the per-pixel error variance is the mean of those over
the 64 coefficients. That's an upper bound (small coefficients round to zero
with less error), and it lands close to what we measure on real files: about
0.023 at quality 85 and 0.038 at quality 75, on a 0-1 scale.

The matcher adds this to its base noise level, so a heavily compressed image
isn't held to the tolerance of a clean one.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
from PIL import Image


def sigma_from_tables(tables) -> Optional[float]:
    """Rounding-error std (0-1 scale) from a PIL ``image.quantization`` dict; table 0 is luminance."""
    if not tables:
        return None
    luma = np.asarray(tables[0], dtype=np.float64)
    return float(np.sqrt((luma ** 2).mean() / 12.0) / 255.0)


def quantization_sigma(path: str) -> Optional[float]:
    """Luminance rounding-error std (0-1 scale) of a JPEG file, or None if it isn't one."""
    try:
        with Image.open(path) as img:
            if img.format != "JPEG":
                return None
            return sigma_from_tables(getattr(img, "quantization", None))
    except OSError:
        return None


def quantization_sigma_for_quality(quality: int) -> float:
    """The same number for an image saved by PIL at ``quality`` (standard IJG tables)."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return sigma_from_tables(Image.open(buf).quantization)
