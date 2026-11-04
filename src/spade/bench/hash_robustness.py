"""How far perceptual hashes move under ordinary edits, and how far apart unrelated photos are.

For every scikit-image benchmark photo this hashes the photo and five edited
copies (resize 0.5x and 2x, JPEG quality 70, colour grade 0.15, 5% crop) and
prints, per algorithm and edit, the largest distance from a photo to its edited
copy. It also prints the smallest distance between two different photos. A
threshold is safe on this benchmark when it sits between the two.

The benchmark is 13 photos, so these numbers say where a threshold clearly
fails, not that it holds in general.

Usage:
    python -m spade.bench.hash_robustness
"""

from __future__ import annotations

import argparse
import itertools
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from spade.bench.datasets import load_photos
from spade.bench.synthetic import _apply_color_transform, _jpeg_roundtrip, _random_color_transform
from spade.hashing import DEFAULT_DISTANCE, hamming, hash_image


def _resize(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    return np.asarray(Image.fromarray(img).resize(size, Image.BICUBIC))


def _grade(img: np.ndarray, strength: float = 0.15, seed: int = 0) -> np.ndarray:
    M, b = _random_color_transform(np.random.RandomState(seed), strength)
    out = _apply_color_transform(img.astype(np.float64) / 255.0, M, b)
    return np.round(out * 255.0).astype(np.uint8)


def _crop(img: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    """Keep the centre, trimming ``fraction`` of each dimension (half from each side)."""
    h, w = img.shape[:2]
    dy, dx = round(h * fraction / 2), round(w * fraction / 2)
    return img[dy:h - dy, dx:w - dx]


# The edits the matching thresholds are meant to survive, plus the crop, which
# is reported but not required (perceptual hashes are not crop-invariant).
EDITS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "resize 0.5x": lambda img: _resize(img, 0.5),
    "resize 2x": lambda img: _resize(img, 2.0),
    "jpeg 70": lambda img: _jpeg_roundtrip(img, 70),
    "grade 0.15": _grade,
    "crop 5%": _crop,
}
REQUIRED_EDITS = ("resize 0.5x", "resize 2x", "jpeg 70", "grade 0.15")


def measure_pairs(
    algorithms: Sequence[str] = ("pdq", "phash"),
    photos: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[Tuple[str, str, str], int], Dict[Tuple[str, str, str], int]]:
    """Every distance the summary is built from.

    Returns ``(edited, unrelated)``: ``edited[(algo, photo, edit)]`` is the
    distance between a photo and its edited copy, ``unrelated[(algo, a, b)]``
    the distance between two different photos.
    """
    if photos is None:
        photos = load_photos()
    originals: Dict[str, Dict[str, str]] = {}
    edited: Dict[Tuple[str, str, str], int] = {}
    for name, img in photos.items():
        originals[name] = {h.algorithm: h.hex for h in hash_image(img, algorithms)}
        for edit, fn in EDITS.items():
            for h in hash_image(fn(img), algorithms):
                edited[(h.algorithm, name, edit)] = hamming(originals[name][h.algorithm], h.hex)
    unrelated = {
        (algo, a, b): hamming(originals[a][algo], originals[b][algo])
        for algo in algorithms
        for a, b in itertools.combinations(sorted(originals), 2)
    }
    return edited, unrelated


def summarize(edited, unrelated):
    """Largest distance per algorithm and edit; closest unrelated pair per algorithm.

    Returns ``(worst, closest)``: ``worst[algo][edit] = (distance, photo)`` and
    ``closest[algo] = (distance, photo_a, photo_b)``.
    """
    worst: Dict[str, Dict[str, Tuple[int, str]]] = {}
    for (algo, photo, edit), d in edited.items():
        cur = worst.setdefault(algo, {}).get(edit)
        if cur is None or d > cur[0]:
            worst[algo][edit] = (d, photo)
    closest: Dict[str, Tuple[int, str, str]] = {}
    for (algo, a, b), d in unrelated.items():
        if algo not in closest or d < closest[algo][0]:
            closest[algo] = (d, a, b)
    return worst, closest


def format_table(worst, closest) -> str:
    algos = list(worst)
    rows: List[List[str]] = [["", *algos]]
    for edit in EDITS:
        rows.append([f"{edit} (largest)", *(str(worst[a][edit][0]) for a in algos)])
    rows.append(["unrelated (smallest)", *(str(closest[a][0]) for a in algos)])
    rows.append(["DEFAULT_DISTANCE", *(str(DEFAULT_DISTANCE[a]) for a in algos)])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = [
        "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths)))
        for r in rows
    ]
    lines.append("")
    for a in algos:
        for edit in EDITS:
            d, photo = worst[a][edit]
            lines.append(f"{a}, {edit}: largest on {photo} ({d} bits)")
        d, x, y = closest[a]
        lines.append(f"{a}, unrelated: closest pair {x} / {y} ({d} bits)")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Perceptual hash distances on the scikit-image photos")
    p.add_argument("--algorithm", action="append", choices=("pdq", "phash", "phash-lib"),
                   help="repeatable; default pdq and phash")
    args = p.parse_args(argv)
    worst, closest = summarize(*measure_pairs(args.algorithm or ("pdq", "phash")))
    print(format_table(worst, closest))


if __name__ == "__main__":
    main()
