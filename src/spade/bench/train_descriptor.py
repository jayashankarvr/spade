"""Fit the learned linear descriptor (``spade.descriptors.learned``).

Usage:
    python -m spade.bench.train_descriptor --sizes 4 5 6 8

For each patch size: cut patches from photo crops and from edited copies of
the same crops (random color grade + JPEG at a random quality), take the
normalized luminance pattern of both, and solve

    St w = lambda Sw w

where Sw is the covariance of (original - edited) pattern differences - how
edits move a patch - and St the covariance of the patterns themselves - how
patches differ from each other. Directions with large lambda tell patches
apart while ignoring what edits do; it's LDA with each patch as its own
class. Photos are split by name: two thirds fit, one third reports.

Writes ``src/spade/descriptors/data/linear_{s}x{s}.npz`` with the projection
(columns sorted by lambda) and the distance threshold that lets the same
share of unrelated pairs through as the luminance descriptor's 0.5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import eigh

from spade.bench.datasets import load_photos
from spade.bench.synthetic import _apply_color_transform, _jpeg_roundtrip, _random_color_transform
from spade.descriptors.core import LuminanceDescriptor
from spade.descriptors.learned import DEFAULT_DIMS, data_path

LUMINANCE_THRESHOLD = 0.5


def _pattern_pairs(rng, photos, names, size, n_crops, qualities, crop=96, per_crop=400):
    descriptor = LuminanceDescriptor(size)
    originals, edited = [], []
    for _ in range(n_crops):
        image = photos[names[rng.randint(len(names))]]
        y, x = rng.randint(0, image.shape[0] - crop), rng.randint(0, image.shape[1] - crop)
        a = image[y:y + crop, x:x + crop]
        M, b = _random_color_transform(rng, 0.15)
        b_img = _jpeg_roundtrip((_apply_color_transform(a / 255.0, M, b) * 255).astype(np.uint8),
                                int(rng.choice(qualities)))
        pa = sliding_window_view(a / 255.0, (size, size, 3)).reshape(-1, size, size, 3).astype(np.float32)
        pb = sliding_window_view(b_img / 255.0, (size, size, 3)).reshape(-1, size, size, 3).astype(np.float32)
        da, ta = descriptor.compute_batch(pa)
        db, tb = descriptor.compute_batch(pb)
        ok = np.nonzero(ta & tb)[0]
        pick = rng.choice(ok, min(per_crop, len(ok)), replace=False)
        originals.append(da[pick])
        edited.append(db[pick])
    return np.concatenate(originals), np.concatenate(edited)


def _project(x, w):
    y = x @ w
    return y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-9)


def _recall_at(a, b, unrelated, fp=0.001):
    true_d = ((a - b) ** 2).sum(1)
    false_d = ((a - unrelated) ** 2).sum(1)
    return float((true_d <= np.quantile(false_d, fp)).mean())


def train(size: int, dims: int = DEFAULT_DIMS, seed: int = 0, out_dir: Path = None) -> dict:
    rng = np.random.RandomState(seed)
    photos = load_photos()
    names = sorted(photos)
    fit_names = [n for i, n in enumerate(names) if i % 3 != 0]
    report_names = [n for i, n in enumerate(names) if i % 3 == 0]

    a, b = _pattern_pairs(rng, photos, fit_names, size, 300, (70, 75, 85, 90, 95))
    within = np.cov((a - b).T) + 1e-6 * np.eye(size * size)
    total = np.cov(a.T)
    values, vectors = eigh(total, within)
    projection = vectors[:, np.argsort(values)[::-1]].astype(np.float32)

    # Threshold letting through the same share of unrelated pairs as before.
    shuffled = a[rng.permutation(len(a))]
    share = float((((a - shuffled) ** 2).sum(1) <= LUMINANCE_THRESHOLD).mean())
    w = projection[:, :dims]
    threshold = float(np.quantile(((_project(a, w) - _project(shuffled, w)) ** 2).sum(1), share))

    ta, tb = _pattern_pairs(rng, photos, report_names, size, 120, (70, 80, 90))
    unrelated = tb[rng.permutation(len(tb))]
    report = {
        "size": size,
        "dims": dims,
        "threshold": threshold,
        "held_out_recall_luminance": _recall_at(ta, tb, unrelated),
        "held_out_recall_learned": _recall_at(_project(ta, w), _project(tb, w), _project(unrelated, w)),
    }
    path = Path(out_dir) / f"linear_{size}x{size}.npz" if out_dir else data_path(size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, projection=projection, threshold=np.float32(threshold), dims=np.int32(dims))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[5])
    parser.add_argument("--dims", type=int, default=DEFAULT_DIMS)
    args = parser.parse_args(argv)
    for size in args.sizes:
        r = train(size, args.dims)
        print(f"{size}x{size}: held-out recall at 0.1% false matches {r['held_out_recall_luminance']:.3f} (luminance) "
              f"-> {r['held_out_recall_learned']:.3f} (learned, {r['dims']}-d), threshold {r['threshold']:.3f}")


if __name__ == "__main__":
    main()
