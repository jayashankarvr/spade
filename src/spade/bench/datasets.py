"""Benchmark datasets beyond the synthetic generator.

Every source yields ``SpliceSample`` objects: positives (``label=1``) with a
donor image and a tampered image containing a region copied from it, and
negatives (``label=0``) pairing images that share nothing.

Sources:

* ``photos``   - splices built from the public-domain photos bundled with
                 scikit-image: real camera noise and compression history,
                 exact ground truth. Small and offline; good for tests and
                 quick runs.
* ``casia2``   - CASIA v2 cross-image splices (``Tp_D_*``). The two authentic
                 images a splice was made from are named in the file name;
                 which one is the donor is worked out from pixels, not by
                 SPADE (see ``_split_donor_host``).
* ``manifest`` - any dataset described by a CSV of donor,tampered,mask,label.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from spade.bench.synthetic import (
    SpliceSample,
    _apply_color_transform,
    _apply_tone_curve,
    _jpeg_roundtrip,
    _random_color_transform,
    _random_tone_curve,
    make_copy_move,
    make_negative_pair,
    make_recolored_splice,
)


def data_dir() -> Path:
    """Where downloaded datasets live: $SPADE_DATA_DIR or ~/.cache/spade."""
    return Path(os.environ.get("SPADE_DATA_DIR", Path.home() / ".cache" / "spade"))


def _read_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


# ----------------------------------------------------------------------
# scikit-image photos
# ----------------------------------------------------------------------

# Natural photos only. The right half of the motorcycle stereo pair is left out:
# it shows the same scene as the left one and would make a mislabeled negative.
_PHOTO_NAMES = (
    "astronaut", "chelsea", "coffee", "rocket", "hubble_deep_field",
    "immunohistochemistry", "retina", "camera", "brick", "grass", "gravel", "moon",
)
# Photos that already contain a duplicated region, so a copy-move test can't
# treat them as clean. SPADE finds these, and any crop covering both copies was
# scored as a false alarm. Found by running match_self on every full photo:
#   grass:  rows 47-77 at x 290-376 reappear 162 px lower, 3 px right (corr 0.91)
#   gravel: the strip at y 0-50, x 352-477 reappears 273 px lower, 36 px right
# (retina also matched, but only its circular rim against its own mirror image;
# match_self now drops symmetric objects like that.)
_SELF_DUPLICATED = {"grass", "gravel"}


def load_photos() -> Dict[str, np.ndarray]:
    """name -> uint8 RGB photo (greyscale ones replicated to three channels)."""
    try:
        import skimage.data as skdata
    except ImportError as e:  # pragma: no cover - depends on the bench extra
        raise ImportError("the photos dataset needs scikit-image: pip install spade-forensics[bench]") from e

    photos = {}
    for name in _PHOTO_NAMES:
        img = getattr(skdata, name)()
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        photos[name] = np.ascontiguousarray(img[:, :, :3])
    left, _ = skdata.stereo_motorcycle()[:2]
    photos["motorcycle"] = np.ascontiguousarray(left)
    return photos


def _random_crop(rng: np.random.RandomState, img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    y = rng.randint(0, h - size + 1)
    x = rng.randint(0, w - size + 1)
    return img[y:y + size, x:x + size]


def _textured_spot(rng: np.random.RandomState, img: np.ndarray, size: int, tries: int = 30) -> Tuple[int, int]:
    """Top-left of a size x size window with real texture (flat sky makes a useless test)."""
    h, w = img.shape[:2]
    best, best_std = (0, 0), -1.0
    for _ in range(tries):
        y, x = rng.randint(0, h - size + 1), rng.randint(0, w - size + 1)
        std = float(img[y:y + size, x:x + size].std())
        if std > best_std:
            best, best_std = (x, y), std
        if std > 25:
            break
    return best


def _cut_fragment(rng, image, size, rotation, flip):
    """A size x size textured fragment, optionally rotated (uniform in +-rotation degrees) and mirrored.

    Cut from a larger window and rotated about its centre, so the fragment is
    all real content (no empty corners). Returns (fragment uint8, info dict).
    """
    window = int(np.ceil(size * 1.5)) + 2 if rotation else size
    x, y = _textured_spot(rng, image, window)
    patch = image[y:y + window, x:x + window]
    angle = float(rng.uniform(-rotation, rotation)) if rotation else 0.0
    if angle:
        patch = np.asarray(Image.fromarray(patch).rotate(angle, resample=Image.BICUBIC))
    off = (window - size) // 2
    fragment = patch[off:off + size, off:off + size]
    mirrored = bool(flip and rng.rand() < 0.5)
    if mirrored:
        fragment = fragment[:, ::-1]
    return np.ascontiguousarray(fragment), {"window": (x, y, window), "angle": angle, "mirrored": mirrored}


def photo_splices(
    n: int,
    negatives: int = 0,
    crop_size: int = 192,
    fragment_size: int = 40,
    color_grade_strength: float = 0.15,
    jpeg_quality: Optional[int] = 90,
    seed: int = 0,
    tone_curve_strength: float = 0.0,
    rotation: float = 0.0,
    flip: bool = False,
) -> List[SpliceSample]:
    """``n`` recolored splices between different photos, plus ``negatives`` unrelated crops.

    ``rotation`` (max degrees) and ``flip`` (mirror half the fragments) make
    geometric variants.
    """
    photos = load_photos()
    names = sorted(photos)
    rng = np.random.RandomState(seed)
    samples = []

    for i in range(n):
        donor_name, host_name = rng.choice(names, 2, replace=False)
        donor = _random_crop(rng, photos[donor_name], crop_size)
        host = _random_crop(rng, photos[host_name], crop_size)

        fragment, geometry = _cut_fragment(rng, donor, fragment_size, rotation, flip)
        region = fragment.astype(np.float32) / 255.0
        M, b = _random_color_transform(rng, color_grade_strength)
        recolored = _apply_color_transform(region, M, b)
        if tone_curve_strength > 0:
            recolored = _apply_tone_curve(recolored, _random_tone_curve(rng, tone_curve_strength))
        recolored = (recolored * 255).astype(np.uint8)

        lx = rng.randint(0, crop_size - fragment_size + 1)
        ly = rng.randint(0, crop_size - fragment_size + 1)
        tampered = host.copy()
        tampered[ly:ly + fragment_size, lx:lx + fragment_size] = recolored
        if jpeg_quality is not None:
            tampered = _jpeg_roundtrip(tampered, jpeg_quality)

        mask = np.zeros(tampered.shape[:2], dtype=bool)
        mask[ly:ly + fragment_size, lx:lx + fragment_size] = True
        samples.append(SpliceSample(
            donor=np.ascontiguousarray(donor),
            tampered=tampered,
            mask=mask,
            bbox=(lx, ly, fragment_size, fragment_size),
            color_M=M,
            color_b=b,
            params={"source": "photos", "index": i, "donor": donor_name, "host": host_name,
                    "jpeg_quality": jpeg_quality, **geometry},
        ))

    for i in range(negatives):
        a, b_name = rng.choice(names, 2, replace=False)
        tampered = _random_crop(rng, photos[b_name], crop_size)
        if jpeg_quality is not None:
            tampered = _jpeg_roundtrip(np.ascontiguousarray(tampered), jpeg_quality)
        samples.append(SpliceSample(
            donor=np.ascontiguousarray(_random_crop(rng, photos[a], crop_size)),
            tampered=np.ascontiguousarray(tampered),
            mask=np.zeros((crop_size, crop_size), dtype=bool),
            params={"source": "photos", "index": i, "donor": a, "host": b_name, "negative": True},
            label=0,
        ))
    return samples


def photo_copy_moves(
    n: int,
    negatives: int = 0,
    crop_size: int = 192,
    fragment_size: int = 40,
    color_grade_strength: float = 0.0,
    jpeg_quality: Optional[int] = 90,
    seed: int = 0,
    rotation: float = 0.0,
    flip: bool = False,
) -> List[SpliceSample]:
    """Within-image copy-moves on photo crops (``donor`` is None), plus untouched crops."""
    photos = load_photos()
    names = sorted(set(photos) - _SELF_DUPLICATED)
    rng = np.random.RandomState(seed)
    samples = []
    fs = fragment_size
    for i in range(n + negatives):
        name = names[rng.randint(len(names))]
        image = np.ascontiguousarray(_random_crop(rng, photos[name], crop_size))
        mask = np.zeros(image.shape[:2], dtype=bool)
        params = {"source": "photos", "index": i, "photo": name, "jpeg_quality": jpeg_quality}
        if i < n:
            fragment, geometry = _cut_fragment(rng, image, fs, rotation, flip)
            wx, wy, window = geometry["window"]
            sx, sy = wx + (window - fs) // 2, wy + (window - fs) // 2
            while True:
                dx, dy = rng.randint(0, crop_size - fs + 1, size=2)
                if abs(int(dx) - sx) >= fs or abs(int(dy) - sy) >= fs:
                    break
            region = fragment.astype(np.float32) / 255.0
            M, b = _random_color_transform(rng, color_grade_strength)
            image = image.copy()
            image[dy:dy + fs, dx:dx + fs] = (_apply_color_transform(region, M, b) * 255).astype(np.uint8)
            mask[sy:sy + fs, sx:sx + fs] = True
            mask[dy:dy + fs, dx:dx + fs] = True
            params.update(copy_move=True, source_region=(sx, sy, fs, fs), bbox=(int(dx), int(dy), fs, fs), **geometry)
        if jpeg_quality is not None:
            image = _jpeg_roundtrip(image, jpeg_quality)
        samples.append(SpliceSample(donor=None, tampered=image, mask=mask, params=params, label=int(i < n)))
    return samples


# ----------------------------------------------------------------------
# CASIA v2
# ----------------------------------------------------------------------

CASIA2_KAGGLE_REF = "divg07/casia-20-image-tampering-detection-dataset"


def _casia_au_path(au_dir: Path, image_id: str) -> Optional[Path]:
    """'ani00018' -> Au_ani_00018.jpg (or whatever extension it has)."""
    stem = f"Au_{image_id[:3]}_{image_id[3:]}"
    for ext in (".jpg", ".JPG", ".jpeg", ".png", ".bmp", ".tif"):
        path = au_dir / (stem + ext)
        if path.exists():
            return path
    return None


def _split_donor_host(tampered: np.ndarray, mask: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Return (donor, host) among two authentic candidates.

    The host is the image the splice was pasted into, so outside the mask it
    agrees with the tampered image almost pixel for pixel. Decided from raw
    pixels so the benchmark never uses SPADE to label its own test data.
    Returns None when neither candidate fits.
    """
    outside = ~mask

    def agreement(candidate):
        if candidate.shape != tampered.shape:
            return -1.0
        t = tampered[outside].astype(np.float32).ravel()
        c = candidate[outside].astype(np.float32).ravel()
        t -= t.mean()
        c -= c.mean()
        return float((t @ c) / (np.linalg.norm(t) * np.linalg.norm(c) + 1e-9))

    score_a, score_b = agreement(a), agreement(b)
    if max(score_a, score_b) < 0.9:
        return None
    return (b, a) if score_a > score_b else (a, b)


def casia2(
    root: Optional[str] = None,
    n: Optional[int] = None,
    negatives: int = 0,
    seed: int = 0,
    copy_move: bool = False,
) -> List[SpliceSample]:
    """CASIA v2 cross-image splices as (donor, tampered, mask) pairs.

    With ``copy_move=True``: the within-image forgeries (``Tp_S_*``) instead,
    with ``donor=None``; negatives are untouched authentic images. CASIA's
    masks mark the pasted copy only, so localization recall is the fair
    number there (SPADE also marks the original).

    Args:
        root: dataset folder containing ``CASIA2/{Au,Tp,CASIA 2 Groundtruth}``;
            defaults to ``data_dir() / "casia2"``.
        n: at most this many positives (seeded random subset).
        negatives: tampered images paired with an authentic image from an
            unrelated category (neither its donor nor its host).
    """
    base = Path(root) if root else data_dir() / "casia2"
    casia = base / "CASIA2" if (base / "CASIA2").exists() else base
    au_dir, tp_dir, gt_dir = casia / "Au", casia / "Tp", casia / "CASIA 2 Groundtruth"
    if not tp_dir.exists():
        raise FileNotFoundError(
            f"CASIA v2 not found under {base}. Run: python -m spade.bench.download casia2"
        )

    rng = np.random.RandomState(seed)
    if copy_move:
        return _casia2_copy_moves(tp_dir, gt_dir, au_dir, rng, n, negatives)
    candidates = sorted(p for p in tp_dir.iterdir() if p.name.startswith("Tp_D_"))
    rng.shuffle(candidates)

    samples: List[SpliceSample] = []
    for tp_path in candidates:
        if n is not None and len(samples) >= n:
            break
        parts = tp_path.stem.split("_")
        if len(parts) < 8:
            continue
        id_a, id_b = parts[5], parts[6]
        gt_path = gt_dir / f"{tp_path.stem}_gt.png"
        path_a, path_b = _casia_au_path(au_dir, id_a), _casia_au_path(au_dir, id_b)
        if not (gt_path.exists() and path_a and path_b):
            continue
        tampered = _read_rgb(tp_path)
        mask = np.asarray(Image.open(gt_path).convert("L")) > 127
        if mask.shape != tampered.shape[:2] or not mask.any() or mask.all():
            continue
        image_a, image_b = _read_rgb(path_a), _read_rgb(path_b)
        split = _split_donor_host(tampered, mask, image_a, image_b)
        if split is None:
            continue
        donor, _ = split
        donor_path = path_a if donor is image_a else path_b
        samples.append(SpliceSample(
            donor=donor,
            tampered=tampered,
            mask=mask,
            params={"source": "casia2", "tampered": tp_path.name, "ids": (id_a, id_b),
                    "tampered_path": str(tp_path), "donor_path": str(donor_path)},
        ))

    if negatives and samples:
        au_paths = sorted(p for p in au_dir.iterdir() if p.name.startswith("Au_"))
        positives = list(samples)
        for i in range(negatives):
            pos = positives[i % len(positives)]
            categories = {pos.params["ids"][0][:3], pos.params["ids"][1][:3]}
            while True:
                other = au_paths[rng.randint(len(au_paths))]
                if other.name.split("_")[1] not in categories:
                    break
            samples.append(SpliceSample(
                donor=_read_rgb(other),
                tampered=pos.tampered,
                mask=np.zeros_like(pos.mask),
                params={"source": "casia2", "tampered": pos.params["tampered"], "donor": other.name, "negative": True,
                        "tampered_path": pos.params["tampered_path"], "donor_path": str(other)},
                label=0,
            ))
    return samples


def _casia2_copy_moves(tp_dir, gt_dir, au_dir, rng, n, negatives) -> List[SpliceSample]:
    candidates = sorted(p for p in tp_dir.iterdir() if p.name.startswith("Tp_S_"))
    rng.shuffle(candidates)
    samples: List[SpliceSample] = []
    for tp_path in candidates:
        if n is not None and len(samples) >= n:
            break
        gt_path = gt_dir / f"{tp_path.stem}_gt.png"
        if not gt_path.exists():
            continue
        tampered = _read_rgb(tp_path)
        mask = np.asarray(Image.open(gt_path).convert("L")) > 127
        if mask.shape != tampered.shape[:2] or not mask.any() or mask.all():
            continue
        samples.append(SpliceSample(donor=None, tampered=tampered, mask=mask,
                                    params={"source": "casia2", "tampered": tp_path.name, "copy_move": True,
                                            "tampered_path": str(tp_path)}))
    au_paths = sorted(p for p in au_dir.iterdir() if p.name.startswith("Au_"))
    for i in rng.choice(len(au_paths), min(negatives, len(au_paths)), replace=False):
        image = _read_rgb(au_paths[i])
        samples.append(SpliceSample(donor=None, tampered=image, mask=np.zeros(image.shape[:2], bool),
                                    params={"source": "casia2", "tampered": au_paths[i].name, "negative": True,
                                            "tampered_path": str(au_paths[i])},
                                    label=0))
    return samples


# ----------------------------------------------------------------------
# CSV manifest
# ----------------------------------------------------------------------

def manifest(path: str) -> List[SpliceSample]:
    """Samples from a CSV with columns donor,tampered,mask,label (paths relative to the CSV).

    ``mask`` may be empty for negatives; ``label`` defaults to 1 when a mask is given.
    """
    base = Path(path).parent
    samples = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tampered = _read_rgb(base / row["tampered"])
            mask_path = (row.get("mask") or "").strip()
            if mask_path:
                mask = np.asarray(Image.open(base / mask_path).convert("L")) > 127
            else:
                mask = np.zeros(tampered.shape[:2], dtype=bool)
            label = int(row["label"]) if (row.get("label") or "").strip() else int(mask.any())
            samples.append(SpliceSample(
                donor=_read_rgb(base / row["donor"]),
                tampered=tampered,
                mask=mask,
                params={"source": "manifest", "tampered": row["tampered"]},
                label=label,
            ))
    return samples


# ----------------------------------------------------------------------

def synthetic(n: int, negatives: int = 0, seed: int = 0, copy_move: bool = False, **kwargs) -> List[SpliceSample]:
    """The procedural generator, in the same shape as the other sources."""
    image_size = kwargs.get("image_size", 96)
    if copy_move:
        kwargs.pop("tone_curve_strength", None)
        pos = [make_copy_move(seed=seed + i, **kwargs) for i in range(n)]
        neg = []
        for i in range(negatives):
            clean = make_negative_pair(seed=10_000 + seed + i, image_size=image_size)
            clean.donor = None
            neg.append(clean)
        return pos + neg
    pos = [make_recolored_splice(seed=seed + i, **kwargs) for i in range(n)]
    neg = [make_negative_pair(seed=10_000 + seed + i, image_size=image_size) for i in range(negatives)]
    return pos + neg


def photos(copy_move: bool = False, **kwargs) -> List[SpliceSample]:
    return photo_copy_moves(**kwargs) if copy_move else photo_splices(**kwargs)


SOURCES = {"synthetic": synthetic, "photos": photos, "casia2": casia2, "manifest": manifest}


def load(name: str, **kwargs) -> List[SpliceSample]:
    if name not in SOURCES:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(SOURCES)}")
    return SOURCES[name](**kwargs)


def split_labels(samples: List[SpliceSample]) -> Tuple[List[SpliceSample], List[SpliceSample]]:
    """(positives, negatives)."""
    return [s for s in samples if s.label == 1], [s for s in samples if s.label == 0]

