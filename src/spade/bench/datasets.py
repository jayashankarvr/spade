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
    _jpeg_roundtrip,
    _random_color_transform,
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


def photo_splices(
    n: int,
    negatives: int = 0,
    crop_size: int = 192,
    fragment_size: int = 40,
    color_grade_strength: float = 0.15,
    jpeg_quality: Optional[int] = 90,
    seed: int = 0,
) -> List[SpliceSample]:
    """``n`` recolored splices between different photos, plus ``negatives`` unrelated crops."""
    photos = load_photos()
    names = sorted(photos)
    rng = np.random.RandomState(seed)
    samples = []

    for i in range(n):
        donor_name, host_name = rng.choice(names, 2, replace=False)
        donor = _random_crop(rng, photos[donor_name], crop_size)
        host = _random_crop(rng, photos[host_name], crop_size)

        fx, fy = _textured_spot(rng, donor, fragment_size)
        region = donor[fy:fy + fragment_size, fx:fx + fragment_size].astype(np.float32) / 255.0
        M, b = _random_color_transform(rng, color_grade_strength)
        recolored = (_apply_color_transform(region, M, b) * 255).astype(np.uint8)

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
                    "donor_region": (fx, fy, fragment_size, fragment_size), "jpeg_quality": jpeg_quality},
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
) -> List[SpliceSample]:
    """CASIA v2 cross-image splices as (donor, tampered, mask) pairs.

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
        split = _split_donor_host(tampered, mask, _read_rgb(path_a), _read_rgb(path_b))
        if split is None:
            continue
        donor, _ = split
        samples.append(SpliceSample(
            donor=donor,
            tampered=tampered,
            mask=mask,
            params={"source": "casia2", "tampered": tp_path.name, "ids": (id_a, id_b)},
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
                params={"source": "casia2", "tampered": pos.params["tampered"], "donor": other.name, "negative": True},
                label=0,
            ))
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

def synthetic(n: int, negatives: int = 0, seed: int = 0, **kwargs) -> List[SpliceSample]:
    """The procedural generator, in the same shape as the other sources."""
    pos = [make_recolored_splice(seed=seed + i, **kwargs) for i in range(n)]
    image_size = kwargs.get("image_size", 96)
    neg = [make_negative_pair(seed=10_000 + seed + i, image_size=image_size) for i in range(negatives)]
    return pos + neg


SOURCES = {"synthetic": synthetic, "photos": photo_splices, "casia2": casia2, "manifest": manifest}


def load(name: str, **kwargs) -> List[SpliceSample]:
    if name not in SOURCES:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(SOURCES)}")
    return SOURCES[name](**kwargs)


def split_labels(samples: List[SpliceSample]) -> Tuple[List[SpliceSample], List[SpliceSample]]:
    """(positives, negatives)."""
    return [s for s in samples if s.label == 1], [s for s in samples if s.label == 0]

