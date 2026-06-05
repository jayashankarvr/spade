#!/usr/bin/env python3
"""Train multi-scale diffusion-map manifolds + GMM natural-image priors.

One-time offline training for the SPADE Option-A Bayes-factor pipeline. For
each patch size in {3, 4, 5, 6} the script learns:

  1. A diffusion-map embedding of the natural-image patch manifold
     (after photometric orbit projection). Uses sklearn.SpectralEmbedding
     with sparse k-NN affinity, trained on a ~50k-patch anchor sample.

  2. A Gaussian Mixture Model density on that embedding, used as the
     null-hypothesis prior P_0(patch) in the Bayes-factor forensic scorer.
     Trained on a much larger sample (default 2M) Nystrom-extended into
     the anchor embedding.

The output for each size is a single self-contained joblib file consumable
by spade.priors.GMMDiffusionPrior.

Usage:
    python scripts/train_priors.py --corpus path/to/images/ --out priors/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

# Make `spade.priors` importable when running this script from the repo root.
import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from spade.priors.normalize import photometric_normalize  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PATCH_SIZES: Tuple[int, ...] = (3, 4, 5, 6)
EMBED_DIM = {3: 8, 4: 12, 5: 16, 6: 20}
GMM_COMPONENTS = {3: 256, 4: 512, 5: 1024, 6: 2048}
EIG_SUBSAMPLE = 50_000           # patches in the spectral eigendecomposition
PRIOR_TRAIN_TARGET = 2_000_000   # patches Nystrom-extended for GMM fitting
PATCHES_PER_IMAGE = 100
KNN_AFFINITY = 25
RANDOM_SEED = 42


@dataclass
class TrainedPrior:
    patch_size: int
    embed_dim: int
    anchor_features: np.ndarray
    anchor_embedding: np.ndarray
    gmm: object
    version: int = 1


# ---------------------------------------------------------------------------
# Corpus iteration and patch sampling
# ---------------------------------------------------------------------------

def iter_image_paths(corpus: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for path in corpus.rglob("*"):
        if path.suffix.lower() in exts:
            yield path


def load_image(path: Path) -> np.ndarray | None:
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


def reservoir_sample_patches(
    image_paths: List[Path],
    size: int,
    n_target: int,
    rng: np.random.Generator,
    patches_per_image: int = PATCHES_PER_IMAGE,
) -> np.ndarray:
    """Reservoir-sample n_target random patches across the corpus.

    Uses Algorithm R so that we can stream very large corpora without
    keeping everything in memory.
    """
    reservoir = np.empty((n_target, size, size, 3), dtype=np.float32)
    seen = 0

    # Shuffle path order so reservoir mixes well even on early-stop
    paths = list(image_paths)
    rng.shuffle(paths)

    for path in paths:
        img = load_image(path)
        if img is None:
            continue
        h, w, _ = img.shape
        if h < size or w < size:
            continue

        n_per = min(patches_per_image, (h - size + 1) * (w - size + 1))
        ys = rng.integers(0, h - size + 1, n_per)
        xs = rng.integers(0, w - size + 1, n_per)

        for x, y in zip(xs, ys):
            patch = img[y:y + size, x:x + size, :].astype(np.float32) / 255.0
            if seen < n_target:
                reservoir[seen] = patch
            else:
                idx = int(rng.integers(0, seen + 1))
                if idx < n_target:
                    reservoir[idx] = patch
            seen += 1

        # Stop once the reservoir is well-mixed (saw >> n_target items)
        if seen >= n_target * 4:
            return reservoir

    if seen < n_target:
        reservoir = reservoir[:seen]
    return reservoir


# ---------------------------------------------------------------------------
# Diffusion map + Nystrom extension
# ---------------------------------------------------------------------------

def fit_diffusion_map(features: np.ndarray, embed_dim: int):
    """Sparse-affinity spectral embedding (= diffusion map at t=1)."""
    from sklearn.manifold import SpectralEmbedding

    model = SpectralEmbedding(
        n_components=embed_dim,
        affinity="nearest_neighbors",
        n_neighbors=KNN_AFFINITY,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    embedding = model.fit_transform(features).astype(np.float32)
    return model, embedding


def nystrom_extend(
    query_features: np.ndarray,
    anchor_features: np.ndarray,
    anchor_embedding: np.ndarray,
    k: int = KNN_AFFINITY,
) -> np.ndarray:
    """Heat-kernel-weighted out-of-sample extension."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(anchor_features)
    distances, indices = nn.kneighbors(query_features)
    sigma = float(np.median(distances)) + 1e-8
    weights = np.exp(-(distances ** 2) / (2.0 * sigma * sigma))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("nk,nke->ne", weights, anchor_embedding[indices]).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# GMM
# ---------------------------------------------------------------------------

def fit_gmm(embedding: np.ndarray, n_components: int):
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        max_iter=200,
        random_state=RANDOM_SEED,
        reg_covar=1e-4,
        verbose=1,
    )
    gmm.fit(embedding)
    return gmm


# ---------------------------------------------------------------------------
# Per-size pipeline
# ---------------------------------------------------------------------------

def train_for_size(
    image_paths: List[Path],
    size: int,
    out_dir: Path,
    rng: np.random.Generator,
) -> Path:
    print(f"\n=== Training prior for {size}x{size} patches ===")

    print(f"  Anchor sampling ({EIG_SUBSAMPLE} patches)...")
    anchors = reservoir_sample_patches(image_paths, size, EIG_SUBSAMPLE, rng)
    print(f"    actual: {len(anchors)} patches")
    anchor_features = photometric_normalize(anchors)

    print(f"  Fitting diffusion map (embed_dim={EMBED_DIM[size]})...")
    _spectral, anchor_embedding = fit_diffusion_map(
        anchor_features, EMBED_DIM[size]
    )

    print(f"  GMM-train sampling ({PRIOR_TRAIN_TARGET} patches)...")
    train_patches = reservoir_sample_patches(
        image_paths, size, PRIOR_TRAIN_TARGET, rng
    )
    print(f"    actual: {len(train_patches)} patches")
    train_features = photometric_normalize(train_patches)

    print("  Nystrom-extending into embedding...")
    train_embedding = nystrom_extend(
        train_features, anchor_features, anchor_embedding
    )

    print(f"  Fitting GMM ({GMM_COMPONENTS[size]} components)...")
    gmm = fit_gmm(train_embedding, GMM_COMPONENTS[size])

    out_path = out_dir / f"prior_{size}x{size}.joblib"
    import joblib
    joblib.dump(
        {
            "patch_size": size,
            "embed_dim": EMBED_DIM[size],
            "anchor_features": anchor_features,
            "anchor_embedding": anchor_embedding,
            "gmm": gmm,
            "version": 1,
        },
        out_path,
        compress=3,
    )
    print(f"  Saved {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, required=True,
        help="Directory of natural images (searched recursively)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("priors"),
        help="Output directory for trained priors",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=list(PATCH_SIZES),
        choices=list(PATCH_SIZES),
        help="Patch sizes to train (default: all of 3 4 5 6)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-mode: tiny subsample sizes for fast end-to-end testing",
    )
    args = parser.parse_args()

    if args.smoke:
        global EIG_SUBSAMPLE, PRIOR_TRAIN_TARGET, GMM_COMPONENTS
        EIG_SUBSAMPLE = 2_000
        PRIOR_TRAIN_TARGET = 20_000
        GMM_COMPONENTS = {3: 32, 4: 48, 5: 64, 6: 80}
        print("[smoke mode] reduced subsample sizes")

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Scanning corpus at {args.corpus}...")
    image_paths = list(iter_image_paths(args.corpus))
    print(f"Found {len(image_paths)} candidate images.")
    if not image_paths:
        raise SystemExit("No images found - aborting.")

    for size in args.sizes:
        train_for_size(image_paths, size, args.out, rng)

    print("\nAll requested priors trained.")


if __name__ == "__main__":
    main()
