#!/usr/bin/env python3
"""Forensic benchmark harness for SPADE.

Generates copy-move forgeries from a corpus of source images, runs both the
legacy chi-square pipeline and the new Bayes-factor pipeline, and reports
detection/localization metrics for side-by-side comparison.

For deployment-grade evaluation point this at CASIA v2.0 / MICC-F220 /
CoMoFoD instead of the synthetic corpus. The harness only requires images
and synthesizes its own forgeries on the fly so any directory of natural
images works.

Metrics reported per pipeline:
  detect@image     fraction of forgeries where best_match's image_id is correct
  localize@px      median pixel distance from best match to true copy location
  bf_separation    log-Bayes-factor margin between true and best-incorrect peak
                   (Bayes pipeline only)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# Ensure src/ is on sys.path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from spade.engine import Config, ForensicsEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Forgery synthesis
# ---------------------------------------------------------------------------

@dataclass
class Forgery:
    target_image: np.ndarray
    source_fragment: np.ndarray
    src_xy: Tuple[int, int]            # original location of the copied region
    paste_xy: Tuple[int, int]          # paste location of the copy
    region_size: int


def make_forgery(
    image: np.ndarray,
    rng: np.random.Generator,
    region_size: int = 32,
    add_noise: float = 0.0,
    jpeg_quality: Optional[int] = None,
) -> Forgery:
    """Copy a random region within an image to a non-overlapping location."""
    h, w = image.shape[:2]
    # Pick source region
    sx = int(rng.integers(0, w - region_size))
    sy = int(rng.integers(0, h - region_size))
    # Pick paste location with non-overlap
    for _ in range(50):
        px = int(rng.integers(0, w - region_size))
        py = int(rng.integers(0, h - region_size))
        if abs(px - sx) >= region_size or abs(py - sy) >= region_size:
            break

    fragment = image[sy:sy + region_size, sx:sx + region_size].copy()
    target = image.copy()
    target[py:py + region_size, px:px + region_size] = fragment

    if add_noise > 0:
        noise = rng.normal(0, add_noise * 255, target.shape)
        target = np.clip(target.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if jpeg_quality is not None:
        from io import BytesIO
        buf = BytesIO()
        Image.fromarray(target).save(buf, format="JPEG", quality=jpeg_quality)
        buf.seek(0)
        target = np.asarray(Image.open(buf).convert("RGB"))

    return Forgery(
        target_image=target,
        source_fragment=fragment,
        src_xy=(sx, sy),
        paste_xy=(px, py),
        region_size=region_size,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    detected: bool
    localization_error_px: float
    elapsed_ms: float
    best_log_bf: Optional[float] = None


@dataclass
class PipelineMetrics:
    name: str
    trials: List[TrialResult] = field(default_factory=list)

    def report(self) -> str:
        if not self.trials:
            return f"{self.name}: no trials"
        det_rate = np.mean([t.detected for t in self.trials])
        loc_med = np.median([t.localization_error_px for t in self.trials])
        loc_p90 = np.quantile([t.localization_error_px for t in self.trials], 0.9)
        elapsed = np.mean([t.elapsed_ms for t in self.trials])
        bf_line = ""
        if any(t.best_log_bf is not None for t in self.trials):
            bfs = [t.best_log_bf for t in self.trials if t.best_log_bf is not None]
            bf_line = f"  best_log_bf:    median={np.median(bfs):.2f}\n"
        return (
            f"{self.name}\n"
            f"  detect@image:   {det_rate:.2%}  ({sum(t.detected for t in self.trials)}/{len(self.trials)})\n"
            f"  localize@px:    median={loc_med:.1f}  p90={loc_p90:.1f}\n"
            f"{bf_line}"
            f"  latency:        mean={elapsed:.0f} ms"
        )


def evaluate_pipeline(
    name: str,
    config: Config,
    forgeries: List[Forgery],
    target_id_template: str = "img_{:03d}",
) -> PipelineMetrics:
    metrics = PipelineMetrics(name=name)
    for i, f in enumerate(forgeries):
        engine = ForensicsEngine(config)
        target_id = target_id_template.format(i)
        engine.index_target(f.target_image, target_id)

        t0 = time.perf_counter()
        result = engine.match(f.source_fragment, return_heatmap=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        detected = (
            result.best_match is not None
            and result.best_match.image_id == target_id
        )
        if detected:
            tx, ty = result.best_match.target_coord
            # Distance to the true paste location (not the original!)
            err = float(np.hypot(tx - f.paste_xy[0], ty - f.paste_xy[1]))
            # Also accept matches near the original source location
            err_alt = float(np.hypot(tx - f.src_xy[0], ty - f.src_xy[1]))
            err = min(err, err_alt)
        else:
            err = float("inf")

        bf = None
        if "log_z" in (result.stats or {}):
            bf = float(result.stats["log_z"])

        metrics.trials.append(TrialResult(
            detected=detected,
            localization_error_px=err,
            elapsed_ms=elapsed_ms,
            best_log_bf=bf,
        ))
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_corpus(corpus: Path, n: int, rng: np.random.Generator) -> List[np.ndarray]:
    paths = sorted(corpus.glob("*.png")) + sorted(corpus.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"No images in {corpus}")
    if len(paths) > n:
        idxs = rng.choice(len(paths), n, replace=False)
        paths = [paths[int(i)] for i in idxs]
    return [np.asarray(Image.open(p).convert("RGB")) for p in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Directory of source images for forgery synthesis")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of forgeries to evaluate")
    parser.add_argument("--region", type=int, default=32)
    parser.add_argument("--noise", type=float, default=0.02,
                        help="Gaussian noise stddev added to forged image (in [0,1])")
    parser.add_argument("--jpeg", type=int, default=None,
                        help="JPEG quality to apply post-forgery (None to disable)")
    parser.add_argument("--priors-dir", type=Path, default=None,
                        help="Optional directory of trained priors for the bayes pipeline")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    images = load_corpus(args.corpus, args.n, rng)
    forgeries: List[Forgery] = []
    for img in images:
        forgeries.append(make_forgery(
            img, rng,
            region_size=args.region,
            add_noise=args.noise,
            jpeg_quality=args.jpeg,
        ))

    print(f"\nForgeries: {len(forgeries)} (region={args.region}px, noise={args.noise}, jpeg={args.jpeg})\n")

    # Pipeline 1: legacy chi-square
    chi2_config = Config(
        scoring_mode="chi2",
        patch_size=3,
        stride=2,
        coherence_enabled=True,
    )
    print("Running chi-square pipeline...")
    m_chi2 = evaluate_pipeline("chi2 (legacy)", chi2_config, forgeries)

    # Pipeline 2: Bayes-factor
    bayes_config = Config(
        scoring_mode="bayes",
        stride=2,
        priors_dir=str(args.priors_dir) if args.priors_dir else None,
    )
    print("Running Bayes-factor pipeline...")
    m_bayes = evaluate_pipeline("bayes (option-A)", bayes_config, forgeries)

    print("\n" + "=" * 50)
    print(m_chi2.report())
    print()
    print(m_bayes.report())
    print("=" * 50)


if __name__ == "__main__":
    main()
