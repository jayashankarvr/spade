"""Benchmark runner: generate samples, run detectors, report numbers.

This is the decision gate (ROADMAP section 8): run SPADE against the baselines
on synthetic recolored splices and see where it actually stands.

Usage:
    python -m spade.bench.runner --n 40 --grade 0.15
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Sequence

import numpy as np

from spade.bench.baselines import Detector, RootSiftRansacDetector, SpadeDetector
from spade.bench.metrics import aggregate, detection_auc, localization_scores
from spade.bench.synthetic import make_negative_pair, make_recolored_splice


def build_samples(
    n: int,
    image_size: int,
    fragment_size: int,
    color_grade_strength: float,
    jpeg_quality=None,
    negatives: int = 0,
):
    """Build a labelled sample set: `n` positives (with splice) + `negatives`."""
    positives = [
        make_recolored_splice(
            seed=i,
            image_size=image_size,
            fragment_size=fragment_size,
            color_grade_strength=color_grade_strength,
            jpeg_quality=jpeg_quality,
        )
        for i in range(n)
    ]
    negs = [make_negative_pair(seed=10_000 + i, image_size=image_size) for i in range(negatives)]
    return positives, negs


def evaluate(detector: Detector, positives: Sequence, negatives: Sequence) -> Dict:
    """Run one detector over positives (localization) + all samples (detection)."""
    loc_scores: List[Dict[str, float]] = []
    det_scores: List[float] = []
    det_labels: List[int] = []

    t0 = time.perf_counter()
    for s in positives:
        mask, score = detector.detect(s.donor, s.tampered)
        loc_scores.append(localization_scores(mask, s.mask))
        det_scores.append(score)
        det_labels.append(1)
    for s in negatives:
        _, score = detector.detect(s.donor, s.tampered)
        det_scores.append(score)
        det_labels.append(0)
    elapsed = time.perf_counter() - t0

    n_imgs = len(positives) + len(negatives)
    out = aggregate(loc_scores)
    out["det_auc"] = detection_auc(det_scores, det_labels) if negatives else float("nan")
    out["sec_per_image"] = elapsed / max(n_imgs, 1)
    return out


def run(
    n: int = 40,
    image_size: int = 96,
    fragment_size: int = 24,
    color_grade_strength: float = 0.15,
    jpeg_quality=None,
    negatives: int = 0,
    detectors: Sequence[Detector] = None,
) -> Dict[str, Dict]:
    if detectors is None:
        detectors = [SpadeDetector(), RootSiftRansacDetector()]
    positives, negs = build_samples(
        n, image_size, fragment_size, color_grade_strength, jpeg_quality, negatives
    )
    return {d.name: evaluate(d, positives, negs) for d in detectors}


def _format_table(results: Dict[str, Dict]) -> str:
    cols = ["iou", "f1", "precision", "recall", "mcc", "det_auc", "sec_per_image"]
    width = max(len(n) for n in results) + 2
    header = "detector".ljust(width) + "".join(c.rjust(13) for c in cols)
    lines = [header, "-" * len(header)]
    for name, r in results.items():
        row = name.ljust(width) + "".join(f"{r.get(c, float('nan')):13.4f}" for c in cols)
        lines.append(row)
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="SPADE benchmark runner")
    p.add_argument("--n", type=int, default=40, help="number of positive (spliced) samples")
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--fragment-size", type=int, default=24)
    p.add_argument("--grade", type=float, default=0.15, help="color-grade strength (0 = none)")
    p.add_argument("--jpeg", type=int, default=None, help="JPEG quality applied to tampered image")
    p.add_argument("--negatives", type=int, default=20, help="number of negative (no-splice) samples")
    args = p.parse_args(argv)

    results = run(
        n=args.n,
        image_size=args.image_size,
        fragment_size=args.fragment_size,
        color_grade_strength=args.grade,
        jpeg_quality=args.jpeg,
        negatives=args.negatives,
    )
    print(
        f"\nSPADE benchmark | n={args.n} pos + {args.negatives} neg | "
        f"size={args.image_size} frag={args.fragment_size} grade={args.grade} jpeg={args.jpeg}\n"
    )
    print(_format_table(results))
    return results


if __name__ == "__main__":
    main()
