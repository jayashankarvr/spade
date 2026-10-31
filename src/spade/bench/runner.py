"""Benchmark runner: load a dataset, run detectors, report numbers.

Runs SPADE against the baselines to see where it actually stands. Any change
to matching defaults should be checked here.

Usage:
    python -m spade.bench.runner --n 40 --grade 0.15
    python -m spade.bench.runner --dataset photos --n 60 --negatives 60 --calibrate
    python -m spade.bench.runner --dataset casia2 --n 300 --negatives 300 --calibrate --json out.json
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Sequence

from spade.bench import calibration
from spade.bench.baselines import Detector, RootSiftRansacDetector
from spade.bench.datasets import load, split_labels
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
    """Synthetic positives and negatives (kept for existing callers)."""
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
    """Synthetic benchmark across detectors (SPADE + RootSIFT by default)."""
    from spade.bench.baselines import SpadeDetector

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


def _dataset_kwargs(args) -> Dict:
    if args.dataset == "synthetic":
        return dict(n=args.n, negatives=args.negatives, image_size=args.image_size,
                    fragment_size=args.fragment_size, color_grade_strength=args.grade,
                    jpeg_quality=args.jpeg, seed=args.seed)
    if args.dataset == "photos":
        return dict(n=args.n, negatives=args.negatives, color_grade_strength=args.grade,
                    jpeg_quality=args.jpeg, seed=args.seed)
    if args.dataset == "casia2":
        return dict(root=args.root, n=args.n, negatives=args.negatives, seed=args.seed)
    return dict(path=args.root)


def main(argv=None):
    p = argparse.ArgumentParser(description="SPADE benchmark runner")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "photos", "casia2", "manifest"])
    p.add_argument("--root", default=None, help="casia2 folder or manifest CSV path")
    p.add_argument("--n", type=int, default=40, help="number of positive (spliced) samples")
    p.add_argument("--negatives", type=int, default=20, help="number of negative (unrelated) pairs")
    p.add_argument("--image-size", type=int, default=96, help="synthetic only")
    p.add_argument("--fragment-size", type=int, default=24, help="synthetic only")
    p.add_argument("--grade", type=float, default=0.15, help="color-grade strength (generated datasets)")
    p.add_argument("--jpeg", type=int, default=None, help="JPEG quality for the tampered image (generated datasets)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calibrate", action="store_true", help="also report false regions per pair vs NFA threshold")
    p.add_argument("--no-baselines", action="store_true", help="skip the RootSIFT baseline")
    p.add_argument("--json", default=None, help="write all numbers to this file")
    args = p.parse_args(argv)

    samples = load(args.dataset, **_dataset_kwargs(args))
    positives, negatives = split_labels(samples)

    records = calibration.score_samples(samples)
    results = {"spade": calibration.summary(records)}
    if not args.no_baselines:
        results["rootsift+ransac"] = evaluate(RootSiftRansacDetector(), positives, negatives)

    print(f"\nSPADE benchmark | {args.dataset} | {len(positives)} pos + {len(negatives)} neg | "
          f"grade={args.grade} jpeg={args.jpeg}\n")
    print(_format_table(results))

    report = {"dataset": args.dataset, "args": vars(args), "results": results}
    if args.calibrate:
        rows = calibration.calibration_table(records)
        print("\nNFA calibration (false regions per pair should stay <= epsilon):\n")
        print(calibration.format_table(rows))
        report["calibration"] = rows

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"\nWrote {args.json}")
    return report


if __name__ == "__main__":
    main()
