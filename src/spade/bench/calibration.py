"""Does NFA < epsilon really mean at most ~epsilon false regions per search?

The a-contrario promise is about false alarms, so we count them directly:
run SPADE on every sample once, keep every region with its NFA, then for a
range of thresholds report

* false regions per image pair - regions on null pairs, plus regions on
  spliced pairs that miss the splice. Calibrated means this stays <= epsilon.
* recall - spliced pairs with a region that lands on the splice.

One pass over the data serves every threshold, and the same records give the
usual localization metrics at the engine's own ``max_nfa``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from spade.aggregation.localize import localize_region
from spade.bench.metrics import aggregate, detection_auc, localization_scores

EPSILONS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
# A region counts as hitting the splice when at least half its footprint is inside the mask.
HIT_OVERLAP = 0.5


@dataclass
class RegionRecord:
    log10_nfa: float
    on_splice: bool          # footprint mostly inside the ground-truth mask


@dataclass
class SampleRecord:
    label: int
    regions: List[RegionRecord] = field(default_factory=list)
    mask: np.ndarray = None          # predicted localization at the engine's max_nfa
    gt_mask: np.ndarray = None
    seconds: float = 0.0

    @property
    def score(self) -> float:
        """-log10 NFA of the best region; the image-level detection score."""
        return -min((r.log10_nfa for r in self.regions), default=1e9)


def score_samples(samples: Sequence, config=None, progress: bool = False) -> List[SampleRecord]:
    """Run SPADE once per sample (index donor, query tampered) and keep every region.

    ``progress`` prints a line to stderr every 10 samples (long real-data runs).
    """
    import sys

    from spade.engine import Config, ForensicsEngine

    config = config or Config()
    records = []
    t0 = time.perf_counter()
    for i, sample in enumerate(samples, start=1):
        start = time.perf_counter()
        engine = ForensicsEngine(config)
        source_sigma, donor_sigma = _jpeg_sigmas(sample)
        if sample.donor is None:  # copy-move: the image is its own source
            result = engine.match_self(sample.tampered, jpeg_sigma=source_sigma)
        else:
            engine.index_target(sample.donor, "donor", jpeg_sigma=donor_sigma)
            result = engine.match(sample.tampered, return_heatmap=False, jpeg_sigma=source_sigma)
        shape = sample.tampered.shape[:2]
        regions = []
        marks_copy_only = (sample.params or {}).get("mask_marks") == "copy"
        for region in result.coherent_regions:
            footprint = localize_region(region.matches, shape, largest_component=False).mask
            if sample.donor is None and marks_copy_only:
                # The mask marks the pasted copy but not its original: a region
                # is right if either of its two halves lands on the copy.
                halves = (footprint, _target_footprint(region, shape))
                hit = any((h & sample.mask).sum() / max(h.sum(), 1) >= HIT_OVERLAP for h in halves)
                regions.append(RegionRecord(float(region.log10_nfa), bool(hit)))
                continue
            if sample.donor is None:
                footprint |= _target_footprint(region, shape)
            inside = (footprint & sample.mask).sum() / max(footprint.sum(), 1)
            regions.append(RegionRecord(float(region.log10_nfa), bool(inside >= HIT_OVERLAP)))
        records.append(SampleRecord(
            label=sample.label,
            regions=regions,
            mask=result.localization.mask,
            gt_mask=sample.mask,
            seconds=time.perf_counter() - start,
        ))
        if progress and (i % 10 == 0 or i == len(samples)):
            elapsed = time.perf_counter() - t0
            print(f"  {i}/{len(samples)} samples done, {elapsed:.0f}s, ~{elapsed / i * (len(samples) - i):.0f}s left",
                  file=sys.stderr, flush=True)
    return records


def _jpeg_sigmas(sample):
    """JPEG rounding error of (tampered, donor), from the files or the generator's quality."""
    from spade.jpeg import quantization_sigma, quantization_sigma_for_quality

    params = sample.params or {}
    if "tampered_path" in params:
        source = quantization_sigma(params["tampered_path"]) or 0.0
        donor = (quantization_sigma(params["donor_path"]) or 0.0) if "donor_path" in params else 0.0
        return source, donor
    quality = params.get("jpeg_quality")
    return (quantization_sigma_for_quality(quality) if quality else 0.0), 0.0


def _target_footprint(region, shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for m in region.matches:
        x, y = m.target_coord
        mask[y:y + m.patch_size, x:x + m.patch_size] = True
    return mask


def calibration_table(records: Sequence[SampleRecord], epsilons=EPSILONS) -> List[Dict[str, float]]:
    """One row per threshold: false regions per pair (should be <= eps) and recall."""
    positives = [r for r in records if r.label == 1]
    rows = []
    for eps in epsilons:
        cut = np.log10(eps)
        false_regions = sum(
            1 for rec in records for reg in rec.regions
            if reg.log10_nfa < cut and not (rec.label == 1 and reg.on_splice)
        )
        null_pairs_hit = sum(
            1 for rec in records if rec.label == 0 and any(reg.log10_nfa < cut for reg in rec.regions)
        )
        n_null = sum(1 for rec in records if rec.label == 0)
        found = sum(1 for rec in positives if any(reg.log10_nfa < cut and reg.on_splice for reg in rec.regions))
        rows.append({
            "epsilon": eps,
            "false_regions_per_pair": false_regions / max(len(records), 1),
            "null_pairs_with_alarm": null_pairs_hit / n_null if n_null else float("nan"),
            "recall": found / len(positives) if positives else float("nan"),
        })
    return rows


def summary(records: Sequence[SampleRecord]) -> Dict[str, float]:
    """Localization metrics (positives) and detection AUC (all), like ``runner.evaluate``."""
    positives = [r for r in records if r.label == 1]
    out = aggregate([localization_scores(r.mask, r.gt_mask) for r in positives])
    has_negatives = any(r.label == 0 for r in records)
    out["det_auc"] = detection_auc([r.score for r in records], [r.label for r in records]) if has_negatives else float("nan")
    out["sec_per_image"] = float(np.mean([r.seconds for r in records])) if records else float("nan")
    return out


def format_table(rows: Sequence[Dict[str, float]]) -> str:
    lines = ["  epsilon   false regions/pair   null pairs w/ alarm   recall",
             "  " + "-" * 58]
    for row in rows:
        ok = "ok" if row["false_regions_per_pair"] <= row["epsilon"] else "OVER"
        lines.append(
            f"  {row['epsilon']:<8g}  {row['false_regions_per_pair']:>10.4f} {ok:>5}   "
            f"{row['null_pairs_with_alarm']:>14.3f}      {row['recall']:>6.3f}"
        )
    return "\n".join(lines)
