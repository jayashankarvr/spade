"""Bayes-factor forensic engine - the Option-A pipeline end-to-end.

Indexes targets at all four patch sizes simultaneously, scores candidate
matches with calibrated log Bayes factors against a natural-image prior,
aggregates evidence in a 4-D Hough accumulator with persistent-homology
peak detection, then runs multi-grid loopy BP over a hierarchical Potts
factor graph to produce per-pixel posteriors and a global log Z forensic
free-energy score.

Relationship to engine.ForensicsEngine
--------------------------------------
This is a parallel implementation, not a subclass. ForensicsEngine.match
delegates here when Config.scoring_mode == "bayes". Same MatchResult-like
return type so downstream consumers (CLI, API, heatmap viewer) work
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from spade.aggregation.belief_propagation import (
    BPGraph,
    BPNode,
    BPEdge,
    build_multiscale_graph,
    loopy_bp,
)
from spade.aggregation.heatmap import Match, SpatialAggregator
from spade.aggregation.hough import HoughAccumulator4D, HoughCandidate, HoughConfig
from spade.aggregation.peaks import PeakHypothesis, PersistentPeakDetector
from spade.descriptors.bayesian import BayesianDescriptor, BayesianDescriptorConfig
from spade.extraction.patches import PatchExtractor
from spade.priors.natural import GMMDiffusionPrior, NaturalImagePrior
from spade.search.index import ANNIndex
from spade.verification.affine import AffineVerifier
from spade.verification.bayes_factor import BayesFactorScorer
from spade.verification.rg_trajectory import RGConsistencyScorer


PATCH_SIZES_DEFAULT: Tuple[int, ...] = (3, 4, 5, 6)


@dataclass
class BayesianConfig:
    patch_sizes: Tuple[int, ...] = PATCH_SIZES_DEFAULT
    stride: int = 1
    entropy_threshold: Optional[float] = None      # priors handle the trivial-patch suppression
    k_neighbors: int = 32
    noise_sigma: float = 0.02
    distance_threshold: float = 1.5                # cosine-equivalent on L2 in [0,2]
    min_log_bf: float = 0.0                        # only positive-evidence votes count
    max_peaks: int = 32
    persistence_floor: float = 5.0                 # nats
    bp_max_iter: int = 30
    bp_damping: float = 0.4
    spatial_radius: float = 4.0
    spatial_beta: float = 1.0
    inclusion_beta: float = 2.0
    rg_enabled: bool = True
    priors_dir: Optional[Path] = None              # directory of prior_NxN.joblib files


@dataclass
class BayesianMatchResult:
    matches: List[Match] = field(default_factory=list)
    best_match: Optional[Match] = None
    heatmap: Optional[np.ndarray] = None
    peak_hypotheses: List[PeakHypothesis] = field(default_factory=list)
    log_z: float = 0.0
    posteriors: Optional[np.ndarray] = None        # (N_source_patches, n_labels)
    stats: Dict[str, float] = field(default_factory=dict)


class BayesianForensicsEngine:
    """Multi-scale Bayes-factor forensic engine."""

    def __init__(self, cfg: Optional[BayesianConfig] = None):
        self.cfg = cfg or BayesianConfig()
        self._priors: Dict[int, NaturalImagePrior] = {}
        if self.cfg.priors_dir is not None:
            self._load_priors(Path(self.cfg.priors_dir))

        self._extractors = {
            s: PatchExtractor(size=s, stride=self.cfg.stride,
                              entropy_threshold=self.cfg.entropy_threshold)
            for s in self.cfg.patch_sizes
        }
        self._descriptors = {
            s: BayesianDescriptor(
                cfg=BayesianDescriptorConfig(),
                prior=self._priors.get(s),
            )
            for s in self.cfg.patch_sizes
        }
        self._verifier = AffineVerifier(noise_sigma=self.cfg.noise_sigma)
        self._scorers = {
            s: BayesFactorScorer(prior=self._priors.get(s),
                                 verifier=self._verifier,
                                 noise_sigma=self.cfg.noise_sigma)
            for s in self.cfg.patch_sizes
        }
        self._rg = RGConsistencyScorer() if self.cfg.rg_enabled else None
        self._aggregator = SpatialAggregator()

        # Per-size FAISS index + per-size target patch storage
        self._indices: Dict[int, ANNIndex] = {}
        self._target_patches: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in self.cfg.patch_sizes}
        self._target_coords: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in self.cfg.patch_sizes}
        self._target_shapes: Dict[str, Tuple[int, int]] = {}

    def _load_priors(self, priors_dir: Path) -> None:
        for s in self.cfg.patch_sizes:
            path = priors_dir / f"prior_{s}x{s}.joblib"
            if path.exists():
                self._priors[s] = GMMDiffusionPrior(path)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_target(self, image: np.ndarray, image_id: str) -> None:
        image = self._normalize_image(image)
        self._target_shapes[image_id] = (image.shape[0], image.shape[1])
        for s in self.cfg.patch_sizes:
            collection = self._extractors[s].extract(image)
            if len(collection.patches) == 0:
                continue
            descriptors = self._descriptors[s].compute_batch(collection.patches)
            if s not in self._indices:
                self._indices[s] = ANNIndex(dim=descriptors.shape[1])
            metadata_list = [
                {"image_id": image_id, "patch_idx": int(i)}
                for i in range(len(descriptors))
            ]
            self._indices[s].add(descriptors, metadata_list)
            self._target_patches[s][image_id] = collection.patches
            self._target_coords[s][image_id] = collection.coords

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(
        self,
        source_image: np.ndarray,
        return_heatmap: bool = True,
    ) -> BayesianMatchResult:
        source_image = self._normalize_image(source_image)
        per_scale_data = self._extract_and_score(source_image)
        if not per_scale_data:
            return BayesianMatchResult(stats={"source_patches": 0})

        accumulator = self._build_accumulator(per_scale_data)
        peaks = PersistentPeakDetector(
            persistence_floor=self.cfg.persistence_floor,
            max_peaks=self.cfg.max_peaks,
        ).detect(accumulator)

        if not peaks:
            return BayesianMatchResult(stats={
                "source_patches": sum(len(d["coords"]) for d in per_scale_data.values()),
                "candidates": sum(len(d["candidates"]) for d in per_scale_data.values()),
                "peaks": 0,
            })

        bp_result = self._run_bp(per_scale_data, peaks)

        matches = self._matches_from_bp(per_scale_data, peaks, bp_result.marginals)
        best_match = max(matches, key=lambda m: m.probability) if matches else None

        heatmap = None
        if return_heatmap and best_match is not None:
            target_shape = self._target_shapes.get(best_match.image_id)
            if target_shape is not None:
                heatmap = self._aggregator.aggregate(
                    [m for m in matches if m.image_id == best_match.image_id],
                    target_shape,
                )

        return BayesianMatchResult(
            matches=matches,
            best_match=best_match,
            heatmap=heatmap,
            peak_hypotheses=peaks,
            log_z=float(bp_result.log_z),
            posteriors=bp_result.marginals,
            stats={
                "source_patches": sum(len(d["coords"]) for d in per_scale_data.values()),
                "candidates": sum(len(d["candidates"]) for d in per_scale_data.values()),
                "peaks": len(peaks),
                "bp_iters": bp_result.iterations,
                "bp_converged": bp_result.converged,
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return image

    def _extract_and_score(
        self, source_image: np.ndarray,
    ) -> Dict[int, dict]:
        """Per-scale: extract source patches, FAISS-lookup, Bayes-factor score."""
        out: Dict[int, dict] = {}
        for s in self.cfg.patch_sizes:
            if s not in self._indices:
                continue
            collection = self._extractors[s].extract(source_image)
            if len(collection.patches) == 0:
                continue
            descriptors = self._descriptors[s].compute_batch(collection.patches)

            candidates: List[dict] = []
            for i, (descriptor, src_patch) in enumerate(zip(descriptors, collection.patches)):
                results = self._indices[s].search(descriptor, self.cfg.k_neighbors)
                src_coord = (
                    float(collection.coords[i, 0]),
                    float(collection.coords[i, 1]),
                )
                # Cache prior log-density once for this source patch
                prior_lp = None
                if self._priors.get(s) is not None:
                    prior_lp = float(
                        self._priors[s].log_density(src_patch[np.newaxis, ...])[0]
                    )
                for r in results:
                    if r.distance > self.cfg.distance_threshold:
                        continue
                    image_id = r.metadata.get("image_id", "")
                    patch_idx = int(r.metadata.get("patch_idx", 0))
                    if image_id not in self._target_patches[s]:
                        continue
                    tgt_patches = self._target_patches[s][image_id]
                    if patch_idx >= len(tgt_patches):
                        continue
                    tgt_patch = tgt_patches[patch_idx]
                    tgt_coord = self._target_coords[s][image_id][patch_idx]

                    bf = self._scorers[s].score(src_patch, tgt_patch, prior_log_density=prior_lp)
                    if not math.isfinite(bf.log_bf) or bf.log_bf < self.cfg.min_log_bf:
                        continue

                    candidates.append({
                        "src_idx": i,
                        "src_coord": src_coord,
                        "tgt_coord": (float(tgt_coord[0]), float(tgt_coord[1])),
                        "image_id": image_id,
                        "log_bf": float(bf.log_bf),
                    })
            if candidates:
                out[s] = {
                    "patches": collection.patches,
                    "coords": collection.coords,
                    "candidates": candidates,
                }
        return out

    def _build_accumulator(self, per_scale_data: Dict[int, dict]) -> HoughAccumulator4D:
        acc = HoughAccumulator4D()
        for s, data in per_scale_data.items():
            for c in data["candidates"]:
                acc.vote(HoughCandidate(
                    source_xy=c["src_coord"],
                    target_xy=c["tgt_coord"],
                    log_bf=c["log_bf"],
                    patch_size=s,
                ))
        return acc

    def _run_bp(
        self,
        per_scale_data: Dict[int, dict],
        peaks: List[PeakHypothesis],
    ):
        """Compute unaries per (scale, source patch) per (peak + null) and run BP."""
        n_labels = len(peaks) + 1   # K hypotheses + null
        unaries: Dict[int, np.ndarray] = {}
        coords: Dict[int, np.ndarray] = {}

        for s, data in per_scale_data.items():
            n = len(data["coords"])
            u = np.zeros((n, n_labels), dtype=np.float64)
            # Null label is a constant, hypotheses get their best supporting log-BF
            best_bf_per_label = np.full((n, len(peaks)), -np.inf, dtype=np.float64)
            for c in data["candidates"]:
                src_idx = c["src_idx"]
                cand_dx = c["tgt_coord"][0] - c["src_coord"][0]
                cand_dy = c["tgt_coord"][1] - c["src_coord"][1]
                for k, p in enumerate(peaks):
                    # Match a candidate to a peak by closeness in (dx, dy)
                    if abs(cand_dx - p.dx) <= 4.0 and abs(cand_dy - p.dy) <= 4.0:
                        best_bf_per_label[src_idx, k] = max(
                            best_bf_per_label[src_idx, k], c["log_bf"]
                        )
            # Where there's no support, log-evidence is 0 (no boost over null)
            best_bf_per_label = np.where(
                np.isfinite(best_bf_per_label), best_bf_per_label, 0.0
            )
            u[:, :len(peaks)] = best_bf_per_label
            # Null label is the prior; set to a small constant so a node
            # with no positive support stays near null.
            u[:, len(peaks)] = 0.5
            unaries[s] = u
            coords[s] = data["coords"].astype(np.float32)

        graph = build_multiscale_graph(
            unaries_per_scale=unaries,
            coords_per_scale=coords,
            spatial_radius=self.cfg.spatial_radius,
            spatial_beta=self.cfg.spatial_beta,
            inclusion_beta=self.cfg.inclusion_beta,
        )
        return loopy_bp(
            graph,
            max_iter=self.cfg.bp_max_iter,
            damping=self.cfg.bp_damping,
        )

    def _matches_from_bp(
        self,
        per_scale_data: Dict[int, dict],
        peaks: List[PeakHypothesis],
        marginals: np.ndarray,
    ) -> List[Match]:
        """Convert posterior assignments back to Match objects for downstream APIs."""
        matches: List[Match] = []
        # Walk graph in same order as build_multiscale_graph
        offset = 0
        for s in sorted(per_scale_data.keys()):
            data = per_scale_data[s]
            n = len(data["coords"])
            for i in range(n):
                row = marginals[offset + i]
                # Best non-null label
                if len(peaks) == 0:
                    offset_idx = None
                else:
                    best_label = int(np.argmax(row[:len(peaks)]))
                    if row[best_label] < row[len(peaks)]:
                        # Null wins
                        continue
                    offset_idx = best_label
                if offset_idx is None:
                    continue
                peak = peaks[offset_idx]
                src_xy = data["coords"][i]
                # Scale the peak's transform (here: pure translation)
                tgt_x = float(src_xy[0]) + peak.dx
                tgt_y = float(src_xy[1]) + peak.dy
                # Approximate image_id from any candidate at this src_idx
                # supporting this peak; fall back to first candidate.
                image_id = None
                for c in data["candidates"]:
                    if c["src_idx"] != i:
                        continue
                    if abs(c["tgt_coord"][0] - tgt_x) <= 4.0 and abs(c["tgt_coord"][1] - tgt_y) <= 4.0:
                        image_id = c["image_id"]
                        break
                if image_id is None:
                    continue
                matches.append(Match(
                    source_coord=(int(src_xy[0]), int(src_xy[1])),
                    target_coord=(int(tgt_x), int(tgt_y)),
                    patch_size=s,
                    probability=float(row[offset_idx]),
                    image_id=image_id,
                ))
            offset += n
        return matches

    @property
    def target_ids(self) -> List[str]:
        return list(self._target_shapes.keys())
