"""Detectors compared in the benchmark.

Every detector implements the same contract:

    detect(donor, tampered) -> (pred_mask: np.ndarray[bool, HxW], score: float)

`pred_mask` localizes the splice in **tampered** coordinates; `score` is an
image-level confidence used for detection ROC-AUC. This keeps SPADE and the
baselines directly comparable.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class Detector:
    """Base detector interface."""

    name = "base"

    def detect(self, donor: np.ndarray, tampered: np.ndarray) -> Tuple[np.ndarray, float]:
        raise NotImplementedError


class SpadeDetector(Detector):
    """SPADE: index the donor, query with the tampered image, localize via the
    source coordinates of spatially-coherent matches."""

    name = "spade"

    def __init__(self, config: Optional[object] = None):
        from spade.engine import Config

        self.config = config or Config(
            patch_size=3,
            entropy_threshold=None,
            min_probability=0.5,
            coherence_enabled=True,
            coherence_min_cluster=3,
            auto_train_pca=False,
        )

    def detect(self, donor: np.ndarray, tampered: np.ndarray) -> Tuple[np.ndarray, float]:
        from spade.engine import ForensicsEngine

        h, w = tampered.shape[:2]
        engine = ForensicsEngine(self.config)
        if engine.index_target(donor, "donor") == 0:
            return np.zeros((h, w), dtype=bool), 0.0

        result = engine.match(tampered, return_heatmap=False)

        # Prefer coherent-region matches (high precision); fall back to all matches.
        if result.coherent_regions:
            matches = [m for r in result.coherent_regions for m in r.matches]
        else:
            matches = result.matches

        # Spatial-density localization: keep the largest connected component of
        # the matched footprints. Its area fraction is the detection score
        # (match counts are ~random; spatial structure discriminates).
        from spade.aggregation.localize import localize_region

        loc = localize_region(matches, (h, w), largest_component=True)
        return loc.mask, loc.area_fraction


class RootSiftRansacDetector(Detector):
    """RootSIFT keypoint matching + RANSAC geometric verification.

    The classic local-feature retrieval baseline: match RootSIFT descriptors
    between donor and tampered, geometrically verify with RANSAC, and fill the
    convex hull of the inlier keypoints in the tampered image.
    """

    name = "rootsift+ransac"

    def __init__(self, ratio: float = 0.75, min_inliers: int = 4):
        self.ratio = ratio
        self.min_inliers = min_inliers

    @staticmethod
    def _root_sift(descriptors: np.ndarray) -> np.ndarray:
        # L1-normalize then square-root (RootSIFT): Hellinger kernel in L2 space.
        descriptors = descriptors / (descriptors.sum(axis=1, keepdims=True) + 1e-7)
        return np.sqrt(descriptors)

    def detect(self, donor: np.ndarray, tampered: np.ndarray) -> Tuple[np.ndarray, float]:
        import cv2

        h, w = tampered.shape[:2]
        empty = np.zeros((h, w), dtype=bool)

        gray_d = cv2.cvtColor(donor, cv2.COLOR_RGB2GRAY)
        gray_t = cv2.cvtColor(tampered, cv2.COLOR_RGB2GRAY)

        sift = cv2.SIFT_create()
        kp_d, des_d = sift.detectAndCompute(gray_d, None)
        kp_t, des_t = sift.detectAndCompute(gray_t, None)
        if des_d is None or des_t is None or len(kp_d) < 2 or len(kp_t) < 2:
            return empty, 0.0

        des_d = self._root_sift(des_d.astype(np.float32))
        des_t = self._root_sift(des_t.astype(np.float32))

        bf = cv2.BFMatcher(cv2.NORM_L2)
        knn = bf.knnMatch(des_t, des_d, k=2)  # query = tampered, train = donor
        good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < self.ratio * n.distance]
        if len(good) < self.min_inliers:
            return empty, float(len(good))

        pts_t = np.float32([kp_t[m.queryIdx].pt for m in good])
        pts_d = np.float32([kp_d[m.trainIdx].pt for m in good])

        _, inliers = cv2.estimateAffinePartial2D(pts_d, pts_t, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if inliers is None:
            return empty, float(len(good))

        inlier_pts = pts_t[inliers.ravel().astype(bool)]
        n_inliers = len(inlier_pts)
        if n_inliers < self.min_inliers:
            return empty, float(n_inliers)

        mask = np.zeros((h, w), dtype=np.uint8)
        if n_inliers >= 3:
            hull = cv2.convexHull(inlier_pts.astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 1)
        else:
            for px, py in inlier_pts.astype(int):
                mask[max(0, py - 1):py + 2, max(0, px - 1):px + 2] = 1
        return mask.astype(bool), float(n_inliers)
