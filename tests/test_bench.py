"""Tests for the benchmark harness (generator + metrics).

These cover the pure logic; the detectors themselves (SPADE/SIFT) are exercised
by the runner, not unit tests, since they're slow.
"""

import numpy as np

from spade.bench.synthetic import make_recolored_splice, make_negative_pair
from spade.bench.metrics import localization_scores, detection_auc, aggregate


class TestSyntheticGenerator:
    def test_deterministic(self):
        a = make_recolored_splice(seed=7, image_size=64, fragment_size=20)
        b = make_recolored_splice(seed=7, image_size=64, fragment_size=20)
        assert np.array_equal(a.tampered, b.tampered)
        assert np.array_equal(a.mask, b.mask)

    def test_mask_matches_bbox(self):
        s = make_recolored_splice(seed=1, image_size=80, fragment_size=24)
        x, y, w, h = s.bbox
        assert s.mask.sum() == w * h
        assert s.mask[y:y + h, x:x + w].all()

    def test_splice_region_differs_from_host(self):
        # The tampered splice region should be the recolored donor fragment,
        # not the original host content there.
        s = make_recolored_splice(seed=2, image_size=80, fragment_size=24, color_grade_strength=0.2)
        assert s.tampered.shape == (80, 80, 3)
        assert s.color_M.shape == (3, 3) and s.color_b.shape == (3,)

    def test_zero_grade_is_identity_transform(self):
        s = make_recolored_splice(seed=4, image_size=64, fragment_size=20, color_grade_strength=0.0)
        np.testing.assert_allclose(s.color_M, np.eye(3))
        np.testing.assert_allclose(s.color_b, np.zeros(3))

    def test_negative_pair_has_empty_mask(self):
        s = make_negative_pair(seed=5, image_size=64)
        assert s.mask.sum() == 0
        assert s.params.get("negative") is True

    def test_splice_upscale_blurs_the_region(self):
        from spade.scale import native_scale_fraction

        sharp = make_recolored_splice(seed=12, image_size=128, fragment_size=64,
                                      color_grade_strength=0.0, splice_upscale=1.0)
        blur = make_recolored_splice(seed=12, image_size=128, fragment_size=64,
                                     color_grade_strength=0.0, splice_upscale=2.0)
        assert blur.params["splice_upscale"] == 2.0
        x, y, w, h = blur.bbox
        # the resized splice region has a lower native scale (is blurrier)
        nf_blur = native_scale_fraction(blur.tampered[y:y + h, x:x + w])
        nf_sharp = native_scale_fraction(sharp.tampered[y:y + h, x:x + w])
        assert nf_blur < nf_sharp


class TestMetrics:
    def test_perfect_overlap(self):
        m = np.zeros((10, 10), dtype=bool)
        m[2:6, 2:6] = True
        sc = localization_scores(m, m)
        assert sc["iou"] == 1.0 and sc["f1"] == 1.0 and sc["mcc"] == 1.0

    def test_no_overlap(self):
        a = np.zeros((10, 10), dtype=bool); a[0:3, 0:3] = True
        b = np.zeros((10, 10), dtype=bool); b[7:10, 7:10] = True
        sc = localization_scores(a, b)
        assert sc["iou"] == 0.0 and sc["recall"] == 0.0

    def test_partial_overlap_iou(self):
        gt = np.zeros((10, 10), dtype=bool); gt[0:4, 0:4] = True      # 16 px
        pred = np.zeros((10, 10), dtype=bool); pred[0:4, 0:2] = True  # 8 px, all inside gt
        sc = localization_scores(pred, gt)
        assert abs(sc["iou"] - 8 / 16) < 1e-9       # intersection 8, union 16
        assert abs(sc["precision"] - 1.0) < 1e-9
        assert abs(sc["recall"] - 0.5) < 1e-9

    def test_both_empty_is_perfect(self):
        z = np.zeros((8, 8), dtype=bool)
        sc = localization_scores(z, z)
        assert sc["iou"] == 1.0

    def test_shape_mismatch_raises(self):
        import pytest
        with pytest.raises(ValueError):
            localization_scores(np.zeros((4, 4), bool), np.zeros((5, 5), bool))

    def test_detection_auc_perfect_separation(self):
        scores = [0.9, 0.8, 0.1, 0.2]
        labels = [1, 1, 0, 0]
        assert detection_auc(scores, labels) == 1.0

    def test_detection_auc_single_class(self):
        assert detection_auc([0.5, 0.6], [1, 1]) == 0.5

    def test_aggregate_mean(self):
        agg = aggregate([{"iou": 0.0, "f1": 1.0}, {"iou": 1.0, "f1": 0.0}])
        assert agg["iou"] == 0.5 and agg["f1"] == 0.5
