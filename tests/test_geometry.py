"""Tests for copy-move detection and rotation-invariant matching."""

import numpy as np
import pytest

from spade import Config, ForensicsEngine
from spade.aggregation.heatmap import Match
from spade.bench.synthetic import make_copy_move, make_negative_pair
from spade.extraction.orientation import canonical_patches, patch_orientations
from spade.verification.coherence import find_rigid_regions


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


class TestCopyMove:
    def test_finds_duplicate_within_image(self):
        sample = make_copy_move(seed=1)
        result = ForensicsEngine().match_self(sample.tampered)
        assert result.significant_regions
        # Both the original and the copy are localized.
        assert _iou(result.localization.mask, sample.mask) > 0.6

    def test_reports_each_pair_once(self):
        result = ForensicsEngine().match_self(make_copy_move(seed=1).tampered)
        offsets = [r.offset for r in result.significant_regions]
        assert not any((-dx, -dy) in offsets for dx, dy in offsets)

    def test_clean_image_has_no_duplicates(self):
        result = ForensicsEngine().match_self(make_negative_pair(seed=201).tampered)
        assert not result.significant_regions

    def test_repetitive_texture_is_not_a_copy(self):
        # Brick and gravel match themselves at many offsets; with no single
        # offset standing out, nothing should be reported as duplicated.
        pytest.importorskip("skimage")
        from spade.bench.datasets import load_photos

        photos = load_photos()
        for name in ("brick", "gravel"):
            crop = np.ascontiguousarray(photos[name][40:200, 40:200])
            assert not ForensicsEngine().match_self(crop).significant_regions

    def test_does_not_touch_engine_index(self):
        engine = ForensicsEngine()
        engine.match_self(make_copy_move(seed=1).tampered)
        assert engine.index.size == 0 and engine.target_count == 0


class TestOrientation:
    def test_exact_rotation_shifts_angle_and_keeps_patch(self):
        from scipy.ndimage import gaussian_filter

        img = gaussian_filter(np.random.RandomState(0).rand(80, 80, 3), (1.5, 1.5, 0)).astype(np.float32)
        rot = np.ascontiguousarray(np.rot90(img, k=-1))
        s = 5
        coords = np.array([[30, 30], [40, 22], [25, 45]])
        centers = coords + (s - 1) / 2
        rot_coords = (np.stack([79 - centers[:, 1], centers[:, 0]], 1) - (s - 1) / 2).astype(int)
        t1 = patch_orientations(img, coords, s)
        t2 = patch_orientations(rot, rot_coords, s)
        np.testing.assert_allclose((t2 - t1) % (2 * np.pi), np.pi / 2, atol=1e-6)
        np.testing.assert_allclose(canonical_patches(img, coords, s, t1), canonical_patches(rot, rot_coords, s, t2), atol=1e-5)


class TestRigidClustering:
    def test_recovers_rotation_and_translation(self):
        rng = np.random.default_rng(0)
        theta = np.radians(170)
        c, s = np.cos(theta), np.sin(theta)
        matches = []
        for _ in range(30):
            x, y = rng.integers(20, 60, 2)
            cx, cy = x + 2, y + 2
            tx, ty = c * cx - s * cy + 100, s * cx + c * cy + 50
            angle = (theta + rng.normal(0, 0.05) + np.pi) % (2 * np.pi) - np.pi
            matches.append(Match((int(x), int(y)), (int(round(tx - 2)), int(round(ty - 2))), 5, 0.9, "a", rotation=float(angle)))
        for _ in range(40):
            matches.append(Match(tuple(rng.integers(0, 90, 2).tolist()), tuple(rng.integers(0, 150, 2).tolist()),
                                 5, 0.9, "a", rotation=float(rng.uniform(-3, 3))))
        best = find_rigid_regions(matches)[0]
        assert len(best.matches) == 30
        assert np.degrees(best.rotation) == pytest.approx(170, abs=1)
        assert best.offset == (100, 50)


def _mirrored_splice(seed=0):
    from spade.bench.synthetic import _textured_image

    rng = np.random.RandomState(seed)
    donor = _textured_image(rng, 96, 96)
    host = _textured_image(rng, 96, 96)
    tampered = host.copy()
    tampered[50:80, 10:40] = donor[20:50, 40:70][:, ::-1]
    mask = np.zeros((96, 96), bool)
    mask[50:80, 10:40] = True
    return donor, tampered, mask


class TestMirroredCopies:
    def test_found_only_with_flip_invariance(self):
        donor, tampered, mask = _mirrored_splice()
        plain = ForensicsEngine(Config(flip_invariant=False))
        plain.index_target(donor, "d")
        assert not (plain.match(tampered).localization.mask & mask).any()

        flip = ForensicsEngine(Config(flip_invariant=True))
        flip.index_target(donor, "d")
        result = flip.match(tampered)
        assert result.significant_regions and result.significant_regions[0].mirrored
        assert _iou(result.localization.mask, mask) > 0.6

    def test_mirrored_region_reports_no_offset(self):
        from spade.report import build_report

        donor, tampered, _ = _mirrored_splice()
        engine = ForensicsEngine()
        engine.index_target(donor, "d")
        result = engine.match(tampered)
        region = build_report(result=result, config=engine.config)["result"]["coherent_regions"][0]
        assert region["mirrored"] and region["offset"] is None

    def test_heatmap_falls_back_to_matches_without_regions(self):
        # With mirroring on (two passes) a match that forms no region still gets a heatmap.
        donor, tampered, _ = _mirrored_splice()
        engine = ForensicsEngine(Config(coherence_enabled=False))
        engine.index_target(donor, "d")
        result = engine.match(tampered)
        assert result.matches and not result.coherent_regions
        assert result.heatmap is not None

    def test_mirrored_search_counts_its_tests(self):
        donor, tampered, _ = _mirrored_splice()
        plain, flip = ForensicsEngine(Config(flip_invariant=False)), ForensicsEngine(Config(flip_invariant=True))
        for engine in (plain, flip):
            engine.index_target(donor, "d")
        assert flip.match(tampered).stats["n_tests"] == pytest.approx(2 * plain.match(tampered).stats["n_tests"])


class TestTargetStorage:
    def test_target_pixels_stored_once(self):
        engine = ForensicsEngine()
        image = (np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8)
        engine.index_target(image, "a")
        target = engine._targets["a"]
        assert target.levels[1.0].dtype == np.uint8 and np.array_equal(target.levels[1.0], image)


class TestQueryScales:
    def test_enlarged_copy_needs_query_scales(self):
        from PIL import Image

        from spade.bench.synthetic import _textured_image

        rng = np.random.RandomState(4)
        donor, host = _textured_image(rng, 128, 128), _textured_image(rng, 128, 128)
        fragment = np.asarray(Image.fromarray(donor[40:72, 40:72]).resize((48, 48), Image.BICUBIC))
        tampered = host.copy()
        tampered[60:108, 20:68] = fragment
        mask = np.zeros((128, 128), bool)
        mask[60:108, 20:68] = True

        plain = ForensicsEngine(Config(flip_invariant=False))
        plain.index_target(donor, "d")
        assert _iou(plain.match(tampered).localization.mask, mask) < 0.2

        scaled = ForensicsEngine(Config(flip_invariant=False, query_scales=[1.0, 0.75, 0.5]))
        scaled.index_target(donor, "d")
        result = scaled.match(tampered)
        assert result.significant_regions and result.significant_regions[0].source_scale < 1.0
        assert _iou(result.localization.mask, mask) > 0.5

    def test_query_scales_validated(self):
        with pytest.raises(ValueError, match="query_scales"):
            Config(query_scales=[0.5])


def test_rotation_alone_turns_the_default_pyramid_off():
    config = Config(rotation_invariant=True)
    assert config.rotation_invariant and not config.pyramid_enabled
