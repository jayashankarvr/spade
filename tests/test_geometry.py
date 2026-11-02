"""Tests for copy-move detection and rotation-invariant matching."""

import numpy as np
import pytest

from spade import Config, ForensicsEngine
from spade.aggregation.heatmap import Match
from spade.bench.synthetic import make_copy_move, make_negative_pair
from spade.extraction.orientation import RingDescriptor, relative_angles, ring_samples
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


class TestRings:
    def test_exact_rotation_keeps_descriptor_and_measures_angle(self):
        from scipy.ndimage import gaussian_filter

        img = gaussian_filter(np.random.RandomState(0).rand(80, 80, 3), (1.5, 1.5, 0)).astype(np.float32)
        rot = np.ascontiguousarray(np.rot90(img, k=-1))   # a quarter turn clockwise on screen
        s = 5
        coords = np.array([[30, 30], [40, 22], [25, 45]])
        centers = coords + (s - 1) / 2
        rot_coords = np.stack([79 - centers[:, 1], centers[:, 0]], 1) - (s - 1) / 2
        a, b = ring_samples(img, coords, s), ring_samples(rot, rot_coords, s)
        desc = RingDescriptor(s)
        np.testing.assert_allclose(desc.compute_batch(a)[0], desc.compute_batch(b)[0], atol=1e-4)
        gray = lambda r: r @ np.array([0.299, 0.587, 0.114], np.float32)
        angle = relative_angles(gray(a), gray(b))
        np.testing.assert_allclose(np.angle(np.exp(1j * (angle - np.pi / 2))), 0, atol=1e-3)
        # Read at the measured angle, the target rings line up with the source.
        np.testing.assert_allclose(ring_samples(rot, rot_coords, s, phase=angle), a, atol=1e-3)


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
        engine = ForensicsEngine(Config(flip_invariant=True))
        engine.index_target(donor, "d")
        result = engine.match(tampered)
        region = build_report(result=result, config=engine.config)["result"]["coherent_regions"][0]
        assert region["mirrored"] and region["offset"] is None

    def test_heatmap_falls_back_to_matches_without_regions(self):
        # With mirroring on (two passes) a match that forms no region still gets a heatmap.
        donor, tampered, _ = _mirrored_splice()
        engine = ForensicsEngine(Config(coherence_enabled=False, flip_invariant=True))
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


class TestSymmetricObjects:
    def _region(self, src, tgt):
        from spade.verification.coherence import CoherentRegion

        return CoherentRegion(offset=(0, 0), matches=[], source_bbox=src, target_bbox=tgt, confidence=0.0, mirrored=True)

    def test_halves_meeting_at_an_axis_are_a_symmetric_object(self):
        from spade.engine import _halves_touch

        # Left and right half of a round object: side by side, same rows.
        assert _halves_touch(self._region((5, 3, 697, 1401), (704, 3, 697, 1401)), 10)

    def test_mirrored_copy_elsewhere_is_kept(self):
        from spade.engine import _halves_touch

        assert not _halves_touch(self._region((10, 10, 40, 40), (120, 15, 40, 40)), 10)   # far apart
        assert not _halves_touch(self._region((10, 10, 40, 40), (52, 90, 40, 40)), 10)    # adjacent column, other rows
        assert not _halves_touch(self._region((10, 10, 40, 40), (55, 22, 40, 40)), 10)    # adjacent, shifted 12 px down

    def test_symmetric_image_has_no_copy_move(self):
        rng = np.random.RandomState(0)
        half = (rng.rand(64, 48, 3) * 255).astype(np.uint8)
        from scipy.ndimage import gaussian_filter

        half = gaussian_filter(half.astype(float), (1, 1, 0)).astype(np.uint8)
        symmetric = np.concatenate([half, half[:, ::-1]], axis=1)
        assert not ForensicsEngine().match_self(symmetric).significant_regions


class TestConnectedParts:
    def test_copy_and_far_strays_are_split(self):
        from spade.engine import _connected_parts

        block = np.array([(x, y) for x in range(100, 140) for y in range(50, 90)])
        strays = np.array([(5, 5), (400, 300), (250, 10)])
        xy = np.concatenate([block, strays])
        parts = _connected_parts(np.arange(len(xy)), xy, gap=5)
        sizes = sorted(len(p) for p in parts)
        assert sizes == [1, 1, 1, len(block)]

    def test_linking_distance_and_row_ids(self):
        from spade.engine import _connected_parts

        # Cells are gap-sized; points in neighbouring cells link, points two cells apart don't.
        xy = np.array([(0, 0), (9, 0), (25, 0)])
        parts = _connected_parts(np.array([7, 8, 9]), xy, gap=5)
        assert sorted(sorted(p.tolist()) for p in parts) == [[7, 8], [9]]

    def test_copy_with_strays_at_its_offset_is_found(self):
        # Chance matches far away at the copy's offset used to stretch its box
        # over the image and hide it.
        from spade.bench.synthetic import _textured_image

        rng = np.random.RandomState(7)
        donor, host = _textured_image(rng, 160, 160), _textured_image(rng, 160, 160)
        tampered = host.copy()
        tampered[60:90, 70:100] = donor[20:50, 30:60]
        engine = ForensicsEngine()
        engine.index_target(donor, "d")
        result = engine.match(tampered)
        mask = np.zeros((160, 160), bool)
        mask[60:90, 70:100] = True
        assert result.significant_regions
        assert _iou(result.localization.mask, mask) > 0.5

    def test_copy_move_is_not_split(self, monkeypatch):
        import spade.engine as eng

        calls = []
        real = eng._connected_parts
        monkeypatch.setattr(eng, "_connected_parts", lambda *a: calls.append(1) or real(*a))
        ForensicsEngine().match_self(make_copy_move(seed=1).tampered)
        assert not calls

    def test_unrelated_photos_stay_quiet(self):
        # camera vs astronaut: smooth areas at one offset formed a tight blob
        # that scored NFA 10^-0.3 before the region-size choice was counted.
        skimage = pytest.importorskip("skimage.data")
        camera = np.repeat(skimage.camera()[:512, :512, None], 3, axis=2)
        engine = ForensicsEngine()
        engine.index_target(camera, "camera")
        assert not engine.match(skimage.astronaut()[:512, :512], return_heatmap=False).significant_regions


class TestRotationMode:
    """End to end with Config(rotation_invariant=True)."""

    # Rings need angular structure: smooth synthetic textures don't have
    # enough, so these use photo content (scikit-image's sample photos).
    @staticmethod
    def _photo(name):
        skimage = pytest.importorskip("skimage.data")
        img = getattr(skimage, name)()
        return np.repeat(img[..., None], 3, axis=2) if img.ndim == 2 else img[..., :3]

    def test_rotated_splice_found_with_its_turn(self):
        from PIL import Image

        donor, host = self._photo("astronaut")[100:260, 100:260], self._photo("coffee")[100:260, 100:260]
        window = self._photo("astronaut")[130:210, 130:210]
        tampered = host.copy()
        tampered[60:108, 50:98] = np.asarray(Image.fromarray(window).rotate(50, resample=Image.BICUBIC))[16:64, 16:64]
        mask = np.zeros(tampered.shape[:2], bool)
        mask[60:108, 50:98] = True
        engine = ForensicsEngine(Config(rotation_invariant=True, flip_invariant=False))
        engine.index_target(donor, "d")
        result = engine.match(tampered, return_heatmap=False)
        assert result.significant_regions
        # The region's turn is PIL's angle (PIL turns content counter-clockwise on screen).
        turn = np.degrees(result.significant_regions[0].rotation)
        assert abs((turn - 50 + 180) % 360 - 180) < 5
        # Regions cover the rings' footprint, so they reach close to the fragment's edges.
        assert _iou(result.localization.mask, mask) > 0.6

    @pytest.mark.xfail(reason="copy-move's repeated-texture guard compares translation offsets, "
                              "which a rotated copy spreads out; being worked on", strict=False)
    def test_rotated_copy_move_found(self):
        from PIL import Image

        image = self._photo("astronaut")[80:272, 80:272].copy()
        window = image[10:90, 10:90]
        image[120:168, 120:168] = np.asarray(Image.fromarray(window).rotate(70, resample=Image.BICUBIC))[16:64, 16:64]
        result = ForensicsEngine(Config(rotation_invariant=True)).match_self(image)
        assert result.significant_regions

    def test_round_object_turning_onto_itself_is_not_a_copy(self):
        rng = np.random.RandomState(0)
        yy, xx = np.mgrid[:200, :200]
        r = np.hypot(xx - 100, yy - 100)
        profile = np.interp(r, np.arange(0, 80, 4), rng.rand(20))
        img = np.where(r < 70, profile, 0.5)[..., None] * np.array([0.9, 0.8, 0.7]) + rng.normal(0, 0.01, (200, 200, 3))
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        assert not ForensicsEngine(Config(rotation_invariant=True)).match_self(img).significant_regions

    def test_unrelated_pair_stays_quiet(self):
        negative = make_negative_pair(seed=203)
        engine = ForensicsEngine(Config(rotation_invariant=True))
        engine.index_target(negative.donor, "d")
        assert not engine.match(negative.tampered, return_heatmap=False).significant_regions

    def test_bundle_from_another_descriptor_is_refused(self, tmp_path):
        from spade.exceptions import IndexStoreError

        engine = ForensicsEngine(Config(rotation_invariant=True))
        engine.index_target(make_negative_pair(seed=204).donor, "d")
        engine.save(str(tmp_path / "rot"))
        assert ForensicsEngine.load(str(tmp_path / "rot")).config.rotation_invariant
        meta = tmp_path / "rot.engine.json"
        import json

        state = json.loads(meta.read_text())
        state["descriptor"] = "LuminanceDescriptor"   # what an older rotation-mode bundle used
        meta.write_text(json.dumps(state))
        with pytest.raises(IndexStoreError):
            ForensicsEngine.load(str(tmp_path / "rot"))

class TestThorough:
    """Config(thorough=True): the common checks plus mirrored, enlarged and rotated copies."""

    @staticmethod
    def _splices():
        from PIL import Image

        skimage = pytest.importorskip("skimage.data")
        donor_full, host = skimage.astronaut(), skimage.coffee()
        donor = donor_full[100:260, 100:260]
        out = {}
        for kind in ("mirrored", "rotated"):
            window = donor_full[130:210, 130:210]
            piece = window[16:64, 16:64][:, ::-1] if kind == "mirrored" else \
                np.asarray(Image.fromarray(window).rotate(50, resample=Image.BICUBIC))[16:64, 16:64]
            tampered = host[100:260, 100:260].copy()
            tampered[60:108, 50:98] = piece
            out[kind] = tampered
        return donor, out

    def test_finds_what_the_default_skips(self):
        donor, tampered = self._splices()
        fast, thorough = ForensicsEngine(), ForensicsEngine(Config(thorough=True))
        for engine in (fast, thorough):
            engine.index_target(donor, "d")
        for kind, image in tampered.items():
            assert not fast.match(image, return_heatmap=False).significant_regions, kind
            assert thorough.match(image, return_heatmap=False).significant_regions, kind

    def test_counts_both_searches(self):
        donor, tampered = self._splices()
        engine = ForensicsEngine(Config(thorough=True))
        engine.index_target(donor, "d")
        plain = engine._search(tampered["mirrored"], None, False, 0.0)
        combined = engine.match(tampered["mirrored"], return_heatmap=False)
        assert combined.stats["n_tests"] >= 2 * plain.stats["n_tests"]

    def test_save_load_and_remove(self, tmp_path):
        donor, tampered = self._splices()
        engine = ForensicsEngine(Config(thorough=True))
        engine.index_target(donor, "d")
        engine.save(str(tmp_path / "b"))
        loaded = ForensicsEngine.load(str(tmp_path / "b"))
        assert loaded._rotated is not None and loaded._rotated.target_ids == ["d"]
        loaded.remove_target("d")
        assert loaded._rotated.target_ids == []

    def test_copy_move_has_no_rotation_search(self, monkeypatch):
        calls = []
        real = ForensicsEngine.__init__

        def spy(self, config=None):
            calls.append(bool(config and config.rotation_invariant))
            real(self, config)

        monkeypatch.setattr(ForensicsEngine, "__init__", spy)
        ForensicsEngine(Config(thorough=True)).match_self(make_copy_move(seed=1).tampered)
        # The thorough engine, its rotation helper, then copy-move's engine without one.
        assert calls == [False, True, False]
