"""End-to-end tests for the matching pipeline, including regressions for past bugs."""

import numpy as np
import pytest
from PIL import Image

from spade import Config, ForensicsEngine
from spade.bench.synthetic import make_negative_pair, make_recolored_splice
from spade.engine import load_image
from spade.exceptions import TargetExistsError


@pytest.fixture(scope="module")
def splice():
    return make_recolored_splice(seed=0, image_size=96, fragment_size=24)


def _true_offset(sample):
    fx, fy, _, _ = sample.params["donor_region"]
    x, y, _, _ = sample.bbox
    return fx - x, fy - y


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


class TestDetection:
    def test_recolored_splice_found_and_localized(self, splice):
        engine = ForensicsEngine()
        engine.index_target(splice.donor, "donor")
        result = engine.match(splice.tampered)

        assert result.significant_regions
        best = result.coherent_regions[0]
        assert best.significant and best.image_id == "donor"
        dx, dy = _true_offset(splice)
        assert abs(best.offset[0] - dx) <= 2 and abs(best.offset[1] - dy) <= 2
        assert _iou(result.localization.mask, splice.mask) > 0.7

    def test_region_recovers_color_transform(self, splice):
        engine = ForensicsEngine()
        engine.index_target(splice.donor, "donor")
        region = engine.match(splice.tampered).coherent_regions[0]
        # The engine fits donor = M @ tampered + b, the inverse of the applied grade.
        expected = np.linalg.inv(splice.color_M)
        assert np.abs(region.color_M - expected).max() < 0.15

    def test_survives_jpeg(self):
        sample = make_recolored_splice(seed=1, image_size=96, fragment_size=24, jpeg_quality=90)
        engine = ForensicsEngine()
        engine.index_target(sample.donor, "donor")
        result = engine.match(sample.tampered)
        assert result.significant_regions
        assert _iou(result.localization.mask, sample.mask) > 0.5

    def test_unrelated_images_not_significant(self):
        # Per-patch matches between unrelated images are common; significant
        # regions must not be. (Plain textures here: the benchmark's negatives
        # contain hard-edged disks, and two disks of the same radius are
        # pixel-identical shapes - a real, non-chance match.)
        from scipy.ndimage import gaussian_filter

        rng = np.random.RandomState(0)
        for _ in range(3):
            donor, source = (
                (np.clip(gaussian_filter(rng.rand(96, 96, 3), (1.5, 1.5, 0)) * 3 - 1, 0, 1) * 255).astype(np.uint8)
                for _ in range(2)
            )
            engine = ForensicsEngine()
            engine.index_target(donor, "donor")
            result = engine.match(source)
            assert result.matches  # plenty of chance patch matches...
            assert not result.significant_regions  # ...but no significant region
            assert not result.localization.mask.any()

    def test_picks_the_right_target(self, splice):
        other = make_negative_pair(seed=7, image_size=96).donor
        engine = ForensicsEngine()
        engine.index_target(other, "other")
        engine.index_target(splice.donor, "donor")
        result = engine.match(splice.tampered)
        assert result.coherent_regions[0].image_id == "donor"
        assert result.best_match.image_id == "donor"

    def test_target_ids_filter(self, splice):
        engine = ForensicsEngine()
        engine.index_target(splice.donor, "donor")
        engine.index_target(make_negative_pair(seed=7).donor, "other")
        result = engine.match(splice.tampered, target_ids=["other"])
        assert all(m.image_id == "other" for m in result.matches)
        assert not result.significant_regions

    def test_heatmap_shape(self, splice):
        engine = ForensicsEngine()
        engine.index_target(splice.donor, "donor")
        result = engine.match(splice.tampered, return_heatmap=True)
        assert result.heatmap.shape == splice.donor.shape[:2]
        assert 0.0 <= result.heatmap.min() and result.heatmap.max() <= 1.0


class TestSignificance:
    def test_max_nfa_is_honoured(self, splice):
        engine = ForensicsEngine()
        engine.index_target(splice.donor, "donor")
        best = engine.match(splice.tampered).coherent_regions[0].log10_nfa
        assert best < 0

        # A threshold just below the best region's NFA must turn it off everywhere.
        strict = ForensicsEngine(Config(max_nfa=10 ** (best - 1)))
        strict.index_target(splice.donor, "donor")
        result = strict.match(splice.tampered)
        assert result.coherent_regions and not result.significant_regions
        assert not result.localization.mask.any()

    def test_targets_without_hits_still_count_as_tests(self, splice):
        # Every target searched is a set of hypotheses tried, hit or not.
        lone = ForensicsEngine()
        lone.index_target(splice.donor, "donor")
        crowded = ForensicsEngine()
        crowded.index_target(splice.donor, "donor")
        for i in range(9):
            # same size as the donor, but flat: no patches, so never a hit
            crowded.index_target(np.full((96, 96, 3), 60 + 10 * i, np.uint8), f"flat{i}")
        a = lone.match(splice.tampered).coherent_regions[0].log10_nfa
        b = crowded.match(splice.tampered).coherent_regions[0].log10_nfa
        assert b == pytest.approx(a + 1.0, abs=0.05)  # 10x the targets -> +1 in log10

    def test_flat_source_returns_empty_localization(self, splice):
        engine = ForensicsEngine()
        engine.index_target(splice.donor, "donor")
        result = engine.match(np.full((40, 40, 3), 90, np.uint8))
        assert result.localization is not None and not result.localization.mask.any()


class TestIndexLifecycle:
    def test_indexing_order_does_not_matter(self, splice):
        # Regression: PCA used to be trained part-way through indexing, leaving
        # earlier targets in a different descriptor space and unmatchable.
        other = make_negative_pair(seed=7).donor

        def offsets(order):
            engine = ForensicsEngine()
            for image, image_id in order:
                engine.index_target(image, image_id)
            result = engine.match(splice.tampered)
            return [(r.image_id, r.offset) for r in result.significant_regions]

        first = offsets([(splice.donor, "donor"), (other, "other")])
        second = offsets([(other, "other"), (splice.donor, "donor")])
        assert first and first == second

    def test_remove_target_frees_index(self, splice):
        # Regression: removal used to keep the vectors in FAISS, where they kept
        # taking nearest-neighbour slots.
        engine = ForensicsEngine()
        engine.index_target(splice.donor, "donor")
        engine.index_target(make_negative_pair(seed=7).donor, "other")
        before = engine.index.size
        assert engine.remove_target("donor")
        assert engine.index.size < before
        assert not engine.has_target("donor")
        result = engine.match(splice.tampered)
        assert all(m.image_id == "other" for m in result.matches)

    def test_max_targets_evicts_oldest(self):
        engine = ForensicsEngine(Config(max_targets=2))
        for i in range(3):
            engine.index_target(make_negative_pair(seed=i, image_size=48).donor, f"t{i}")
        assert engine.target_ids == ["t1", "t2"]
        assert engine.index.image_ids == ["t1", "t2"]

    def test_failed_index_does_not_evict(self, monkeypatch):
        engine = ForensicsEngine(Config(max_targets=2))
        for i in range(2):
            engine.index_target(make_negative_pair(seed=i, image_size=48).donor, f"t{i}")

        def boom(image):
            raise MemoryError("simulated")

        monkeypatch.setattr(engine, "_describe_target", boom)
        with pytest.raises(MemoryError):
            engine.index_target(make_negative_pair(seed=5, image_size=48).donor, "t2")
        assert engine.target_ids == ["t0", "t1"]

    def test_empty_target_does_not_steal_matches(self, splice):
        engine = ForensicsEngine()
        assert engine.index_target(np.full((64, 64, 3), 100, np.uint8), "flat") == 0
        engine.index_target(splice.donor, "donor")
        result = engine.match(splice.tampered)
        assert result.significant_regions
        assert {m.image_id for m in result.matches} == {"donor"}

    def test_duplicate_id_rejected(self, splice):
        engine = ForensicsEngine()
        engine.index_target(splice.donor, "donor")
        with pytest.raises(TargetExistsError):
            engine.index_target(splice.donor, "donor")

    def test_save_load_and_search(self, splice, tmp_path):
        # Regression: `spade index` + `spade search` used to query with a different
        # descriptor projection than the one the index was built with.
        donor_path = tmp_path / "donor.png"
        Image.fromarray(splice.donor).save(donor_path)

        engine = ForensicsEngine()
        engine.index_target(load_image(str(donor_path)), "donor", image_path=str(donor_path))
        expected = engine.match(splice.tampered).coherent_regions[0].offset
        engine.save(str(tmp_path / "bundle"))

        loaded = ForensicsEngine.load(str(tmp_path / "bundle"))
        assert loaded.target_ids == ["donor"]
        assert loaded.index.size == engine.index.size
        result = loaded.match(splice.tampered)  # patches reloaded from donor.png
        assert result.significant_regions
        assert result.coherent_regions[0].offset == expected

    def test_load_keeps_index_shaping_settings(self, splice, tmp_path):
        engine = ForensicsEngine(Config(patch_size=4))
        engine.index_target(splice.donor, "donor")
        engine.save(str(tmp_path / "bundle"))
        loaded = ForensicsEngine.load(str(tmp_path / "bundle"), patch_size=6, min_probability=0.2)
        assert loaded.config.patch_size == 4
        assert loaded.config.min_probability == 0.2

    def test_loads_bundle_without_jpeg_sigma(self, splice, tmp_path):
        # Bundles written before JPEG-aware noise stored 4 fields per target.
        import json

        engine = ForensicsEngine()
        engine.index_target(splice.donor, "donor")
        engine.save(str(tmp_path / "bundle"))
        meta = tmp_path / "bundle.engine.json"
        state = json.loads(meta.read_text())
        state["targets"] = [row[:4] for row in state["targets"]]
        meta.write_text(json.dumps(state))
        loaded = ForensicsEngine.load(str(tmp_path / "bundle"))
        assert loaded._targets["donor"].jpeg_sigma == 0.0

    def test_load_applies_nprobe_override(self, splice, tmp_path):
        engine = ForensicsEngine()
        engine.index_target(splice.donor, "donor")
        engine.save(str(tmp_path / "bundle"))
        loaded = ForensicsEngine.load(str(tmp_path / "bundle"), ivf_nprobe=64)
        assert loaded.index.nprobe == 64
        assert loaded.config.index_type == loaded.index.kind

    def test_missing_image_file_is_skipped(self, splice, tmp_path):
        engine = ForensicsEngine()
        engine.index_target(splice.donor, "donor", image_path=str(tmp_path / "gone.png"))
        engine.save(str(tmp_path / "bundle"))
        loaded = ForensicsEngine.load(str(tmp_path / "bundle"))
        result = loaded.match(splice.tampered)
        assert result.matches == []
