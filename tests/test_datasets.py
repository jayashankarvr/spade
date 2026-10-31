"""Tests for benchmark datasets, calibration and the runner."""

import csv

import numpy as np
import pytest
from PIL import Image

from spade.bench import calibration
from spade.bench.datasets import _split_donor_host, casia2, load, manifest, split_labels
from spade.bench.runner import main as runner_main


def _texture(seed, shape=(80, 96)):
    rng = np.random.RandomState(seed)
    from scipy.ndimage import gaussian_filter

    img = gaussian_filter(rng.rand(*shape, 3), (1.0, 1.0, 0))
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)


def _fake_casia(root, n=3):
    """Mini CASIA v2 tree with the real naming scheme: Tp_D_..._<idA>_<idB>_<n>."""
    au, tp, gt = (root / "CASIA2" / d for d in ("Au", "Tp", "CASIA 2 Groundtruth"))
    for d in (au, tp, gt):
        d.mkdir(parents=True)
    for i in range(n):
        host, donor = _texture(2 * i), _texture(2 * i + 1)
        Image.fromarray(host).save(au / f"Au_ani_{i:05d}.jpg", quality=98)
        Image.fromarray(donor).save(au / f"Au_sec_{i:05d}.jpg", quality=98)
        host, donor = (np.asarray(Image.open(au / f"Au_{c}_{i:05d}.jpg")) for c in ("ani", "sec"))
        tampered = host.copy()
        tampered[20:50, 30:70] = donor[10:40, 5:45]
        mask = np.zeros(host.shape[:2], np.uint8)
        mask[20:50, 30:70] = 255
        name = f"Tp_D_CND_M_N_ani{i:05d}_sec{i:05d}_{i:05d}"
        Image.fromarray(tampered).save(tp / f"{name}.tif")
        Image.fromarray(mask).save(gt / f"{name}_gt.png")
    Image.fromarray(_texture(99)).save(au / "Au_arc_00001.jpg")
    return root


class TestPhotos:
    def test_positives_and_negatives(self):
        pytest.importorskip("skimage")
        samples = load("photos", n=3, negatives=2, crop_size=96, fragment_size=24)
        pos, neg = split_labels(samples)
        assert len(pos) == 3 and len(neg) == 2
        for s in pos:
            assert s.mask.sum() == 24 * 24
            assert s.params["donor"] != s.params["host"]
        assert all(not s.mask.any() for s in neg)

    def test_deterministic(self):
        pytest.importorskip("skimage")
        a = load("photos", n=2, crop_size=96, seed=3)
        b = load("photos", n=2, crop_size=96, seed=3)
        assert all(np.array_equal(x.tampered, y.tampered) for x, y in zip(a, b))


class TestCasia:
    def test_donor_found_from_pixels(self, tmp_path):
        samples = casia2(root=str(_fake_casia(tmp_path)), negatives=2)
        pos, neg = split_labels(samples)
        assert len(pos) == 3 and len(neg) == 2
        for s in pos:
            # the donor is the 'sec' image: the pasted block came from it
            idx = int(s.params["ids"][1][3:])
            expected = np.asarray(Image.open(tmp_path / "CASIA2" / "Au" / f"Au_sec_{idx:05d}.jpg"))
            assert np.array_equal(s.donor, expected)
            assert s.mask.sum() == 30 * 40
        for s in neg:
            assert s.params["donor"] == "Au_arc_00001.jpg"  # only unrelated category available

    def test_split_rejects_when_neither_matches(self):
        tampered = _texture(1)
        mask = np.zeros(tampered.shape[:2], bool)
        mask[:10, :10] = True
        assert _split_donor_host(tampered, mask, _texture(2), _texture(3)) is None

    def test_missing_dataset_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="spade.bench.download"):
            casia2(root=str(tmp_path))


class TestManifest:
    def test_reads_csv(self, tmp_path):
        Image.fromarray(_texture(1)).save(tmp_path / "d.png")
        Image.fromarray(_texture(2)).save(tmp_path / "t.png")
        m = np.zeros((80, 96), np.uint8)
        m[:8, :8] = 255
        Image.fromarray(m).save(tmp_path / "m.png")
        with open(tmp_path / "set.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["donor", "tampered", "mask", "label"])
            w.writerow(["d.png", "t.png", "m.png", ""])
            w.writerow(["d.png", "t.png", "", "0"])
        samples = manifest(str(tmp_path / "set.csv"))
        assert [s.label for s in samples] == [1, 0]
        assert samples[0].mask.sum() == 64 and not samples[1].mask.any()


class TestCalibration:
    def test_table_counts_false_regions_and_recall(self):
        R, S = calibration.RegionRecord, calibration.SampleRecord
        records = [
            S(label=1, regions=[R(-5.0, True), R(-0.5, False)]),
            S(label=1, regions=[R(0.5, True)]),
            S(label=0, regions=[R(-2.0, False)]),
            S(label=0, regions=[]),
        ]
        rows = {r["epsilon"]: r for r in calibration.calibration_table(records, epsilons=(1e-3, 1.0, 10.0))}
        assert rows[1e-3]["false_regions_per_pair"] == 0.0 and rows[1e-3]["recall"] == 0.5
        assert rows[1.0]["false_regions_per_pair"] == pytest.approx(2 / 4)
        assert rows[1.0]["null_pairs_with_alarm"] == 0.5
        assert rows[10.0]["recall"] == 1.0

    def test_scores_real_samples(self):
        samples = load("synthetic", n=2, negatives=1)
        records = calibration.score_samples(samples)
        assert [r.label for r in records] == [1, 1, 0]
        assert all(any(reg.on_splice for reg in r.regions) for r in records[:2])
        summary = calibration.summary(records)
        assert summary["iou"] > 0.5 and summary["det_auc"] == 1.0


def test_runner_end_to_end(tmp_path):
    out = tmp_path / "r.json"
    report = runner_main(["--n", "2", "--negatives", "1", "--calibrate", "--no-baselines", "--json", str(out)])
    assert out.exists() and report["calibration"]
    assert report["results"]["spade"]["iou"] > 0.5
