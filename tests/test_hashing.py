"""Tests for spade.hashing: PDQ against Meta's reference vectors, pHash against imagehash."""

import csv
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from spade.engine import load_image
from spade.exceptions import DependencyError
from spade.hashing import (
    DEFAULT_DISTANCE,
    VARIANTS,
    ImageHash,
    available_algorithms,
    dihedral_images,
    hamming,
    hash_image,
)

pytest.importorskip("pdqhash")
imagehash = pytest.importorskip("imagehash")

DATA = Path(__file__).parent / "data" / "hashes"


def _expected():
    with open(DATA / "expected.csv", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def original():
    return load_image(str(DATA / "bridge-1-original.jpg"))


def test_constants():
    assert VARIANTS == (
        "original", "rot90", "rot180", "rot270",
        "flip_x", "flip_y", "flip_diag", "flip_antidiag",
    )
    assert DEFAULT_DISTANCE == {"pdq": 31, "phash": 10, "phash-lib": 10}
    assert "photodna" not in available_algorithms()
    assert {"pdq", "phash"} <= set(available_algorithms())


def test_pdq_matches_meta_reference():
    for row in _expected():
        img = load_image(str(DATA / row["filename"]))
        h = hash_image(img, ["pdq"])[0]
        assert h.algorithm == "pdq"
        assert h.hex == row["hash"], row["filename"]
        assert h.quality == int(row["quality"])


def test_pdq_dihedral_matches_rotated_files(original):
    hashes = hash_image(original, ["pdq"], dihedral=True)
    assert [h.variant for h in hashes] == list(VARIANTS)
    assert [h.hex for h in hashes] == [r["hash"] for r in _expected()]


def test_dihedral_images_follow_meta_order(original):
    expected = [r["hash"] for r in _expected()]
    imgs = dihedral_images(original)
    assert len(imgs) == 8
    for i, img in enumerate(imgs):
        assert hash_image(img, ["pdq"])[0].hex == expected[i], VARIANTS[i]


def test_phash_matches_imagehash():
    for row in _expected():
        path = DATA / row["filename"]
        want = str(imagehash.phash(Image.open(path).convert("RGB")))
        got = hash_image(load_image(str(path)), ["phash"])[0]
        assert got.algorithm == "phash"
        assert got.hex == want
        assert got.quality is None


def test_phash_dihedral_hashes_each_transform(original):
    hashes = hash_image(original, ["phash"], dihedral=True)
    assert [h.variant for h in hashes] == list(VARIANTS)
    for h, img in zip(hashes, dihedral_images(original)):
        assert h.hex == str(imagehash.phash(Image.fromarray(img)))


def test_odd_image_modes_hash(tmp_path):
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, (64, 80), dtype=np.uint8)
    assert len(hash_image(gray)) == len(available_algorithms())

    rgb = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    alpha = np.dstack([rgb, np.full((64, 80), 255, np.uint8)])
    rgba_path = tmp_path / "rgba.png"
    Image.fromarray(alpha, "RGBA").save(rgba_path)
    pal_path = tmp_path / "pal.png"
    Image.fromarray(rgb).convert("P").save(pal_path)

    rgba = load_image(str(rgba_path))
    pal = load_image(str(pal_path))
    assert len(hash_image(pal)) == len(available_algorithms())
    for algo in ("pdq", "phash"):
        assert hash_image(rgba, [algo])[0].hex == hash_image(rgb, [algo])[0].hex


def test_tiny_image():
    img = np.random.default_rng(1).integers(0, 256, (4, 4, 3), dtype=np.uint8)
    pdq = hash_image(img, ["pdq"])[0]
    assert pdq.quality < 50
    assert len(hash_image(img, ["phash"])[0].hex) == 16


def test_hamming():
    a = ImageHash("pdq", "00" * 32, 100)
    assert hamming(a, a) == 0
    assert hamming(a, "00" * 31 + "01") == 1
    assert hamming("ff", "00") == 8
    p = ImageHash("phash", "00" * 8, None)
    with pytest.raises(ValueError):
        hamming(a, p)
    with pytest.raises(ValueError):
        hamming("00", "0000")


def test_bits_packed():
    h = ImageHash("phash", "80" + "00" * 7, None)
    assert h.bits.dtype == np.uint8 and h.bits.shape == (8,)
    assert h.bits[0] == 0x80


def test_unknown_and_photodna(original):
    with pytest.raises(ValueError, match="photodna isn't hashed locally"):
        hash_image(original, ["photodna"])
    with pytest.raises(ValueError):
        hash_image(original, ["nope"])


def test_missing_dependency(original):
    with mock.patch.dict(sys.modules, {"pdqhash": None}):
        with pytest.raises(DependencyError, match=r"spade-forensics\[hash\]"):
            hash_image(original, ["pdq"])


def test_phash_lib_missing_is_clear(original):
    with mock.patch("spade.hashing._load_phash_lib", return_value=None):
        assert "phash-lib" not in available_algorithms()
        with pytest.raises(DependencyError, match="GPLv3"):
            hash_image(original, ["phash-lib"])


def test_phash_lib_matches_library():
    import ctypes

    from spade.hashing import _load_phash_lib

    lib = _load_phash_lib()
    if lib is None:
        pytest.skip("libpHash isn't installed")
    assert "phash-lib" in available_algorithms()
    for row in _expected():
        path = DATA / row["filename"]
        out = ctypes.c_uint64()
        assert lib.ph_dct_imagehash(str(path).encode(), ctypes.byref(out)) == 0
        got = hash_image(load_image(str(path)), ["phash-lib"])[0]
        assert got.algorithm == "phash-lib"
        assert got.quality is None
        assert got.hex == f"{out.value:016x}"
