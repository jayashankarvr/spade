"""Tests for spade.hashing: PDQ against Meta's reference vectors, pHash against imagehash."""

import csv
import os
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
    _load_phash_lib,
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
    assert h.packed.dtype == np.uint8 and h.packed.shape == (8,)
    assert h.packed[0] == 0x80


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

    hash_file = _load_phash_lib()
    if hash_file is None:
        pytest.skip("libpHash isn't installed")
    assert "phash-lib" in available_algorithms()
    for row in _expected():
        path = DATA / row["filename"]
        out = ctypes.c_uint64()
        assert hash_file(str(path).encode(), ctypes.byref(out)) == 0
        got = hash_image(load_image(str(path)), ["phash-lib"])[0]
        assert got.algorithm == "phash-lib"
        assert got.quality is None
        assert got.hex == f"{out.value:016x}"


def _fake_hasher(value=0xABC, rc=0, seen=None):
    def fn(path, out):
        if seen is not None:
            seen.append(os.fsdecode(path))
            assert os.path.exists(path)
        out._obj.value = value
        return rc

    return fn


def test_phash_lib_formats_hex(original):
    with mock.patch("spade.hashing._load_phash_lib", return_value=_fake_hasher(0xABC)):
        (h,) = hash_image(original, ["phash-lib"])
    assert (h.algorithm, h.hex, h.quality, h.variant) == ("phash-lib", "0000000000000abc", None, "original")


def test_phash_lib_dihedral(original):
    with mock.patch("spade.hashing._load_phash_lib", return_value=_fake_hasher()):
        hashes = hash_image(original, ["phash-lib"], dihedral=True)
    assert [h.variant for h in hashes] == list(VARIANTS)


def test_phash_lib_failure_and_temp_cleanup(original):
    seen = []
    with mock.patch("spade.hashing._load_phash_lib", return_value=_fake_hasher(seen=seen)):
        hash_image(original, ["phash-lib"])
    with mock.patch("spade.hashing._load_phash_lib", return_value=_fake_hasher(rc=1, seen=seen)):
        with pytest.raises(RuntimeError, match="libpHash"):
            hash_image(original, ["phash-lib"])
    assert len(seen) == 2
    assert not any(os.path.exists(p) for p in seen)


class _FakeCDLL:
    def __init__(self, **symbols):
        self.__dict__.update(symbols)


@pytest.fixture
def fresh_loader():
    _load_phash_lib.cache_clear()
    yield
    _load_phash_lib.cache_clear()


def test_phash_lib_mangled_symbol_fallback(fresh_loader, monkeypatch):
    monkeypatch.delenv("SPADE_PHASH_LIB", raising=False)
    fn = _fake_hasher()
    lib = _FakeCDLL(_Z16ph_dct_imagehashPKcRm=fn)
    with mock.patch("ctypes.util.find_library", return_value="libpHash.so.0"), mock.patch("ctypes.CDLL", return_value=lib):
        assert _load_phash_lib() is fn
        assert "phash-lib" in available_algorithms()


def test_phash_lib_loaded_without_symbol(fresh_loader, original, monkeypatch):
    monkeypatch.delenv("SPADE_PHASH_LIB", raising=False)
    with mock.patch("ctypes.util.find_library", return_value="/usr/lib/libpHash.so.0"), mock.patch("ctypes.CDLL", return_value=_FakeCDLL()):
        assert "phash-lib" not in available_algorithms()
        with pytest.raises(DependencyError, match=r"/usr/lib/libpHash.so.0.*doesn't export ph_dct_imagehash"):
            hash_image(original, ["phash-lib"])


def test_phash_lib_bad_env_path(fresh_loader, original, monkeypatch, tmp_path):
    bad = str(tmp_path / "nope.so")
    monkeypatch.setenv("SPADE_PHASH_LIB", bad)
    with mock.patch("ctypes.util.find_library", return_value="libpHash.so.0") as find:
        assert "phash-lib" not in available_algorithms()
        with pytest.raises(DependencyError, match="SPADE_PHASH_LIB=.*nope.so"):
            hash_image(original, ["phash-lib"])
        find.assert_not_called()


def test_available_algorithms_needs_the_packages(monkeypatch):
    monkeypatch.setitem(sys.modules, "pdqhash", None)
    monkeypatch.setitem(sys.modules, "imagehash", None)
    assert "pdq" not in available_algorithms()
    assert "phash" not in available_algorithms()
    with pytest.raises(DependencyError):
        hash_image(np.zeros((16, 16, 3), dtype=np.uint8), ["pdq"])


def _cimg_nearest_indices(old: int, new: int) -> list:
    """Source index for each output pixel, transcribed from CImg 2.2's resize
    (interpolation_type=1, the default Meta's pdqio.cpp uses): each step is
    ``floor((i + 1.0) * old / new) - floor(i * old / new)``, starting at 0."""
    out, pos, curr = [], 0, 0
    for i in range(new):
        out.append(pos)
        old_curr, curr = curr, int((i + 1.0) * old / new)
        pos += curr - old_curr
    return out


def test_pdq_downsample_picks_cimg_pixels():
    from spade.hashing import _downsample

    h, w = 700, 2000
    rows, cols = np.mgrid[0:h, 0:w]
    coords = np.dstack([rows, cols])  # each pixel holds its own (row, col)
    small = _downsample(coords)
    assert small.shape[:2] == (512, 512)
    assert list(small[:, 0, 0]) == _cimg_nearest_indices(h, 512)
    assert list(small[0, :, 1]) == _cimg_nearest_indices(w, 512)
    # A few by hand: floor(i * size / 512).
    assert [small[i, 0, 0] for i in (0, 1, 2, 511)] == [0, 1, 2, 698]
    assert [small[0, j, 1] for j in (0, 1, 2, 511)] == [0, 3, 7, 1996]
    # 512 or smaller on both sides is left alone.
    edge = np.zeros((512, 300, 3), np.uint8)
    assert _downsample(edge) is edge


@pytest.fixture(scope="module")
def robustness():
    pytest.importorskip("skimage")
    from spade.bench.hash_robustness import measure_pairs

    return measure_pairs(("pdq", "phash"))


# PDQ on the brick texture moves 58 bits when halved, whatever the resampling
# filter (56-80 measured). The photo is a fine, even texture with almost no
# large-scale structure for PDQ's 16x16 DCT to hold on to. 31 is Meta's
# published threshold and 13 photos are no reason to move it, so this one is
# recorded as a known miss rather than hidden by a looser threshold.
_KNOWN_MISSES = {("pdq", "brick", "resize 0.5x")}


def test_edited_copies_within_distance(robustness):
    from spade.bench.hash_robustness import REQUIRED_EDITS

    edited, _ = robustness
    for (algo, photo, edit), d in edited.items():
        if edit not in REQUIRED_EDITS:
            continue
        if (algo, photo, edit) in _KNOWN_MISSES:
            assert d > DEFAULT_DISTANCE[algo], "known miss now passes; drop it from _KNOWN_MISSES"
            continue
        assert d <= DEFAULT_DISTANCE[algo], (algo, photo, edit, d)


def test_unrelated_photos_beyond_distance(robustness):
    _, unrelated = robustness
    for (algo, a, b), d in unrelated.items():
        assert d > DEFAULT_DISTANCE[algo], (algo, a, b, d)


def test_dihedral_images_accepts_2d():
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
    variants = dihedral_images(gray)
    assert len(variants) == 8
    assert all(v.ndim == 3 and v.shape[2] == 3 for v in variants)


def test_pixel_types_keep_their_scale():
    # Same pixels in different types must give the same hash and quality,
    # dark frames included: stretching them would get past the quality gate.
    from spade.hashing import _as_rgb

    rng = np.random.default_rng(3)
    for top in (4, 255):
        pixels = rng.integers(0, top + 1, size=(64, 64, 3), dtype=np.uint8)
        forms = [
            pixels.astype(np.uint16) * 257,
            pixels.astype(np.float32) / 255,
            pixels.astype(np.float64) / 255,
            pixels.astype(np.int32),
        ]
        if top == 255:
            forms.append(pixels.astype(np.float64))  # 0..255 floats
        for form in forms:
            np.testing.assert_array_equal(_as_rgb(form), pixels, err_msg=str(form.dtype))
    np.testing.assert_array_equal(_as_rgb(np.ones((4, 4, 3), dtype=bool)), 255)


def test_dark_frame_quality_does_not_depend_on_type():
    pytest.importorskip("pdqhash")
    rng = np.random.default_rng(4)
    dark = rng.integers(0, 5, size=(256, 256, 3), dtype=np.uint8)
    reference = hash_image(dark, ["pdq"])[0]
    for form in (dark.astype(np.uint16) * 257, dark.astype(np.float64) / 255):
        h = hash_image(form, ["pdq"])[0]
        assert (h.hex, h.quality) == (reference.hex, reference.quality)


def test_unreadable_pixel_values_refused():
    from spade.hashing import _as_rgb

    bad = [np.full((4, 4, 3), -1.0), np.full((4, 4, 3), np.nan), np.full((4, 4, 3), 300.0),
           np.full((4, 4, 3), 300, dtype=np.int32), np.full((4, 4, 3), -1, dtype=np.int16)]
    for image in bad:
        with pytest.raises(ValueError):
            _as_rgb(image)


def test_popcount_table_matches_bitwise_count():
    from spade import hashing

    if not hasattr(np, "bitwise_count"):
        pytest.skip("numpy has no bitwise_count")
    rng = np.random.default_rng(3)
    packed = rng.integers(0, 256, (20, 32), dtype=np.uint8)
    query = rng.integers(0, 256, 32, dtype=np.uint8)
    xor = np.bitwise_xor(packed, query)
    table = hashing._POPCOUNT[xor].sum(axis=1, dtype=np.int64)
    assert np.array_equal(table, np.bitwise_count(xor).sum(axis=1, dtype=np.int64))
    assert np.array_equal(table, hashing._distances(packed, query))
