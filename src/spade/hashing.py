"""Perceptual image hashes: PDQ (Meta), pHash (imagehash) and libpHash (optional).

Every hash comes from the reference library (pdqhash, imagehash); nothing here
re-implements an algorithm. The libraries are optional, installed with
``pip install spade-forensics[hash]``. libpHash is GPLv3, so it is never bundled;
it is used only if the user installed it themselves.
"""

import csv
import ctypes
import ctypes.util
import functools
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from .exceptions import DependencyError

# Meta's order for pdqhash.compute_dihedral, and the order of the sample files
# in ThreatExchange's pdq/data/reg-test-input/dih.
VARIANTS = (
    "original",
    "rot90",
    "rot180",
    "rot270",
    "flip_x",
    "flip_y",
    "flip_diag",
    "flip_antidiag",
)

# Hamming distance at or below which two hashes count as a match.
DEFAULT_DISTANCE: Dict[str, int] = {"pdq": 31, "phash": 10, "phash-lib": 10}

# Meta's reference tool downsamples anything larger than this to a square of
# this size before hashing, with CImg's default (nearest neighbour) resize.
_PDQ_DOWNSAMPLE = 512

_INSTALL_HINT = "pip install spade-forensics[hash]"


@dataclass
class ImageHash:
    """One hash of one image (or of one dihedral variant of it)."""

    algorithm: str
    hex: str
    quality: Optional[int]
    variant: str = "original"

    @property
    def bits(self) -> np.ndarray:
        """The hash as packed bytes, ``len(hex) // 2`` of them."""
        return np.frombuffer(bytes.fromhex(self.hex), dtype=np.uint8)


def dihedral_images(image: np.ndarray) -> List[np.ndarray]:
    """The eight rotations and flips of an image, in ``VARIANTS`` order."""
    return [
        image,
        np.rot90(image, 1),
        np.rot90(image, 2),
        np.rot90(image, 3),
        image[::-1],
        image[:, ::-1],
        image.transpose(1, 0, 2),
        np.rot90(image, 2).transpose(1, 0, 2),
    ]


def _as_rgb(image: np.ndarray) -> np.ndarray:
    """Contiguous uint8 (H, W, 3) from a grayscale, RGB or RGBA array."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected a (H, W) or (H, W, 3) image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _bits_to_hex(bits: np.ndarray) -> str:
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes().hex()


class _Pdq:
    name = "pdq"
    bits = 256

    def compute(self, image: np.ndarray, dihedral: bool) -> List[ImageHash]:
        try:
            import pdqhash
        except ImportError as e:
            raise DependencyError(f"PDQ needs the pdqhash package: {_INSTALL_HINT}") from e

        rgb = _as_rgb(image)
        if dihedral:
            # Meta's tool hashes the transformed file, so downsample each
            # transform rather than the original once.
            rgbs = [_downsample(_as_rgb(t)) for t in dihedral_images(rgb)]
            results = [pdqhash.compute(t) for t in rgbs]
        else:
            results = [pdqhash.compute(_downsample(rgb))]
        names = VARIANTS if dihedral else VARIANTS[:1]
        return [
            ImageHash(self.name, _bits_to_hex(vec), int(quality), variant)
            for (vec, quality), variant in zip(results, names)
        ]


def _downsample(rgb: np.ndarray) -> np.ndarray:
    """Nearest-neighbour resize to 512x512 when either side exceeds 512."""
    h, w = rgb.shape[:2]
    if h <= _PDQ_DOWNSAMPLE and w <= _PDQ_DOWNSAMPLE:
        return rgb
    rows = np.arange(_PDQ_DOWNSAMPLE) * h // _PDQ_DOWNSAMPLE
    cols = np.arange(_PDQ_DOWNSAMPLE) * w // _PDQ_DOWNSAMPLE
    return np.ascontiguousarray(rgb[rows][:, cols])


class _Phash:
    name = "phash"
    bits = 64

    def compute(self, image: np.ndarray, dihedral: bool) -> List[ImageHash]:
        try:
            import imagehash
        except ImportError as e:
            raise DependencyError(f"pHash needs the imagehash package: {_INSTALL_HINT}") from e

        rgb = _as_rgb(image)
        images = dihedral_images(rgb) if dihedral else [rgb]
        names = VARIANTS if dihedral else VARIANTS[:1]
        return [
            ImageHash(self.name, str(imagehash.phash(Image.fromarray(np.ascontiguousarray(img)))), None, variant)
            for img, variant in zip(images, names)
        ]


_PHASH_LIB_HINT = (
    "libpHash isn't installed, so 'phash-lib' is unavailable. It is GPLv3 and "
    "SPADE doesn't ship it; install it separately (e.g. apt install libphash0), "
    "or point SPADE_PHASH_LIB at libpHash.so"
)


# pHash.h declares ph_dct_imagehash(const char*, ulong64&) in C++, so unless the
# header was built with extern "C" the symbol is mangled (64-bit Linux names).
_PHASH_SYMBOLS = (
    "ph_dct_imagehash",
    "_Z16ph_dct_imagehashPKcRm",  # ulong64 is unsigned long
    "_Z16ph_dct_imagehashPKcRy",  # ulong64 is unsigned long long
)


@functools.lru_cache(maxsize=None)
def _load_phash_lib():
    """The user's own ``ph_dct_imagehash`` as a callable, or None if libpHash isn't there.

    ``SPADE_PHASH_LIB`` wins, and with no fallback to the system search: a path
    that was set but won't load is an error, not a missing library. Raises
    DependencyError when a library was found but is unusable.
    """
    explicit = os.environ.get("SPADE_PHASH_LIB")
    path = explicit or ctypes.util.find_library("pHash")
    if not path:
        return None
    try:
        lib = ctypes.CDLL(path)
    except OSError as e:
        if explicit:
            raise DependencyError(f"SPADE_PHASH_LIB={path} couldn't be loaded: {e}") from e
        return None
    for symbol in _PHASH_SYMBOLS:
        try:
            fn = getattr(lib, symbol)
        except AttributeError:
            continue
        fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
        fn.restype = ctypes.c_int
        return fn
    raise DependencyError(f"libpHash was found at {path} but doesn't export ph_dct_imagehash")


class _PhashLib:
    """pHash's own DCT hash (``ph_dct_imagehash``), through the user's libpHash."""

    name = "phash-lib"
    bits = 64

    def compute(self, image: np.ndarray, dihedral: bool) -> List[ImageHash]:
        hash_file = _load_phash_lib()
        if hash_file is None:
            raise DependencyError(_PHASH_LIB_HINT)

        rgb = _as_rgb(image)
        images = dihedral_images(rgb) if dihedral else [rgb]
        names = VARIANTS if dihedral else VARIANTS[:1]
        out = []
        for img, variant in zip(images, names):
            # The library only reads files; PNG keeps the pixels exact.
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            try:
                Image.fromarray(np.ascontiguousarray(img)).save(path, format="PNG")
                value = ctypes.c_uint64()
                if hash_file(os.fsencode(path), ctypes.byref(value)) != 0:
                    raise RuntimeError(f"libpHash couldn't hash the image (variant '{variant}')")
            finally:
                os.remove(path)
            out.append(ImageHash(self.name, f"{value.value:016x}", None, variant))
        return out


_ALGORITHMS = {a.name: a for a in (_Pdq(), _Phash(), _PhashLib())}


def available_algorithms() -> List[str]:
    """Algorithms hashed in this process (never ``photodna``)."""
    names = list(_ALGORITHMS)
    try:
        usable = _load_phash_lib() is not None
    except DependencyError:
        usable = False
    if not usable:
        names.remove("phash-lib")
    return names


def hash_image(
    image: np.ndarray,
    algorithms: Optional[Sequence[str]] = None,
    dihedral: bool = False,
) -> List[ImageHash]:
    """Hash an image with each algorithm; with ``dihedral``, hash all eight variants."""
    if algorithms is None:
        algorithms = available_algorithms()
    out: List[ImageHash] = []
    for name in algorithms:
        if name == "photodna":
            raise ValueError("photodna isn't hashed locally; use spade.photodna")
        if name not in _ALGORITHMS:
            raise ValueError(f"unknown hash algorithm '{name}'; available: {available_algorithms()}")
        out.extend(_ALGORITHMS[name].compute(image, dihedral))
    return out


def hamming(a: Union[ImageHash, str], b: Union[ImageHash, str]) -> int:
    """Number of differing bits between two hashes of the same algorithm and length."""
    if isinstance(a, ImageHash) and isinstance(b, ImageHash) and a.algorithm != b.algorithm:
        raise ValueError(f"can't compare '{a.algorithm}' with '{b.algorithm}' hashes")
    ha = a.hex if isinstance(a, ImageHash) else a
    hb = b.hex if isinstance(b, ImageHash) else b
    if len(ha) != len(hb):
        raise ValueError(f"hash lengths differ ({len(ha)} vs {len(hb)} hex digits)")
    return bin(int(ha, 16) ^ int(hb, 16)).count("1")


# Hex digits in a list entry, per algorithm that is matched locally.
_HEX_LENGTH = {"pdq": 64, "phash": 16, "phash-lib": 16}


@dataclass
class HashList:
    """Known hashes to match against: parallel arrays of algorithm, hex and id."""

    algorithms: np.ndarray
    hexes: np.ndarray
    ids: np.ndarray
    # Per-algorithm (row indices, packed byte matrix), built on first use.
    _matrices: Dict[str, tuple] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.hexes)

    def matrix(self, algorithm: str) -> Optional[tuple]:
        """``(indices, packed)`` for one algorithm, or None if the list has none."""
        if algorithm not in self._matrices:
            indices = np.flatnonzero(self.algorithms == algorithm)
            if len(indices) == 0:
                self._matrices[algorithm] = None
            else:
                width = _HEX_LENGTH[algorithm] // 2
                raw = bytes.fromhex("".join(self.hexes[indices]))
                packed = np.frombuffer(raw, dtype=np.uint8).reshape(len(indices), width)
                self._matrices[algorithm] = (indices, packed)
        return self._matrices[algorithm]


def load_hash_list(path: str, algorithm: Optional[str] = None) -> HashList:
    """Read a CSV hash list.

    Rows are ``algorithm,hash,id``, ``hash,id`` or just ``hash``; the last two
    need ``algorithm``. A header line, blank lines, CRLF endings, upper-case hex
    and a UTF-8 BOM are all fine. Rows without an id get their line number
    in the file as id. A bad row raises ValueError naming its line.
    """
    algos: List[str] = []
    hexes: List[str] = []
    ids: List[str] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, skipinitialspace=True)
        first = True
        for row in reader:
            row = [cell.strip() for cell in row]
            if not any(row):
                continue
            line = reader.line_num
            if first:
                first = False
                if row[0].lower() in ("algorithm", "hash"):
                    continue
            if len(row) > 3:
                raise ValueError(f"{path}, line {line}: expected at most 3 columns, got {len(row)}")
            if len(row) == 3:
                name, value, ident = row[0].lower(), row[1], row[2]
            else:
                name, value = algorithm, row[0]
                ident = row[1] if len(row) == 2 else str(line)
                if name is None:
                    raise ValueError(
                        f"{path}, line {line}: no algorithm column and none given; pass algorithm="
                    )
                name = name.lower()
            if name != "photodna":
                if name not in _HEX_LENGTH:
                    raise ValueError(f"{path}, line {line}: unknown algorithm '{name}'")
                value = value.lower()
                if len(value) != _HEX_LENGTH[name]:
                    raise ValueError(
                        f"{path}, line {line}: a {name} hash has {_HEX_LENGTH[name]} hex digits, "
                        f"got {len(value)}"
                    )
                if any(c not in "0123456789abcdef" for c in value):
                    raise ValueError(f"{path}, line {line}: '{value}' isn't hex")
            algos.append(name)
            hexes.append(value)
            ids.append(ident)
    return HashList(
        algorithms=np.array(algos, dtype=object),
        hexes=np.array(hexes, dtype=object),
        ids=np.array(ids, dtype=object),
    )


@dataclass
class HashMatch:
    """A list entry within range of one of the image's hashes."""

    id: str
    algorithm: str
    hex: str
    distance: int
    variant: str


@dataclass
class MatchReport:
    matches: List[HashMatch]
    skipped_low_quality: List[ImageHash]


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _distances(packed: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Hamming distance from ``query`` to every row of ``packed``."""
    xor = np.bitwise_xor(packed, query)
    if hasattr(np, "bitwise_count"):  # numpy 2+
        if xor.shape[1] % 8 == 0:
            xor = xor.view(np.uint64)
        return np.bitwise_count(xor).sum(axis=1, dtype=np.int64)
    return _POPCOUNT[xor].sum(axis=1, dtype=np.int64)


def match_hashes(
    hashes: Sequence[ImageHash],
    hash_list: HashList,
    max_distance: Optional[int] = None,
    min_quality: int = 50,
) -> MatchReport:
    """Find list entries within ``max_distance`` of any of the hashes, nearest first.

    Only entries of the same algorithm are compared. ``max_distance`` defaults
    to the algorithm's ``DEFAULT_DISTANCE``. Hashes with a quality below
    ``min_quality`` are skipped (pHash has no quality and is never skipped).
    An entry matched by several variants is reported once, at its closest.
    """
    skipped: List[ImageHash] = []
    best: Dict[tuple, HashMatch] = {}
    for h in hashes:
        if h.quality is not None and h.quality < min_quality:
            skipped.append(h)
            continue
        limit = max_distance if max_distance is not None else DEFAULT_DISTANCE.get(h.algorithm)
        if limit is None or h.algorithm not in _HEX_LENGTH:
            continue
        found = hash_list.matrix(h.algorithm)
        if found is None:
            continue
        indices, packed = found
        if packed.shape[1] != len(h.bits):
            continue
        dist = _distances(packed, h.bits)
        for pos in np.flatnonzero(dist <= limit):
            idx = int(indices[pos])
            d = int(dist[pos])
            key = (h.algorithm, idx)
            if key not in best or d < best[key].distance:
                best[key] = HashMatch(
                    id=str(hash_list.ids[idx]),
                    algorithm=h.algorithm,
                    hex=str(hash_list.hexes[idx]),
                    distance=d,
                    variant=h.variant,
                )
    matches = sorted(best.values(), key=lambda m: m.distance)
    return MatchReport(matches=matches, skipped_low_quality=skipped)
