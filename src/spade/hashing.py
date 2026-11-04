"""Perceptual image hashes: PDQ (Meta) and pHash (imagehash).

Every hash comes from the reference library (pdqhash, imagehash); nothing here
re-implements an algorithm. The libraries are optional, installed with
``pip install spade-forensics[hash]``.
"""

from dataclasses import dataclass
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


_ALGORITHMS = {a.name: a for a in (_Pdq(), _Phash())}


def available_algorithms() -> List[str]:
    """Algorithms hashed in this process (never ``photodna``)."""
    return list(_ALGORITHMS)


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
