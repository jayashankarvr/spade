"""Tests for native-scale estimation and the scale-inconsistency cue."""

import numpy as np
from PIL import Image

from spade.scale import (
    native_scale_fraction,
    scale_inconsistency,
    scale_inconsistency_map,
    scale_inconsistency_score,
)


def _texture(seed, size):
    rng = np.random.RandomState(seed)
    # high-frequency detail + a few hard-edged blobs (real content at native scale)
    img = (rng.rand(size, size, 3) * 255).astype(np.uint8)
    yy, xx = np.ogrid[:size, :size]
    for _ in range(8):
        cy, cx, r = rng.randint(0, size), rng.randint(0, size), rng.randint(3, size // 4)
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = rng.randint(0, 255, 3)
    return img


def _upscale(img, factor):
    h, w = img.shape[:2]
    small = Image.fromarray(img).resize((int(w / factor), int(h / factor)), Image.LANCZOS)
    return np.asarray(small.resize((w, h), Image.BICUBIC))


class TestNativeScaleFraction:
    def test_native_image_is_full_scale(self):
        img = _texture(0, 96)
        assert native_scale_fraction(img) >= 0.85

    def test_upscaled_image_detected(self):
        img = _upscale(_texture(1, 96), factor=2.0)
        # ~2x upscaled -> native fraction clearly below 1.0
        assert native_scale_fraction(img) <= 0.65

    def test_returns_fraction_in_range(self):
        v = native_scale_fraction(_texture(2, 64))
        assert 0.0 < v <= 1.0

    def test_accepts_float_image(self):
        img = _texture(3, 64).astype(np.float32) / 255.0
        assert 0.0 < native_scale_fraction(img) <= 1.0


class TestScaleInconsistencyCue:
    def test_uniform_image_low_inconsistency(self):
        img = _texture(4, 160)
        assert scale_inconsistency_score(img, window=64, stride=48) < 0.25

    def test_resized_splice_flagged(self):
        host = _texture(5, 160)
        frag = _upscale(_texture(6, 64), factor=2.0)  # blurry, lower native scale
        tampered = host.copy()
        # Place the splice on a window-grid position (stride 32) so one window
        # lands purely on it; the cue is about the region, not sub-window alignment.
        tampered[64:128, 64:128] = frag
        grid = scale_inconsistency_map(tampered, window=64, stride=32)
        # host windows read ~1.0; the splice-aligned window reads clearly lower
        assert np.nanmin(grid) <= 0.7
        assert scale_inconsistency_score(tampered, window=64, stride=32) > 0.2

    def test_map_shape(self):
        grid = scale_inconsistency_map(_texture(7, 128), window=64, stride=32)
        assert grid.ndim == 2 and np.isfinite(grid).all()


class TestScaleInconsistencyResult:
    def test_uniform_image_low_score(self):
        si = scale_inconsistency(_texture(8, 192), window=64, stride=32)
        assert si.score < 0.25
        assert 0.0 < si.min_fraction <= 1.0

    def test_resized_region_flagged_and_localized(self):
        host = _texture(9, 192)
        frag = _upscale(_texture(10, 64), factor=2.0)  # blurry resized splice
        tampered = host.copy()
        tampered[64:128, 64:128] = frag
        si = scale_inconsistency(tampered, window=64, stride=32)
        assert si.score > 0.2
        # the flagged anomaly window should overlap the splice region
        ax, ay, aw, ah = si.anomaly_bbox
        assert 32 <= ax <= 96 and 32 <= ay <= 96

    def test_score_delegates(self):
        img = _texture(11, 160)
        assert scale_inconsistency_score(img) == scale_inconsistency(img).score
