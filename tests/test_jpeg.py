"""Tests for the JPEG rounding-error model."""

import io

import numpy as np
import pytest
from PIL import Image

from spade.jpeg import quantization_sigma, quantization_sigma_for_quality


def _save(tmp_path, name, fmt, **kw):
    img = Image.fromarray((np.random.RandomState(0).rand(32, 32, 3) * 255).astype(np.uint8))
    path = tmp_path / name
    img.save(path, format=fmt, **kw)
    return str(path)


def test_lower_quality_means_more_noise():
    sigmas = [quantization_sigma_for_quality(q) for q in (95, 85, 75, 50)]
    assert sigmas == sorted(sigmas)
    assert quantization_sigma_for_quality(75) == pytest.approx(0.038, abs=0.003)


def test_reads_file_tables(tmp_path):
    path = _save(tmp_path, "a.jpg", "JPEG", quality=75)
    assert quantization_sigma(path) == pytest.approx(quantization_sigma_for_quality(75))


def test_non_jpeg_is_none(tmp_path):
    assert quantization_sigma(_save(tmp_path, "a.png", "PNG")) is None
    assert quantization_sigma(str(tmp_path / "missing.jpg")) is None


def test_measured_error_is_close_to_model():
    # The model is an upper bound that should be in the right range for real content.
    from spade.bench.synthetic import _textured_image

    img = _textured_image(np.random.RandomState(1), 128, 128)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=85)
    back = np.asarray(Image.open(buf)).astype(np.float64)
    luma = np.array([0.299, 0.587, 0.114])
    measured = np.std((back - img) @ luma) / 255
    assert 0.3 * quantization_sigma_for_quality(85) < measured < 1.2 * quantization_sigma_for_quality(85)
