"""Tests for CLI parameter validation."""

import pytest

click = pytest.importorskip("click")

from spade.cli import validate_params


class TestValidateParams:
    def test_accepts_full_patch_size_range(self):
        for ps in (3, 4, 8, 12, 16):
            validate_params(0.5, ps, 2.5)  # should not raise

    def test_rejects_patch_size_out_of_range(self):
        for ps in (2, 17, 32):
            with pytest.raises(click.BadParameter, match="between 3 and 16"):
                validate_params(0.5, ps, 2.5)

    def test_rejects_bad_threshold(self):
        with pytest.raises(click.BadParameter, match="threshold"):
            validate_params(1.5, 3, 2.5)

    def test_rejects_negative_entropy(self):
        with pytest.raises(click.BadParameter, match="entropy"):
            validate_params(0.5, 3, -1.0)
