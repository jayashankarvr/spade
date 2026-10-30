"""Tests for the a-contrario region significance."""

import math

import pytest

from spade.verification.nfa import log10_nfa, overlap_factor


class TestNFA:
    def test_no_successes_is_just_the_number_of_tests(self):
        assert log10_nfa(0, 50, 0.01, 1000) == pytest.approx(3.0)

    def test_more_agreement_is_more_significant(self):
        values = [log10_nfa(k, 100, 0.01, 1e6) for k in (2, 5, 10, 20)]
        assert values == sorted(values, reverse=True)

    def test_chance_level_agreement_is_not_significant(self):
        # 1 hit in 100 at p = 0.01 is exactly what chance gives
        assert log10_nfa(1, 100, 0.01, 1e3) > 0

    def test_strong_region_is_significant(self):
        assert log10_nfa(30, 40, 0.01, 1e8) < -10

    def test_underflow_stays_finite(self):
        value = log10_nfa(400, 400, 1e-6, 1e9)
        assert math.isfinite(value) and value < -1000

    def test_overlap_factor(self):
        assert overlap_factor(5, 1) == 25
        assert overlap_factor(5, 5) == 1
        assert overlap_factor(3, 10) == 1
