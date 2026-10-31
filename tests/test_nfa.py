"""Tests for the a-contrario region significance."""

import math

import pytest

from spade.verification.nfa import fdr_select, log10_nfa, overlap_factor


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


class TestFDR:
    @staticmethod
    def _result(log10_ps, n_tests):
        from spade.engine import MatchResult
        from spade.verification.coherence import CoherentRegion

        regions = [
            CoherentRegion(offset=(0, 0), matches=[], source_bbox=(0, 0, 1, 1), target_bbox=(0, 0, 1, 1),
                           confidence=0.0, log10_nfa=lp + math.log10(n_tests), n_tests=n_tests)
            for lp in log10_ps
        ]
        return MatchResult([], None, None, regions, {"n_tests": n_tests})

    def test_benjamini_hochberg_cutoff(self):
        # m = 100 hypotheses, q = 0.1: thresholds are i * 0.001
        res = self._result([math.log10(v) for v in (0.0005, 0.0015, 0.0029, 0.02)], 100)
        chosen = fdr_select([res], q=0.1)
        assert [round(10 ** r.log10_p, 4) for r in chosen] == [0.0005, 0.0015, 0.0029]

    def test_searches_without_regions_still_count(self):
        hit = self._result([math.log10(0.0005)], 100)
        empty = self._result([], 900)
        assert len(fdr_select([hit], q=0.1)) == 1
        assert fdr_select([hit, empty], q=0.1) == []  # 0.0005 > 1 * 0.1 / 1000

    def test_nothing_to_select(self):
        assert fdr_select([], q=0.05) == []
