"""Tests for match-based localization (spatial-density filtering)."""

import numpy as np

from spade.aggregation.heatmap import Match
from spade.aggregation.localize import (
    largest_connected_component,
    localize,
    localize_region,
    mask_to_bbox,
    match_footprint_mask,
)


def _m(x, y, ps=3):
    return Match(source_coord=(x, y), target_coord=(0, 0), patch_size=ps, probability=0.9)


class TestLargestConnectedComponent:
    def test_keeps_only_biggest(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:10, 2:10] = True       # big blob (64 px)
        mask[15, 15] = True           # scattered speck
        out = largest_connected_component(mask)
        assert out[2:10, 2:10].all()
        assert not out[15, 15]

    def test_empty_mask(self):
        mask = np.zeros((5, 5), dtype=bool)
        assert not largest_connected_component(mask).any()

    def test_single_component_unchanged(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:6, 3:6] = True
        assert np.array_equal(largest_connected_component(mask), mask)


class TestLocalize:
    def test_footprint_mask_marks_patch_area(self):
        mask = match_footprint_mask([_m(1, 1, ps=3)], (10, 10))
        assert mask[1:4, 1:4].all()
        assert mask.sum() == 9

    def test_localize_drops_scattered_matches(self):
        # A dense cluster plus two isolated stray matches.
        matches = [_m(x, y) for x in range(2, 8) for y in range(2, 8)]
        matches += [_m(18, 18), _m(0, 18)]
        full = localize(matches, (24, 24), largest_component=False)
        filtered = localize(matches, (24, 24), largest_component=True)
        assert filtered.sum() < full.sum()
        assert not filtered[18:21, 18:21].any()   # stray removed

    def test_mask_to_bbox(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 2:5] = True
        assert mask_to_bbox(mask) == (2, 3, 3, 4)

    def test_mask_to_bbox_empty(self):
        assert mask_to_bbox(np.zeros((4, 4), dtype=bool)) == (0, 0, 0, 0)


class TestLocalizeRegion:
    def test_area_fraction_is_detection_score(self):
        matches = [_m(x, y) for x in range(2, 8) for y in range(2, 8)]
        loc = localize_region(matches, (20, 20))
        assert loc.area == int(loc.mask.sum())
        assert 0.0 < loc.area_fraction <= 1.0
        assert abs(loc.area_fraction - loc.area / 400.0) < 1e-9

    def test_more_matches_higher_score(self):
        small = [_m(x, y) for x in range(2, 5) for y in range(2, 5)]
        large = [_m(x, y) for x in range(2, 14) for y in range(2, 14)]
        assert localize_region(large, (24, 24)).area_fraction > localize_region(small, (24, 24)).area_fraction

    def test_empty_matches_zero_score(self):
        loc = localize_region([], (16, 16))
        assert loc.area == 0 and loc.area_fraction == 0.0 and loc.bbox == (0, 0, 0, 0)
