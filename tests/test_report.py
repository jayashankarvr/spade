"""Tests for the JSON forensic report contract."""

import json

import numpy as np

from spade import Config
from spade.engine import MatchResult
from spade.aggregation.heatmap import Match
from spade.verification.coherence import CoherentRegion
from spade.report import SCHEMA_VERSION, build_report, to_json, _color_transform


def _match(prob=0.9, with_transform=True):
    return Match(
        source_coord=(1, 2),
        target_coord=(10, 20),
        patch_size=3,
        probability=prob,
        image_id="target",
        color_M=np.eye(3) * 1.1 if with_transform else None,
        color_b=np.array([0.05, 0.0, -0.02]) if with_transform else None,
    )


def _result(matches):
    best = max(matches, key=lambda m: m.probability) if matches else None
    return MatchResult(
        matches=matches,
        best_match=best,
        heatmap=None,
        coherent_regions=[],
        stats={"source_patches": 5, "total_matches": len(matches), "coherent_regions": 0},
    )


class TestColorTransform:
    def test_identity_flagged(self):
        ct = _color_transform(np.eye(3), np.zeros(3))
        assert ct["is_identity"] is True
        assert ct["M"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def test_non_identity(self):
        ct = _color_transform(np.eye(3) * 1.2, np.array([0.1, 0.0, 0.0]))
        assert ct["is_identity"] is False

    def test_none_when_missing(self):
        assert _color_transform(None, None) is None

    def test_rejects_wrong_shape(self):
        assert _color_transform(np.eye(2), np.zeros(2)) is None


class TestBuildReport:
    def test_schema_and_top_level_keys(self):
        report = build_report(result=_result([_match()]), config=Config())
        assert report["schema_version"] == SCHEMA_VERSION
        for key in ("tool", "generated_at", "inputs", "config", "result", "artifacts"):
            assert key in report

    def test_best_match_includes_color_transform(self):
        report = build_report(result=_result([_match()]), config=Config())
        bm = report["result"]["best_match"]
        assert bm is not None
        assert bm["color_transform"] is not None
        assert bm["color_transform"]["is_identity"] is False
        assert report["result"]["match_found"] is True

    def test_no_match(self):
        report = build_report(result=_result([]), config=Config())
        assert report["result"]["match_found"] is False
        assert report["result"]["best_match"] is None
        assert report["result"]["num_matches"] == 0

    def test_config_subset_recorded(self):
        report = build_report(result=_result([]), config=Config(patch_size=4))
        assert report["config"]["patch_size"] == 4
        assert "min_probability" in report["config"]

    def test_coherent_region_color_transform(self):
        matches = [_match(), _match()]
        region = CoherentRegion(
            offset=(9, 18),
            matches=matches,
            source_bbox=(1, 2, 3, 3),
            target_bbox=(10, 20, 3, 3),
            confidence=0.88,
        )
        result = MatchResult(
            matches=matches,
            best_match=matches[0],
            heatmap=None,
            coherent_regions=[region],
            stats={"source_patches": 5, "total_matches": 2, "coherent_regions": 1},
        )
        report = build_report(result=result, config=Config())
        regions = report["result"]["coherent_regions"]
        assert len(regions) == 1
        assert regions[0]["num_matches"] == 2
        assert regions[0]["color_transform"] is not None
        assert regions[0]["source_bbox"] == {"x": 1, "y": 2, "width": 3, "height": 3}

    def test_round_trips_to_json(self):
        report = build_report(result=_result([_match()]), config=Config())
        # Must be serializable (no numpy types leaking through)
        text = to_json(report)
        reloaded = json.loads(text)
        assert reloaded["schema_version"] == SCHEMA_VERSION

    def test_deterministic_timestamp_override(self):
        report = build_report(
            result=_result([]), config=Config(), generated_at="2026-01-01T00:00:00+00:00"
        )
        assert report["generated_at"] == "2026-01-01T00:00:00+00:00"
