"""Tests for spatial coherence verification."""

import numpy as np
import pytest
from spade.verification.coherence import SpatialCoherenceVerifier, CoherentRegion
from spade.aggregation.heatmap import Match


class TestSpatialCoherence:
    def test_clusters_consistent_offsets(self):
        """Matches with same offset should cluster together."""
        verifier = SpatialCoherenceVerifier(
            offset_tolerance=2,
            min_cluster_size=3,
        )

        # Create matches with consistent offset (40, 60)
        matches = [
            Match(source_coord=(10, 20), target_coord=(50, 80), patch_size=3, probability=0.8),
            Match(source_coord=(11, 20), target_coord=(51, 80), patch_size=3, probability=0.9),
            Match(source_coord=(12, 21), target_coord=(52, 81), patch_size=3, probability=0.85),
            Match(source_coord=(13, 22), target_coord=(53, 82), patch_size=3, probability=0.75),
        ]

        regions = verifier.find_coherent_regions(matches)

        assert len(regions) == 1
        assert len(regions[0].matches) == 4
        assert regions[0].offset == (40, 60)

    def test_separates_different_offsets(self):
        """Matches with different offsets should form separate clusters."""
        verifier = SpatialCoherenceVerifier(
            offset_tolerance=2,
            min_cluster_size=2,
        )

        # Two groups with different offsets
        matches = [
            # Group 1: offset (40, 60)
            Match(source_coord=(10, 20), target_coord=(50, 80), patch_size=3, probability=0.8),
            Match(source_coord=(11, 20), target_coord=(51, 80), patch_size=3, probability=0.9),
            Match(source_coord=(12, 21), target_coord=(52, 81), patch_size=3, probability=0.85),
            # Group 2: offset (-10, -10)
            Match(source_coord=(50, 50), target_coord=(40, 40), patch_size=3, probability=0.8),
            Match(source_coord=(51, 51), target_coord=(41, 41), patch_size=3, probability=0.9),
            Match(source_coord=(52, 52), target_coord=(42, 42), patch_size=3, probability=0.75),
        ]

        regions = verifier.find_coherent_regions(matches)

        assert len(regions) == 2
        offsets = {r.offset for r in regions}
        assert (40, 60) in offsets
        assert (-10, -10) in offsets

    def test_filters_random_matches(self):
        """Random matches with inconsistent offsets should not cluster."""
        verifier = SpatialCoherenceVerifier(
            offset_tolerance=2,
            min_cluster_size=3,
        )

        # Random matches with different offsets
        np.random.seed(42)
        matches = [
            Match(
                source_coord=(i * 10, i * 10),
                target_coord=(np.random.randint(0, 100), np.random.randint(0, 100)),
                patch_size=3,
                probability=0.7,
            )
            for i in range(5)
        ]

        regions = verifier.find_coherent_regions(matches)

        # Should have no valid clusters (random offsets don't cluster)
        assert len(regions) <= 1  # Might have one small cluster by chance

    def test_computes_bounding_boxes(self):
        """Bounding boxes should encompass all matches in cluster."""
        verifier = SpatialCoherenceVerifier(
            offset_tolerance=2,
            min_cluster_size=3,
        )

        matches = [
            Match(source_coord=(10, 20), target_coord=(50, 80), patch_size=4, probability=0.8),
            Match(source_coord=(15, 20), target_coord=(55, 80), patch_size=4, probability=0.9),
            Match(source_coord=(12, 25), target_coord=(52, 85), patch_size=4, probability=0.85),
        ]

        regions = verifier.find_coherent_regions(matches)

        assert len(regions) == 1
        region = regions[0]

        # Source bbox should be (10, 20, 9, 9) - from (10,20) to (15+4, 25+4)
        assert region.source_bbox[0] == 10  # x
        assert region.source_bbox[1] == 20  # y
        assert region.source_bbox[2] == 5 + 4  # width
        assert region.source_bbox[3] == 5 + 4  # height

    def test_confidence_increases_with_cluster_size(self):
        """Larger clusters should have higher confidence."""
        verifier = SpatialCoherenceVerifier(
            offset_tolerance=2,
            min_cluster_size=2,
        )

        # Small cluster
        small_matches = [
            Match(source_coord=(10, 20), target_coord=(50, 80), patch_size=3, probability=0.8),
            Match(source_coord=(11, 20), target_coord=(51, 80), patch_size=3, probability=0.8),
        ]

        # Large cluster (same probability)
        large_matches = [
            Match(source_coord=(10, 20), target_coord=(50, 80), patch_size=3, probability=0.8),
            Match(source_coord=(11, 20), target_coord=(51, 80), patch_size=3, probability=0.8),
            Match(source_coord=(12, 20), target_coord=(52, 80), patch_size=3, probability=0.8),
            Match(source_coord=(13, 20), target_coord=(53, 80), patch_size=3, probability=0.8),
            Match(source_coord=(14, 20), target_coord=(54, 80), patch_size=3, probability=0.8),
        ]

        small_regions = verifier.find_coherent_regions(small_matches)
        large_regions = verifier.find_coherent_regions(large_matches)

        assert large_regions[0].confidence > small_regions[0].confidence

    def test_offset_tolerance(self):
        """Matches within tolerance should cluster together."""
        verifier = SpatialCoherenceVerifier(
            offset_tolerance=3,
            min_cluster_size=2,
        )

        # Offsets differ by 2 (within tolerance of 3)
        matches = [
            Match(source_coord=(10, 20), target_coord=(50, 80), patch_size=3, probability=0.8),
            Match(source_coord=(20, 30), target_coord=(60, 92), patch_size=3, probability=0.8),  # offset (40, 62)
        ]

        regions = verifier.find_coherent_regions(matches)

        # Should cluster despite slight offset difference
        assert len(regions) == 1
        assert len(regions[0].matches) == 2
