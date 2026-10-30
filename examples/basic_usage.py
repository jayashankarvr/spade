"""Basic usage example for SPADE forensics engine."""

import numpy as np
from spade import ForensicsEngine, Config


def create_test_image(width: int, height: int, seed: int = 42) -> np.ndarray:
    """Create a test image with random texture (high entropy for patch extraction)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def main():
    # Create test images
    target = create_test_image(100, 100)

    # Extract a fragment from the target and modify it slightly
    fragment = target[30:50, 40:60].copy()
    fragment = (fragment.astype(float) * 1.1).clip(0, 255).astype(np.uint8)

    # Embed fragment in a new image (different seed for different background)
    source = create_test_image(80, 80, seed=123)
    source[20:40, 25:45] = fragment

    engine = ForensicsEngine(Config(patch_size=5))

    # Index the target
    num_patches = engine.index_target(target, "target_001")
    print(f"Indexed {num_patches} patches from target")

    # Match source against target
    result = engine.match(source)

    print(f"\nSource patches analyzed: {result.stats['source_patches']}")
    print(f"Matches found: {result.stats['total_matches']}")

    # Individual patch matches are weak evidence; regions are what count.
    for region in result.significant_regions:
        print(f"\nCopied region from {region.image_id}:")
        print(f"  Source bbox (x, y, w, h): {region.source_bbox}")
        print(f"  Target bbox (x, y, w, h): {region.target_bbox}")
        print(f"  Patches agreeing: {len(region.matches)}")
        print(f"  log10 NFA: {region.log10_nfa:.1f}  (expected false alarms: 10^{region.log10_nfa:.1f})")
    if not result.significant_regions:
        print("\nNo significant copied region found.")


if __name__ == "__main__":
    main()
