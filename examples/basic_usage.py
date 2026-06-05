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

    # Configure and run SPADE
    config = Config(
        patch_size=3,
        entropy_threshold=2.0,
        min_probability=0.5,
    )
    engine = ForensicsEngine(config)

    # Index the target
    num_patches = engine.index_target(target, "target_001")
    print(f"Indexed {num_patches} patches from target")

    # Match source against target
    result = engine.match(source)

    print(f"\nSource patches analyzed: {result.stats['source_patches']}")
    print(f"Matches found: {result.stats['total_matches']}")

    if result.best_match:
        print(f"\nBest match:")
        print(f"  Probability: {result.best_match.probability:.3f}")
        print(f"  Source location: {result.best_match.source_coord}")
        print(f"  Target location: {result.best_match.target_coord}")


if __name__ == "__main__":
    main()
