"""Command-line interface for SPADE."""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    click = None
    CLICK_AVAILABLE = False

from spade import __version__
from spade.engine import ForensicsEngine, Config, load_image


def main():
    """Entry point for CLI."""
    if not CLICK_AVAILABLE:
        print("click required for CLI: pip install click")
        sys.exit(1)
    cli()


@click.group()
@click.version_option(version=__version__)
def cli():
    """SPADE - Image forensics for tiny fragment detection."""
    pass


def validate_params(threshold: float, patch_size: int, entropy: float) -> None:
    """Validate common CLI parameters."""
    if not 0 <= threshold <= 1:
        raise click.BadParameter(f"threshold must be between 0 and 1, got {threshold}")
    if patch_size not in (3, 4, 5, 6):
        raise click.BadParameter(f"patch-size must be 3, 4, 5, or 6, got {patch_size}")
    if entropy < 0:
        raise click.BadParameter(f"entropy must be >= 0, got {entropy}")


def safe_load_image(path: str) -> np.ndarray:
    """Load image with helpful error messages."""
    try:
        img = load_image(path)
    except FileNotFoundError:
        raise click.ClickException(f"Image not found: {path}")
    except Exception as e:
        raise click.ClickException(f"Failed to load image {path}: {e}")

    if img.ndim != 3 or img.shape[2] != 3:
        raise click.ClickException(
            f"Image must be RGB, got shape {img.shape}. "
            "Convert grayscale images to RGB first."
        )
    if img.shape[0] < 3 or img.shape[1] < 3:
        raise click.ClickException(
            f"Image too small ({img.shape[1]}x{img.shape[0]}). "
            "Minimum size is 3x3 pixels."
        )
    return img


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.argument("target", type=click.Path(exists=True))
@click.option("-t", "--threshold", default=0.5, help="Minimum match probability (0-1)")
@click.option("-p", "--patch-size", default=3, help="Patch size (3, 4, 5, or 6)")
@click.option("-e", "--entropy", default=2.5, help="Entropy threshold (0 to disable)")
@click.option("-o", "--output", type=click.Path(), help="Output heatmap image path")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def match(
    source: str,
    target: str,
    threshold: float,
    patch_size: int,
    entropy: float,
    output: Optional[str],
    verbose: bool,
):
    """Match SOURCE image against TARGET image."""
    validate_params(threshold, patch_size, entropy)

    if verbose:
        click.echo(f"Loading source: {source}")
    source_img = safe_load_image(source)

    if verbose:
        click.echo(f"Loading target: {target}")
    target_img = safe_load_image(target)

    config = Config(
        patch_size=patch_size,
        entropy_threshold=entropy if entropy > 0 else None,
        min_probability=threshold,
    )

    engine = ForensicsEngine(config)

    if verbose:
        click.echo("Indexing target...")
    num_patches = engine.index_target(target_img, "target")
    if num_patches == 0:
        raise click.ClickException(
            "No patches extracted from target. "
            "Try lowering entropy threshold (-e 0) or using a larger image."
        )
    if verbose:
        click.echo(f"  Indexed {num_patches} patches")

    if verbose:
        click.echo("Matching...")
    result = engine.match(source_img)

    click.echo(f"\nSource patches: {result.stats['source_patches']}")
    click.echo(f"Matches found: {result.stats['total_matches']}")

    if result.best_match:
        click.echo(f"\nBest match:")
        click.echo(f"  Probability: {result.best_match.probability:.3f}")
        click.echo(f"  Location: {result.best_match.target_coord}")
    else:
        click.echo("\nNo matches found above threshold.")

    if output and result.heatmap is not None:
        heatmap_colored = colorize_heatmap(result.heatmap)
        Image.fromarray(heatmap_colored).save(output)
        click.echo(f"\nHeatmap saved to: {output}")


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("-o", "--output", default="index.spade", help="Output index file path")
@click.option("-p", "--pattern", default="*.jpg,*.png", help="File patterns to index")
def index(directory: str, output: str, pattern: str):
    """Index all images in DIRECTORY for later searching."""
    dir_path = Path(directory)
    patterns = [p.strip() for p in pattern.split(",")]

    image_files = []
    for p in patterns:
        image_files.extend(dir_path.glob(p))
        image_files.extend(dir_path.glob(p.upper()))

    if not image_files:
        click.echo("No images found.")
        return

    click.echo(f"Found {len(image_files)} images")

    engine = ForensicsEngine()
    total_patches = 0

    with click.progressbar(image_files, label="Indexing") as progress:
        for img_path in progress:
            try:
                img = load_image(str(img_path))
                total_patches += engine.index_target(img, str(img_path))
            except Exception as e:
                click.echo(f"\nSkipped {img_path}: {e}")

    click.echo(f"\nIndexed {total_patches} patches from {len(image_files)} images")

    engine.index.save(output)
    click.echo(f"Index saved to: {output}")


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.argument("index_file", type=click.Path(exists=True))
@click.option("-t", "--threshold", default=0.5, help="Minimum match probability (0-1)")
@click.option("-k", "--top-k", default=10, help="Number of top matches to show")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def search(
    source: str,
    index_file: str,
    threshold: float,
    top_k: int,
    verbose: bool,
):
    """Search SOURCE image against a saved INDEX_FILE."""
    if not 0 <= threshold <= 1:
        raise click.BadParameter(f"threshold must be between 0 and 1, got {threshold}")

    if verbose:
        click.echo(f"Loading index: {index_file}")

    engine = ForensicsEngine(Config(min_probability=threshold))
    try:
        engine.index.load(index_file)
    except Exception as e:
        raise click.ClickException(f"Failed to load index: {e}")

    if engine.index.size == 0:
        raise click.ClickException("Index is empty. No images have been indexed.")

    if verbose:
        click.echo(f"  Index contains {engine.index.size} descriptors")

    # Collect unique image paths from metadata
    image_paths = set()
    for meta in engine.index.metadata:
        if "image_id" in meta:
            image_paths.add(meta["image_id"])

    if verbose:
        click.echo(f"  From {len(image_paths)} images")

    # Try to reload images for verification
    loaded_count = 0
    for img_path in image_paths:
        if Path(img_path).exists():
            try:
                img = load_image(img_path)
                # Re-extract patches for verification
                patches = engine.extractor.extract(img)
                engine._target_patches[img_path] = patches
                engine._target_shapes[img_path] = img.shape[:2]
                loaded_count += 1
            except Exception as e:
                if verbose:
                    click.echo(f"  Could not load {img_path}: {e}")

    if loaded_count < len(image_paths):
        click.echo(f"Warning: Only {loaded_count}/{len(image_paths)} images available for verification")
        click.echo("Matches from missing images will be skipped (cannot verify without original image)")

    if verbose:
        click.echo(f"\nLoading source: {source}")
    source_img = safe_load_image(source)

    if verbose:
        click.echo("Searching...")
    result = engine.match(source_img, return_heatmap=False)

    click.echo(f"\nSource patches: {result.stats['source_patches']}")
    click.echo(f"Verified matches: {result.stats['total_matches']}")

    if result.matches:
        # Group by image and show top results
        by_image = {}
        for m in result.matches:
            if m.image_id not in by_image or m.probability > by_image[m.image_id].probability:
                by_image[m.image_id] = m

        sorted_matches = sorted(by_image.values(), key=lambda x: x.probability, reverse=True)

        click.echo(f"\nTop matches:")
        for i, m in enumerate(sorted_matches[:top_k]):
            click.echo(f"  {i+1}. {m.image_id}")
            click.echo(f"     Probability: {m.probability:.3f}, Location: {m.target_coord}")
    else:
        click.echo("\nNo matches found above threshold.")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8000, help="Port to listen on")
@click.option("--patch-size", default=3, help="Patch size (3, 4, 5, or 6)")
@click.option("--pyramid/--no-pyramid", default=False, help="Enable multi-scale pyramid")
@click.option("--lsh/--no-lsh", default=False, help="Enable LSH pre-filtering")
def serve(
    host: str,
    port: int,
    patch_size: int,
    pyramid: bool,
    lsh: bool,
):
    """Start the REST API server."""
    try:
        from spade.api import run_server
    except ImportError:
        raise click.ClickException(
            "FastAPI required for API server: pip install spade-forensics[api]"
        )

    config = Config(
        patch_size=patch_size,
        pyramid_enabled=pyramid,
        lsh_enabled=lsh,
    )

    click.echo(f"Starting SPADE API server on {host}:{port}")
    click.echo(f"  Patch size: {patch_size}")
    click.echo(f"  Pyramid: {'enabled' if pyramid else 'disabled'}")
    click.echo(f"  LSH: {'enabled' if lsh else 'disabled'}")
    click.echo(f"\nAPI documentation: http://localhost:{port}/docs\n")

    run_server(host=host, port=port, config=config)


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Apply hot colormap to heatmap for visualization."""
    h = np.clip(heatmap, 0, 1)

    # Simple hot colormap: black -> red -> yellow -> white
    r = np.clip(h * 3, 0, 1)
    g = np.clip(h * 3 - 1, 0, 1)
    b = np.clip(h * 3 - 2, 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


if __name__ == "__main__":
    main()
