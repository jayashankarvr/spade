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
from spade.exceptions import ConfigurationError


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


def validate_params(threshold: Optional[float], patch_size: Optional[int], entropy: Optional[float]) -> None:
    """Validate common CLI parameters (None means "use the Config default")."""
    if threshold is not None and not 0 <= threshold <= 1:
        raise click.BadParameter(f"threshold must be between 0 and 1, got {threshold}")
    if patch_size is not None and not (3 <= patch_size <= 16):
        raise click.BadParameter(f"patch-size must be between 3 and 16, got {patch_size}")
    if entropy is not None and entropy < 0:
        raise click.BadParameter(f"entropy must be >= 0, got {entropy}")


def build_config(threshold: Optional[float], patch_size: Optional[int], entropy: Optional[float]) -> Config:
    """Config from CLI flags; anything left as None keeps its Config/SPADE_* default."""
    overrides = {}
    if entropy is not None:
        overrides["entropy_threshold"] = entropy if entropy > 0 else None
    if threshold is not None:
        overrides["min_probability"] = threshold
    if patch_size is not None:
        overrides["patch_size"] = patch_size
    try:
        return Config(**overrides)
    except ConfigurationError as e:
        raise click.BadParameter(str(e))


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


def echo_regions(result, limit: int = 5) -> None:
    """Print the most significant regions, or say there are none."""
    significant = result.significant_regions
    if not significant:
        click.echo("\nNo significant copied region found (no region with NFA < max_nfa).")
        return
    click.echo(f"\nSignificant regions: {len(significant)}")
    for region in significant[:limit]:
        x, y, w, h = region.source_bbox
        tx, ty, tw, th = region.target_bbox
        click.echo(
            f"  {region.image_id}: source ({x},{y}) {w}x{h} -> target ({tx},{ty}) {tw}x{th}, "
            f"{len(region.matches)} patches, log10 NFA {region.log10_nfa:.1f}"
        )


THRESHOLD_HELP = "Per-patch fit p-value floor (0-1); default from config"


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.argument("target", type=click.Path(exists=True))
@click.option("-t", "--threshold", type=float, default=None, help=THRESHOLD_HELP)
@click.option("-p", "--patch-size", type=int, default=None, help="Patch size (3-16); default from config")
@click.option("-e", "--entropy", type=float, default=None, help="Entropy threshold in bits (0 disables); default from config")
@click.option("-o", "--output", type=click.Path(), help="Output heatmap image path")
@click.option("-j", "--json", "json_output", type=click.Path(), help="Write JSON forensic report to this path")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def match(
    source: str,
    target: str,
    threshold: Optional[float],
    patch_size: Optional[int],
    entropy: Optional[float],
    output: Optional[str],
    json_output: Optional[str],
    verbose: bool,
):
    """Match SOURCE image against TARGET image."""
    validate_params(threshold, patch_size, entropy)
    config = build_config(threshold, patch_size, entropy)

    if verbose:
        click.echo(f"Loading source: {source}")
    source_img = safe_load_image(source)

    if verbose:
        click.echo(f"Loading target: {target}")
    target_img = safe_load_image(target)

    engine = ForensicsEngine(config)

    if verbose:
        click.echo("Indexing target...")
    num_patches = engine.index_target(target_img, "target")
    if num_patches == 0:
        raise click.ClickException(
            "No textured patches in target. "
            "Try lowering the entropy threshold (-e 0) or using a larger image."
        )
    if verbose:
        click.echo(f"  Indexed {num_patches} patches")
        click.echo("Matching...")
    result = engine.match(source_img)

    click.echo(f"\nSource patches: {result.stats['source_patches']}")
    click.echo(f"Patch matches: {result.stats['total_matches']}")
    echo_regions(result)

    if output and result.heatmap is not None:
        heatmap_colored = colorize_heatmap(result.heatmap)
        Image.fromarray(heatmap_colored).save(output)
        click.echo(f"\nHeatmap saved to: {output}")

    if json_output:
        from spade.report import build_report, write_report

        report = build_report(
            result=result,
            config=config,
            source_path=source,
            source_image=source_img,
            target_path=target,
            target_image=target_img,
            heatmap_path=output if (output and result.heatmap is not None) else None,
        )
        write_report(report, json_output)
        click.echo(f"\nForensic report written to: {json_output}")


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("-o", "--output", default="index.spade", help="Output index path prefix")
@click.option("-p", "--pattern", default="*.jpg,*.png", help="File patterns to index")
@click.option("--patch-size", type=int, default=None, help="Patch size (3-16); default from config")
def index(directory: str, output: str, pattern: str, patch_size: Optional[int]):
    """Index all images in DIRECTORY for later searching."""
    dir_path = Path(directory)
    patterns = [p.strip() for p in pattern.split(",")]

    image_files = []
    for p in patterns:
        image_files.extend(dir_path.glob(p))
        image_files.extend(dir_path.glob(p.upper()))

    # De-duplicate: on case-insensitive filesystems the lowercase and uppercase
    # globs return the same files, which would otherwise index each image twice.
    image_files = sorted(set(image_files))

    if not image_files:
        click.echo("No images found.")
        return

    click.echo(f"Found {len(image_files)} images")

    engine = ForensicsEngine(build_config(None, patch_size, None))
    total_patches = 0

    with click.progressbar(image_files, label="Indexing") as progress:
        for img_path in progress:
            try:
                img = load_image(str(img_path))
                # The absolute path lets `spade search` reload patches for verification.
                total_patches += engine.index_target(img, str(img_path), image_path=str(img_path.resolve()))
            except Exception as e:
                click.echo(f"\nSkipped {img_path}: {e}")

    click.echo(f"\nIndexed {total_patches} patches from {engine.target_count} images")

    engine.save(output)
    click.echo(f"Index saved to: {output}")


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.argument("index_file")
@click.option("-t", "--threshold", type=float, default=None, help=THRESHOLD_HELP)
@click.option("-k", "--top-k", default=10, help="Number of regions to show")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def search(
    source: str,
    index_file: str,
    threshold: Optional[float],
    top_k: int,
    verbose: bool,
):
    """Search SOURCE image against a saved INDEX_FILE (the prefix given to `spade index -o`)."""
    validate_params(threshold, None, None)

    if verbose:
        click.echo(f"Loading index: {index_file}")
    overrides = {} if threshold is None else {"min_probability": threshold}
    try:
        engine = ForensicsEngine.load(index_file, **overrides)
    except Exception as e:
        raise click.ClickException(f"Failed to load index: {e}")

    if engine.index.size == 0:
        raise click.ClickException("Index is empty. No images have been indexed.")
    if verbose:
        click.echo(f"  {engine.index.size} descriptors from {engine.target_count} images")

    source_img = safe_load_image(source)
    if verbose:
        click.echo("Searching...")
    result = engine.match(source_img, return_heatmap=False)

    click.echo(f"\nSource patches: {result.stats['source_patches']}")
    click.echo(f"Patch matches: {result.stats['total_matches']}")
    echo_regions(result, limit=top_k)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8000, help="Port to listen on")
@click.option("--patch-size", type=int, default=None, help="Patch size (3-16); default from config")
@click.option("--pyramid/--no-pyramid", default=False, help="Enable multi-scale pyramid")
def serve(
    host: str,
    port: int,
    patch_size: Optional[int],
    pyramid: bool,
):
    """Start the REST API server."""
    try:
        from spade.api import run_server
    except ImportError:
        raise click.ClickException(
            "FastAPI required for API server: pip install spade-forensics[api]"
        )

    overrides = {"pyramid_enabled": pyramid}
    if patch_size is not None:
        overrides["patch_size"] = patch_size
    config = Config(**overrides)

    click.echo(f"Starting SPADE API server on {host}:{port}")
    click.echo(f"  Patch size: {config.patch_size}")
    click.echo(f"  Pyramid: {'enabled' if pyramid else 'disabled'}")
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
