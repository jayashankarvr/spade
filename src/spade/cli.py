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
from spade.jpeg import quantization_sigma


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


THOROUGH_HELP = ("Also check mirrored, enlarged and rotated copies, not just copied, recolored and "
                 "shrunk ones (rotation across images only, not within one image); about 13x slower")


def build_config(threshold: Optional[float], patch_size: Optional[int], entropy: Optional[float],
                 thorough: bool = False) -> Config:
    """Config from CLI flags; anything left as None keeps its Config/SPADE_* default."""
    overrides = {"thorough": True} if thorough else {}
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
@click.option("--html", "html_output", type=click.Path(), help="Write a visual HTML report to this path")
@click.option("--thorough", is_flag=True, help=THOROUGH_HELP)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def match(
    source: str,
    target: str,
    threshold: Optional[float],
    patch_size: Optional[int],
    entropy: Optional[float],
    output: Optional[str],
    json_output: Optional[str],
    html_output: Optional[str],
    thorough: bool,
    verbose: bool,
):
    """Match SOURCE image against TARGET image."""
    validate_params(threshold, patch_size, entropy)
    config = build_config(threshold, patch_size, entropy, thorough)

    if verbose:
        click.echo(f"Loading source: {source}")
    source_img = safe_load_image(source)

    if verbose:
        click.echo(f"Loading target: {target}")
    target_img = safe_load_image(target)

    engine = ForensicsEngine(config)

    if verbose:
        click.echo("Indexing target...")
    num_patches = engine.index_target(target_img, "target", jpeg_sigma=quantization_sigma(target) or 0.0)
    if num_patches == 0:
        raise click.ClickException(
            "No textured patches in target. "
            "Try lowering the entropy threshold (-e 0) or using a larger image."
        )
    if verbose:
        click.echo(f"  Indexed {num_patches} patches")
        click.echo("Matching...")
    result = engine.match(source_img, jpeg_sigma=quantization_sigma(source) or 0.0)

    click.echo(f"\nSource patches: {result.stats['source_patches']}")
    click.echo(f"Patch matches: {result.stats['total_matches']}")
    echo_regions(result)

    if output and result.heatmap is not None:
        heatmap_colored = colorize_heatmap(result.heatmap)
        Image.fromarray(heatmap_colored).save(output)
        click.echo(f"\nHeatmap saved to: {output}")

    if json_output or html_output:
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
        if json_output:
            write_report(report, json_output)
            click.echo(f"\nForensic report written to: {json_output}")
        if html_output:
            from spade.visual import render_html

            with open(html_output, "w") as f:
                f.write(render_html(report, source_img, target_img, result.heatmap))
            click.echo(f"\nVisual report written to: {html_output}")


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("-o", "--output", default="index.spade", help="Output index path prefix")
@click.option("-p", "--pattern", default="*.jpg,*.png", help="File patterns to index")
@click.option("--patch-size", type=int, default=None, help="Patch size (3-16); default from config")
@click.option("--thorough", is_flag=True, help=THOROUGH_HELP + " (decided when indexing; search and scan follow the index)")
def index(directory: str, output: str, pattern: str, patch_size: Optional[int], thorough: bool):
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

    engine = ForensicsEngine(build_config(None, patch_size, None, thorough))
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
    result = engine.match(source_img, return_heatmap=False, jpeg_sigma=quantization_sigma(source) or 0.0)

    click.echo(f"\nSource patches: {result.stats['source_patches']}")
    click.echo(f"Patch matches: {result.stats['total_matches']}")
    echo_regions(result, limit=top_k)


@cli.command()
@click.argument("index_file")
@click.argument("sources", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--fdr", default=0.05, show_default=True, help="False discovery rate across all reported regions")
def scan(index_file: str, sources, fdr: float):
    """Search many SOURCES against INDEX_FILE, keeping the false discovery rate at --fdr.

    NFA < 1 bounds chance regions per search; across many searches, this
    bounds the share of reported regions that are wrong instead.
    """
    from spade.verification.nfa import fdr_select

    if not 0 < fdr < 1:
        raise click.BadParameter(f"fdr must be in (0, 1), got {fdr}")
    try:
        engine = ForensicsEngine.load(index_file)
    except Exception as e:
        raise click.ClickException(f"Failed to load index: {e}")

    results = {}
    with click.progressbar(sources, label="Searching") as progress:
        for path in progress:
            results[path] = engine.match(
                safe_load_image(path), return_heatmap=False, jpeg_sigma=quantization_sigma(path) or 0.0
            )

    selected = fdr_select(list(results.values()), q=fdr)
    owner = {id(r): path for path, res in results.items() for r in res.coherent_regions}
    click.echo(f"\n{len(selected)} region(s) at FDR {fdr} across {len(sources)} image(s):")
    for region in selected:
        x, y, w, h = region.source_bbox
        click.echo(
            f"  {owner[id(region)]}: ({x},{y}) {w}x{h} <- {region.image_id}, "
            f"{len(region.matches)} patches, log10 NFA {region.log10_nfa:.1f}"
        )


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("-p", "--pattern", default="*.jpg,*.png", help="File patterns to include")
@click.option("--fdr", default=0.05, show_default=True, help="False discovery rate across all reported regions")
@click.option("--copy-moves", is_flag=True, help="Also look for regions duplicated within each image")
@click.option("--json", "json_output", type=click.Path(), help="Write the links as JSON")
@click.option("--dot", "dot_output", type=click.Path(), help="Write a Graphviz graph of the links")
@click.option("--thorough", is_flag=True, help=THOROUGH_HELP)
def provenance(directory: str, pattern: str, fdr: float, copy_moves: bool, json_output, dot_output, thorough: bool):
    """Find which images in DIRECTORY share copied regions with which.

    Links are undirected: the pixels don't say which image was the original.
    """
    import json

    from spade.provenance import build_provenance, to_dict, to_dot

    paths = sorted({p for pat in pattern.split(",") for p in Path(directory).glob(pat.strip())})
    if len(paths) < 2 and not copy_moves:
        raise click.ClickException("Need at least two images (or --copy-moves).")
    engine = ForensicsEngine(build_config(None, None, None, thorough))
    images = {}
    with click.progressbar(paths, label="Indexing") as progress:
        for path in progress:
            image = safe_load_image(str(path))
            sigma = quantization_sigma(str(path)) or 0.0
            engine.index_target(image, path.name, jpeg_sigma=sigma)
            images[path.name] = (image, sigma)
    click.echo("Searching every image against the others...")
    links = build_provenance(engine, images, fdr=fdr, copy_moves=copy_moves)

    click.echo(f"\n{len(links)} link(s) at FDR {fdr} among {len(paths)} image(s):")
    for link in links:
        what = "duplicated within" if link.a == link.b else "shared by"
        pair = link.a if link.a == link.b else f"{link.a} <-> {link.b}"
        click.echo(f"  {len(link.regions)} region(s) {what} {pair}, best log10 NFA {link.log10_nfa:.1f}")
    if json_output:
        with open(json_output, "w") as f:
            json.dump(to_dict(links), f, indent=2)
    if dot_output:
        with open(dot_output, "w") as f:
            f.write(to_dot(links))


@cli.command("hash")
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("-a", "--algorithm", "algorithms", multiple=True,
              help="Hash algorithm: pdq, phash, phash-lib, or all (every one available here). "
                   "Repeat for several. Default: all. photodna (licensees only, needs "
                   "SPADE_PHOTODNA_KEY) is used only when named")
@click.option("--against", "hash_list", type=click.Path(exists=True, dir_okay=False),
              help="CSV list of known hashes (algorithm,hash,id) to match against")
@click.option("--max-distance", type=click.IntRange(min=0), default=None,
              help="Largest Hamming distance that counts as a match; default depends on the algorithm")
@click.option("--min-quality", type=click.IntRange(0, 100), default=50, show_default=True,
              help="Skip matching PDQ hashes of images with a lower quality score")
@click.option("--thorough", is_flag=True,
              help="Also hash the eight rotations and flips of each image, to match rotated or flipped copies")
@click.option("--json", "json_output", type=click.Path(), help="Write the hashes and matches as JSON")
def hash_command(images, algorithms, hash_list, max_distance, min_quality, thorough, json_output):
    """Print perceptual hashes of IMAGES and optionally match them against a list."""
    import json

    from spade.exceptions import DependencyError
    from spade.hashing import available_algorithms, hash_image, load_hash_list, match_hashes

    local = available_algorithms()
    names = []
    use_photodna = False
    for name in algorithms or ("all",):
        name = name.lower()
        if name == "photodna":
            # Only ever on when asked for by name; "all" skips it.
            use_photodna = True
            continue
        for n in (local if name == "all" else [name]):
            if n not in names:
                names.append(n)
    # An unavailable phash-lib is reported by hash_image with install advice.
    known = set(local) | {"phash-lib"}
    for n in names:
        if n not in known:
            raise click.ClickException(f"Unknown algorithm '{n}'; choose from: {', '.join(sorted(known))}, all")

    try:
        known_hashes = load_hash_list(hash_list) if hash_list else None
    except (ValueError, OSError) as e:
        raise click.ClickException(f"Can't read hash list: {e}")

    photodna_client = None
    if use_photodna:
        from spade import photodna
        try:
            photodna_client = photodna.PhotoDnaClient()
        except photodna.PhotoDnaError as e:
            raise click.ClickException(str(e))

    report = []
    for path in images:
        hashes = []
        if names:
            image = safe_load_image(path)
            try:
                hashes = hash_image(image, names, dihedral=thorough)
            except (DependencyError, ValueError) as e:
                raise click.ClickException(str(e))
        verdict = None
        if photodna_client is not None:
            click.echo(f"Sending {path} to the PhotoDNA Cloud Service", err=True)
            try:
                with open(path, "rb") as f:
                    verdict = photodna_client.match(f.read())
            except OSError as e:
                raise click.ClickException(f"Can't read {path}: {e}")
            except photodna.PhotoDnaError as e:
                raise click.ClickException(str(e))
        result = (
            match_hashes(hashes, known_hashes, max_distance=max_distance, min_quality=min_quality)
            if known_hashes is not None else None
        )
        report.append((path, hashes, result, verdict))

        for h in hashes:
            line = f"{path} {h.algorithm} {h.hex}"
            if h.quality is not None:
                line += f" quality={h.quality}"
            if thorough:
                line += f" variant={h.variant}"
            click.echo(line)
        if verdict is not None:
            click.echo(f"{path} photodna {json.dumps(verdict, sort_keys=True)}")
        if result is not None:
            for m in result.matches:
                click.echo(f"  match {m.id} ({m.algorithm}) distance={m.distance} variant={m.variant}")
            if not result.matches:
                click.echo("  no match")
            seen = {(h.algorithm, h.quality) for h in result.skipped_low_quality}
            for algo, quality in sorted(seen):
                click.echo(f"  skipped {algo} hash: quality {quality} is below {min_quality}")

    if json_output:
        def hash_dict(h):
            return {"algorithm": h.algorithm, "hex": h.hex, "quality": h.quality, "variant": h.variant}

        doc = {"images": [
            {
                "path": path,
                "hashes": [hash_dict(h) for h in hashes],
                "matches": [
                    {"id": m.id, "algorithm": m.algorithm, "hex": m.hex,
                     "distance": m.distance, "variant": m.variant}
                    for m in (result.matches if result else [])
                ],
                "skipped_low_quality": [hash_dict(h) for h in (result.skipped_low_quality if result else [])],
            }
            for path, hashes, result, verdict in report
        ]}
        for entry, (_, _, _, verdict) in zip(doc["images"], report):
            if verdict is not None:
                entry["photodna"] = verdict
        with open(json_output, "w") as f:
            json.dump(doc, f, indent=2)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8000, help="Port to listen on")
@click.option("--patch-size", type=int, default=None, help="Patch size (3-16); default from config")
@click.option("--pyramid/--no-pyramid", default=None, help="Index targets at several scales (default: on)")
@click.option("--thorough", is_flag=True, help=THOROUGH_HELP)
def serve(
    host: str,
    port: int,
    patch_size: Optional[int],
    pyramid: Optional[bool],
    thorough: bool,
):
    """Start the REST API server."""
    try:
        from spade.api import run_server
    except ImportError:
        raise click.ClickException(
            "FastAPI required for API server: pip install spade-forensics[api]"
        )

    overrides = {"thorough": True} if thorough else {}
    if pyramid is not None:
        overrides["pyramid_enabled"] = pyramid
    if patch_size is not None:
        overrides["patch_size"] = patch_size
    config = Config(**overrides)

    click.echo(f"Starting SPADE API server on {host}:{port}")
    click.echo(f"  Patch size: {config.patch_size}")
    click.echo(f"  Pyramid: {'enabled' if config.pyramid_enabled else 'disabled'}")
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
