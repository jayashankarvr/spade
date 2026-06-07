"""Versioned JSON forensic report.

The report is the stable output contract for SPADE: a machine-readable,
reproducible, audit-defensible record of a match. It captures *what* was
compared (input file hashes + shapes), *how* (tool version + config), and the
*result* (localization, the recovered affine color transform, and a calibrated
chi-square probability).

Everything downstream - the benchmark harness, the CLI/API, third-party
integrations - should consume this schema rather than SPADE's in-memory objects.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

# Bump on any breaking change to the structure below.
# 1.1: added top-level "cues" (image-level forensic signals) + result.localization.
SCHEMA_VERSION = "1.1"

# Config fields worth recording for reproducibility (kept deliberately small and
# forensically relevant rather than dumping the whole Config).
_REPORTED_CONFIG_FIELDS = (
    "patch_size",
    "stride",
    "entropy_threshold",
    "descriptor_dim",
    "k_neighbors",
    "distance_threshold",
    "min_probability",
    "noise_sigma",
    "scoring_mode",
    "pyramid_enabled",
    "coherence_enabled",
)


def sha256_file(path: str) -> Optional[str]:
    """Return the SHA-256 hex digest of a file, or None if it can't be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _image_meta(path: Optional[str], image: Optional[np.ndarray]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"path": path, "sha256": sha256_file(path) if path else None}
    if image is not None:
        meta["height"] = int(image.shape[0])
        meta["width"] = int(image.shape[1])
    else:
        meta["height"] = None
        meta["width"] = None
    return meta


def _color_transform(
    M: Optional[np.ndarray], b: Optional[np.ndarray], ndigits: int = 6
) -> Optional[Dict[str, Any]]:
    """Serialize a recovered affine color transform (target = M @ source + b)."""
    if M is None or b is None:
        return None
    M = np.asarray(M, dtype=float)
    b = np.asarray(b, dtype=float)
    if M.shape != (3, 3) or b.shape != (3,):
        return None
    is_identity = bool(np.allclose(M, np.eye(3), atol=1e-3) and np.allclose(b, 0, atol=1e-3))
    return {
        "M": np.round(M, ndigits).tolist(),
        "b": np.round(b, ndigits).tolist(),
        "is_identity": is_identity,
    }


def _match_dict(match: Any) -> Optional[Dict[str, Any]]:
    if match is None:
        return None
    return {
        "source_coord": [int(match.source_coord[0]), int(match.source_coord[1])],
        "target_coord": [int(match.target_coord[0]), int(match.target_coord[1])],
        "patch_size": int(match.patch_size),
        "probability": float(match.probability),
        "image_id": match.image_id,
        "color_transform": _color_transform(
            getattr(match, "color_M", None), getattr(match, "color_b", None)
        ),
    }


def _bbox_dict(bbox: Any) -> Dict[str, int]:
    return {
        "x": int(bbox[0]),
        "y": int(bbox[1]),
        "width": int(bbox[2]),
        "height": int(bbox[3]),
    }


def _region_color_transform(region: Any) -> Optional[Dict[str, Any]]:
    """Aggregate (median) color transform across a coherent region's matches.

    A genuine recolored splice shares one consistent transform, so the median
    over the region is both robust and the forensically meaningful quantity.
    """
    Ms = [m.color_M for m in region.matches if getattr(m, "color_M", None) is not None]
    bs = [m.color_b for m in region.matches if getattr(m, "color_b", None) is not None]
    if not Ms or not bs:
        return None
    return _color_transform(np.median(np.stack(Ms), axis=0), np.median(np.stack(bs), axis=0))


def _config_dict(config: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in _REPORTED_CONFIG_FIELDS:
        if hasattr(config, name):
            out[name] = getattr(config, name)
    return out


def build_report(
    *,
    result: Any,
    config: Any,
    source_path: Optional[str] = None,
    source_image: Optional[np.ndarray] = None,
    target_path: Optional[str] = None,
    target_image: Optional[np.ndarray] = None,
    heatmap_path: Optional[str] = None,
    tool_version: Optional[str] = None,
    generated_at: Optional[str] = None,
    compute_cues: bool = True,
) -> Dict[str, Any]:
    """Build a forensic report dict from a MatchResult.

    Args:
        result: a ``MatchResult`` (matches, best_match, coherent_regions, stats).
        config: the engine ``Config`` used (a relevant subset is recorded).
        source_path/source_image, target_path/target_image: inputs (paths hashed).
        heatmap_path: path to a written heatmap artifact, if any.
        tool_version: SPADE version string (defaults to the installed version).
        generated_at: ISO-8601 timestamp (defaults to now, UTC).

    Returns:
        A JSON-serializable dict conforming to ``SCHEMA_VERSION``.
    """
    if tool_version is None:
        from spade import __version__ as tool_version
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).isoformat()

    # Localization + detection score from the spatial-density component (in
    # source/query coordinates). Computed when the source image shape is known.
    localization: Optional[Dict[str, Any]] = None
    if source_image is not None:
        from spade.aggregation.localize import localize_region

        # localize_region keeps the single largest connected component, matching
        # the single-splice positioning. A genuine multi-splice image would need
        # per-region localization here.
        coherent = [m for r in result.coherent_regions for m in r.matches]
        matches = coherent if coherent else result.matches
        loc = localize_region(matches, (int(source_image.shape[0]), int(source_image.shape[1])))
        localization = {
            "bbox": {"x": loc.bbox[0], "y": loc.bbox[1], "width": loc.bbox[2], "height": loc.bbox[3]},
            "area_px": loc.area,
            "detection_score": round(loc.area_fraction, 6),
        }

    # Image-level forensic cues, independent of the match. Computed on the source
    # (suspected tampered) image. Currently: resize/resampling inconsistency.
    # Opt-out via compute_cues=False (it does a per-window resample round trip).
    cues: Optional[Dict[str, Any]] = None
    if compute_cues and source_image is not None:
        from spade.scale import scale_inconsistency

        si = scale_inconsistency(source_image)
        cues = {
            "scale_inconsistency": {
                "score": round(si.score, 6),
                "min_native_fraction": round(si.min_fraction, 4),
                "median_native_fraction": round(si.median_fraction, 4),
                "anomaly_bbox": {
                    "x": si.anomaly_bbox[0],
                    "y": si.anomaly_bbox[1],
                    "width": si.anomaly_bbox[2],
                    "height": si.anomaly_bbox[3],
                },
            }
        }

    regions: List[Dict[str, Any]] = []
    for region in result.coherent_regions:
        regions.append(
            {
                "offset": [int(region.offset[0]), int(region.offset[1])],
                "num_matches": len(region.matches),
                "confidence": float(region.confidence),
                "source_bbox": _bbox_dict(region.source_bbox),
                "target_bbox": _bbox_dict(region.target_bbox),
                "color_transform": _region_color_transform(region),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "spade", "version": tool_version},
        "generated_at": generated_at,
        "inputs": {
            "source": _image_meta(source_path, source_image),
            "target": _image_meta(target_path, target_image),
        },
        "config": _config_dict(config),
        "result": {
            "match_found": result.best_match is not None,
            "num_matches": len(result.matches),
            "num_coherent_regions": len(result.coherent_regions),
            "localization": localization,
            "best_match": _match_dict(result.best_match),
            "coherent_regions": regions,
            "stats": dict(result.stats),
        },
        "cues": cues,
        "artifacts": {"heatmap": heatmap_path},
    }


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def to_json(report: Dict[str, Any], indent: int = 2) -> str:
    """Serialize a report dict to a JSON string."""
    return json.dumps(report, indent=indent, default=_json_default)


def write_report(report: Dict[str, Any], path: str) -> None:
    """Write a report dict to a JSON file."""
    with open(path, "w") as f:
        f.write(to_json(report))
