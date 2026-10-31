"""Which images in a collection share copied regions with which.

Every image is searched against all the others; regions are kept under a
false discovery rate across the whole collection (``nfa.fdr_select``), and
each pair of images becomes one link carrying its regions. Optionally each
image is also searched against itself for copy-moves.

Links are undirected. When A and B share a region, the pixels alone don't say
which was the original; ordering needs outside evidence (capture times, file
history, which version is larger or less compressed), so we don't guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from spade.verification.coherence import CoherentRegion
from spade.verification.nfa import fdr_select


@dataclass
class Link:
    """Regions shared by two images (or duplicated within one, when a == b)."""
    a: str
    b: str
    regions: List[CoherentRegion] = field(default_factory=list)  # source side in a, target side in b

    @property
    def log10_nfa(self) -> float:
        return min(r.log10_nfa for r in self.regions)


def build_provenance(
    engine,
    images: Dict[str, Tuple[np.ndarray, float]],
    fdr: float = 0.05,
    copy_moves: bool = False,
) -> List[Link]:
    """Links between images that share regions.

    Args:
        engine: a ``ForensicsEngine`` with every image in ``images`` indexed
            under the same id.
        images: image id -> (RGB image, JPEG rounding-error sigma).
        fdr: false discovery rate across all reported regions.
        copy_moves: also look for regions duplicated within each image.

    Returns:
        Links, most significant first.
    """
    results, owners = [], {}
    ids = list(images)
    for image_id in ids:
        image, sigma = images[image_id]
        others = [other for other in ids if other != image_id]
        searches = [engine.match(image, target_ids=others, return_heatmap=False, jpeg_sigma=sigma)] if others else []
        if copy_moves:
            searches.append(engine.match_self(image, jpeg_sigma=sigma))
        for result in searches:
            results.append(result)
            for region in result.coherent_regions:
                owners[id(region)] = image_id

    links: Dict[Tuple[str, str], Link] = {}
    for region in fdr_select(results, q=fdr):
        a = owners[id(region)]
        b = region.image_id if region.image_id in images else a  # match_self names its target "self"
        key = tuple(sorted((a, b)))
        link = links.get(key)
        # Both directions of a pair usually find the same region; keep one per pair direction.
        if link is None:
            links[key] = Link(a=a, b=b, regions=[region])
        elif (link.a, link.b) == (a, b):
            link.regions.append(region)
        elif region.log10_nfa < link.log10_nfa:
            links[key] = Link(a=a, b=b, regions=[region])
    return sorted(links.values(), key=lambda link: link.log10_nfa)


def to_dot(links: Iterable[Link], names: Optional[Dict[str, str]] = None) -> str:
    """Graphviz text for the links (``dot -Tsvg``); edge labels give region count and best NFA."""
    names = names or {}
    lines = ["graph provenance {", "  node [shape=box, fontname=Helvetica];"]
    for link in links:
        label = f"{len(link.regions)} region{'s' if len(link.regions) > 1 else ''}, NFA 1e{link.log10_nfa:.0f}"
        a, b = (names.get(x, x).replace('"', "'") for x in (link.a, link.b))
        lines.append(f'  "{a}" -- "{b}" [label="{label}"];')
    lines.append("}")
    return "\n".join(lines)


def to_dict(links: Iterable[Link]) -> List[dict]:
    """JSON-ready links with every region's boxes and significance."""
    out = []
    for link in links:
        out.append({
            "a": link.a,
            "b": link.b,
            "log10_nfa": round(link.log10_nfa, 4),
            "regions": [
                {
                    "bbox_in_a": list(map(int, r.source_bbox)),
                    "bbox_in_b": list(map(int, r.target_bbox)),
                    "num_matches": len(r.matches),
                    "log10_nfa": round(float(r.log10_nfa), 4),
                    "mirrored": bool(r.mirrored),
                }
                for r in link.regions
            ],
        })
    return out
