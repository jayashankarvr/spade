"""A self-contained HTML page for a forensic report, for people to look at.

Built from the JSON report (``spade.report``) plus the images, so the page and
the machine-readable record can't disagree. Images are embedded as PNG data
URIs; the file can be mailed or archived on its own.
"""

from __future__ import annotations

import base64
import html
import io
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# Region outline colours, cycled. Chosen to stay distinguishable on photos.
_COLOURS = ("#e4572e", "#29b6f6", "#ffc914", "#76b041", "#c879ff", "#ff8fab")
_MAX_SIDE = 900


def _png_data_uri(image: np.ndarray) -> str:
    img = Image.fromarray(np.asarray(image))
    scale = min(1.0, _MAX_SIDE / max(img.size))
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _heatmap_rgb(heatmap: np.ndarray) -> np.ndarray:
    h = np.clip(heatmap, 0, 1)
    rgb = np.stack([np.clip(h * 3, 0, 1), np.clip(h * 3 - 1, 0, 1), np.clip(h * 3 - 2, 0, 1)], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _figure(image: np.ndarray, boxes: List[Dict[str, Any]], caption: str) -> str:
    """An image with region outlines drawn as an SVG overlay in image coordinates."""
    h, w = image.shape[:2]
    rects = "".join(
        f'<rect x="{b["x"]}" y="{b["y"]}" width="{b["width"]}" height="{b["height"]}" '
        f'stroke="{b["colour"]}" vector-effect="non-scaling-stroke"><title>{html.escape(b["label"])}</title></rect>'
        for b in boxes
    )
    return (
        f'<figure><div class="frame"><img src="{_png_data_uri(image)}" alt="{html.escape(caption)}">'
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">{rects}</svg></div>'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _fmt_transform(ct: Optional[Dict[str, Any]]) -> str:
    if not ct:
        return "&ndash;"
    if ct.get("is_identity"):
        return "none (colors unchanged)"
    rows = "<br>".join(" ".join(f"{v:+.3f}" for v in row) for row in ct["M"])
    bias = " ".join(f"{v:+.3f}" for v in ct["b"])
    return f'<span class="mono">M = <br>{rows}<br>b = {bias}</span>'


def _nfa_text(log10_nfa: Optional[float]) -> str:
    if log10_nfa is None:
        return "&ndash;"
    return f"10<sup>{log10_nfa:.1f}</sup>"


def render_html(
    report: Dict[str, Any],
    source_image: np.ndarray,
    target_image: Optional[np.ndarray] = None,
    heatmap: Optional[np.ndarray] = None,
) -> str:
    """The report as one HTML page."""
    result = report["result"]
    regions = [r for r in result.get("coherent_regions", []) if r.get("significant")]
    src_boxes, tgt_boxes, rows = [], [], []
    for i, region in enumerate(regions):
        colour = _COLOURS[i % len(_COLOURS)]
        label = f"Region {i + 1}"
        src_boxes.append({**region["source_bbox"], "colour": colour, "label": label})
        tgt_boxes.append({**region["target_bbox"], "colour": colour, "label": label})
        sb, tb = region["source_bbox"], region["target_bbox"]
        rows.append(
            f'<tr><td><span class="swatch" style="background:{colour}"></span>{i + 1}</td>'
            f"<td>{sb['x']}, {sb['y']} &middot; {sb['width']}&times;{sb['height']}</td>"
            f"<td>{tb['x']}, {tb['y']} &middot; {tb['width']}&times;{tb['height']}</td>"
            f"<td>{region['num_matches']}</td><td>{_nfa_text(region.get('log10_nfa'))}</td>"
            f"<td>{_fmt_transform(region.get('color_transform'))}</td></tr>"
        )

    if regions:
        best = min(r["log10_nfa"] for r in regions)
        verdict = (
            f'<p class="verdict hit">{len(regions)} copied region{"s" if len(regions) > 1 else ""} found. '
            f"The strongest would be expected by chance {_nfa_text(best)} times in images with no copy.</p>"
        )
    else:
        verdict = '<p class="verdict miss">No region significant enough to report.</p>'

    figures = [_figure(source_image, src_boxes, "Questioned image (source)")]
    if target_image is not None:
        figures.append(_figure(target_image, tgt_boxes, "Reference image (target)"))
    if heatmap is not None:
        figures.append(_figure(_heatmap_rgb(heatmap), [], "Match density on the reference image"))

    table = (
        "<table><thead><tr><th>#</th><th>In questioned image (x, y &middot; size)</th>"
        "<th>In reference image</th><th>Agreeing patches</th><th>Expected by chance</th>"
        "<th>Color change (reference = M &middot; questioned + b)</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        if rows else ""
    )

    cues = report.get("cues") or {}
    cue_html = ""
    si = cues.get("scale_inconsistency")
    if si:
        cue_html = (
            "<h2>Other cues</h2><p>Resize inconsistency score "
            f"<b>{si['score']:.3f}</b> (0 = uniform; the most resized-looking window is at "
            f"{si['anomaly_bbox']['x']}, {si['anomaly_bbox']['y']}).</p>"
        )

    inputs = report.get("inputs", {})
    meta_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(str(meta['path'])) if meta.get('path') else '&ndash;'}</td>"
        f"<td class='mono'>{html.escape(str(meta.get('sha256') or ''))}</td></tr>"
        for name, meta in inputs.items() if meta
    )
    config = ", ".join(f"{k}={v}" for k, v in (report.get("config") or {}).items())
    tool = report.get("tool", {})

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SPADE report</title>
<style>
:root {{ --bg: #fafaf8; --fg: #1d1d1b; --muted: #6b6b66; --line: #deded8; --hit: #b3261e; --ok: #2e7d32; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #161615; --fg: #ececea; --muted: #a1a19b; --line: #33332f; --hit: #ff8a80; --ok: #81c784; }}
}}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; margin: 0 auto;
       max-width: 1100px; padding: 24px 16px 48px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} h2 {{ font-size: 16px; margin: 28px 0 8px; }}
.sub, figcaption, .foot {{ color: var(--muted); font-size: 13px; }}
.verdict {{ font-size: 17px; padding: 12px 14px; border-left: 4px solid; background: color-mix(in srgb, var(--line) 35%, transparent); }}
.verdict.hit {{ border-color: var(--hit); }} .verdict.miss {{ border-color: var(--ok); }}
.figs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
figure {{ margin: 0; }} .frame {{ position: relative; }}
.frame img {{ display: block; width: 100%; height: auto; image-rendering: pixelated; }}
.frame svg {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
.frame rect {{ fill: none; stroke-width: 2.5; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
.mono {{ font-family: ui-monospace, monospace; font-size: 12px; word-break: break-all; }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }}
.table-wrap {{ overflow-x: auto; }}
</style></head><body>
<h1>Copied-region report</h1>
<p class="sub">{html.escape(tool.get('name', 'spade'))} {html.escape(str(tool.get('version', '')))} &middot; {html.escape(report.get('generated_at', ''))}</p>
{verdict}
<div class="figs">{''.join(figures)}</div>
{'<h2>Regions</h2><div class="table-wrap">' + table + '</div>' if table else ''}
{cue_html}
<h2>Inputs</h2><div class="table-wrap"><table>{meta_rows}</table></div>
<p class="foot">Settings: {html.escape(config)}. &ldquo;Expected by chance&rdquo; is the region's NFA: how many
regions at least this good a search would turn up in images that share nothing. Content that is identical by
design (logos, text, repeated patterns) can match for real; look before concluding.</p>
</body></html>
"""
