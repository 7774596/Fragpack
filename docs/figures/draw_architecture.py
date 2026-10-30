# SPDX-License-Identifier: Apache-2.0
"""Render the FragPack / vLLM / LMCache architecture as editable vectors.

Rebuild with ``python docs/figures/draw_architecture.py --work-dir PATH``;
choose an external scratch directory for PATH. Exports resolve from this file.
SVG is the editable export (live text and named groups); PDF embeds fonts.
The PNG in build/ is only a visual-QA preview, not the authoring source.

Topology: docs/architecture.md and the approved integration explanation.
LMCache integration is proposed; blob export/import alone is not a connector.
Layout references (structural ideas only, no copied artwork or palette):
  Mooncake, FAST 2025, Figure 2:
  https://www.usenix.org/system/files/fast25-qin.pdf#page=3
  DistServe, OSDI 2024, Figure 6:
  https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=9
Integration references:
  https://docs.lmcache.ai/mp/architecture.html
  https://docs.vllm.ai/en/stable/api/vllm/distributed/kv_transfer/kv_connector/v1/base/
Fonts: Liberation Serif, an installed Times-compatible face. The PDF embeds
the actual font; SVG preserves the font family and editable text.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--work-dir",
    type=Path,
    required=True,
    help="External scratch directory for the preview and Matplotlib cache",
)
BUILD = parser.parse_args().work_dir.expanduser().resolve() / "build"
OUT = Path(__file__).resolve().parent
BUILD.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(BUILD / "mpl-cache"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update(
    {
        "font.family": "Liberation Serif",
        "font.size": 22,
        "svg.fonttype": "none",
        "svg.hashsalt": "fragpack-integration-architecture",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "text.color": "#172634",
    }
)

W, H = 2160, 1080
fig = plt.figure(figsize=(18, 9), dpi=120)
ax = fig.add_axes((0, 0, 1, 1))
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis("off")

INK = "#172634"
MUTED = "#52606A"
BLUE = "#315F8C"
BLUE_PALE = "#F4F8FC"
BLUE_NODE = "#E6EFF8"
ORANGE = "#B56D24"
ORANGE_PALE = "#FFF9F0"
ORANGE_NODE = "#F8E7CC"
GREEN = "#377F6A"
GREEN_PALE = "#F3F9F6"
GREEN_NODE = "#E2F0E9"
DASH = (0, (6, 4))
labels: list[tuple[object, tuple[float, float, float, float] | None]] = []


def box(
    name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    edge: str,
    fill: str = "white",
    dashed: bool = False,
    width: float = 1.6,
    radius: float = 6,
    zorder: float = 1,
) -> None:
    """Add a named semantic SVG shape using fixed, shared layout coordinates."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        edgecolor=edge,
        facecolor=fill,
        linewidth=width,
        linestyle=DASH if dashed else "solid",
        zorder=zorder,
    )
    patch.set_gid(name)
    ax.add_patch(patch)


def label(
    name: str,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 22,
    color: str = INK,
    bold: bool = False,
    align: str = "center",
    italic: bool = False,
    bounds: tuple[float, float, float, float] | None = None,
) -> None:
    """Add live SVG text and retain optional bounds for clipping checks."""
    obj = ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        fontweight="bold" if bold else "normal",
        fontstyle="italic" if italic else "normal",
        ha=align,
        va="center",
        linespacing=1.15,
        zorder=5,
    )
    obj.set_gid(name)
    labels.append((obj, bounds))


def arrow(
    name: str,
    points: list[tuple[float, float]],
    *,
    color: str = INK,
    dashed: bool = False,
    both: bool = False,
    width: float = 1.8,
) -> None:
    """Draw a routed connector with arrowheads only at semantic endpoints."""
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    obj = FancyArrowPatch(
        path=path,
        arrowstyle="<|-|>" if both else "-|>",
        mutation_scale=15,
        linewidth=width,
        linestyle=DASH if dashed else "solid",
        color=color,
        joinstyle="round",
        capstyle="round",
        zorder=3,
    )
    obj.set_gid(name)
    ax.add_patch(obj)


def tiles(name: str, x: float, y: float, *, packed: bool) -> None:
    """Draw schematic KV records without encoding a quantitative size claim."""
    width = 20 if packed else 25
    fill = "#E6BD88" if packed else "#A6C0D9"
    for row in range(2):
        for col in range(4):
            patch = Rectangle(
                (x + col * (width + 5), y + row * 25),
                width,
                20,
                edgecolor="white",
                linewidth=0.8,
                facecolor=fill,
                zorder=4,
            )
            patch.set_gid(f"{name}-{row}-{col}")
            ax.add_patch(patch)


# Ownership boundaries: FragPack is nested in the inference worker, while
# external storage is separate. Parallel backends are not a serial tier chain.
box("vllm-process", 40, 50, 1110, 875, edge=BLUE, fill=BLUE_PALE, width=1.7)
box("lmcache-service", 1670, 50, 450, 875, edge=GREEN, fill=GREEN_PALE, width=1.7)
box(
    "proposed-adapter",
    1200,
    225,
    420,
    605,
    edge=ORANGE,
    fill="#FFFCF5",
    dashed=True,
    width=1.6,
)
label(
    "vllm-heading",
    80,
    88,
    "vLLM inference process",
    size=29,
    color=BLUE,
    bold=True,
    align="left",
)
label("lmcache-heading", 1895, 88, "LMCache", size=29, color=GREEN, bold=True)
label("lmcache-subheading", 1895, 133, "External KV cache", size=22, color=MUTED)

box("scheduler", 80, 175, 405, 120, edge=BLUE, fill=BLUE_NODE)
label(
    "scheduler-title",
    282.5,
    213,
    "Request scheduler",
    size=24,
    bold=True,
    bounds=(90, 185, 385, 100),
)
label(
    "scheduler-detail",
    282.5,
    263,
    "APC · block allocation",
    size=20,
    bounds=(90, 185, 385, 100),
)
box("model-execution", 690, 175, 420, 120, edge=BLUE, fill=BLUE_NODE)
label(
    "model-title",
    900,
    213,
    "Model execution",
    size=24,
    bold=True,
    bounds=(700, 185, 400, 100),
)
label(
    "model-detail", 900, 263, "Prefill / decode", size=20, bounds=(700, 185, 400, 100)
)
arrow("schedule-model", [(485, 235), (690, 235)], color=BLUE)
label("schedule-label", 587.5, 209, "Schedule", size=19, color=MUTED)

# The scheduler owns lookup and readiness decisions. The control edge never
# starts from model execution and stays above the model node before turning.
arrow(
    "external-lookup-and-readiness",
    [(365, 175), (365, 130), (1185, 130), (1185, 350), (1710, 350)],
    color=BLUE,
    dashed=True,
    both=True,
)
label("lookup-label", 805, 108, "Prefix lookup / readiness", size=20, color=BLUE)

box(
    "fragpack-worker-plugin",
    80,
    425,
    1030,
    480,
    edge=ORANGE,
    fill=ORANGE_PALE,
    width=1.9,
)
label(
    "fragpack-heading",
    110,
    463,
    "FragPack · GPU worker plugin",
    size=27,
    color=ORANGE,
    bold=True,
    align="left",
)
arrow("block-lifecycle", [(282.5, 295), (282.5, 425)], color=BLUE)
label(
    "lifecycle-label", 304, 355, "Block lifecycle", size=20, color=MUTED, align="left"
)
arrow("attention-calls", [(900, 295), (900, 425)], color=BLUE, both=True)
label(
    "attention-label", 875, 355, "Attention calls", size=20, color=MUTED, align="right"
)

box("open-blocks", 110, 510, 255, 150, edge=BLUE, fill="white")
label("open-title", 237.5, 537, "Open blocks", size=23, bold=True)
label("open-dtype", 237.5, 572, "BF16", size=19, color=MUTED)
tiles("raw-kv-tiles", 180, 601, packed=False)

box("closed-records", 580, 510, 265, 150, edge=ORANGE, fill="white")
label("closed-title", 712.5, 537, "Closed blocks", size=23, bold=True)
label("closed-format", 712.5, 572, "Packed records", size=19, color=MUTED)
tiles("packed-kv-tiles", 665, 601, packed=True)
arrow("encode-closed-blocks", [(365, 608), (580, 608)], color=ORANGE)
label(
    "encode-label",
    472.5,
    551,
    "Encode",
    size=19,
    color=ORANGE,
    bounds=(375, 530, 195, 45),
)
label("encode-size", 472.5, 580, "128 tokens", size=18, color=MUTED)

box("blob-port", 925, 510, 155, 150, edge=ORANGE, fill="white")
label("blob-title", 1002.5, 557, "Blob I/O", size=23, bold=True)
label(
    "blob-actions",
    1002.5,
    609,
    "export\nimport",
    size=19,
    color=MUTED,
    bounds=(933, 520, 139, 130),
)
arrow("records-to-blob", [(845, 608), (925, 608)], color=ORANGE, both=True)

box("fused-decode", 345, 730, 500, 103, edge=ORANGE, fill=ORANGE_NODE)
label("decode-title", 595, 757, "Fused decode attention", size=24, bold=True)
label(
    "decode-detail",
    595,
    806,
    "On-demand KV reconstruction",
    size=19,
    bounds=(355, 740, 480, 85),
)
arrow(
    "raw-to-decode", [(237.5, 660), (237.5, 710), (445, 710), (445, 730)], color=ORANGE
)
arrow("packed-to-decode", [(712.5, 660), (712.5, 730)], color=ORANGE)
label(
    "prefill-path",
    595,
    868,
    "Prefill: BF16 workspace → FlashAttention",
    size=20,
    color=MUTED,
)

label(
    "adapter-heading-1",
    1410,
    270,
    "FragPack–LMCache",
    size=24,
    bold=True,
    color=ORANGE,
    bounds=(1210, 235, 400, 100),
)
label("adapter-heading-2", 1410, 308, "adapter (proposed)", size=22, color=ORANGE)
label(
    "connector-api",
    1410,
    410,
    "KVConnector interface",
    size=21,
    bold=True,
    bounds=(1210, 385, 400, 50),
)
label(
    "mapping", 1410, 480, "Block / chunk mapping", size=21, bounds=(1235, 455, 350, 50)
)
label(
    "identity",
    1410,
    540,
    "Format + table identity",
    size=21,
    bounds=(1235, 515, 350, 50),
)
label(
    "completion",
    1410,
    600,
    "Transfer + completion",
    size=21,
    bounds=(1235, 575, 350, 50),
)

# Only the blob port owns this proposed transport path. Attention has no
# direct edge to external storage, and blob primitives are not disk/network I/O.
arrow(
    "compressed-blob-save-restore",
    [(1002.5, 660), (1002.5, 710), (1710, 710)],
    color=ORANGE,
    dashed=True,
    both=True,
)
label("blob-transfer-label", 1410, 678, "Compressed KV blobs", size=21, color=ORANGE)
label("blob-transfer-direction", 1410, 751, "save / restore", size=20, color=MUTED)

box("cache-manager", 1710, 255, 370, 145, edge=GREEN, fill=GREEN_NODE)
label(
    "cache-manager-title",
    1895,
    293,
    "Cache manager",
    size=24,
    bold=True,
    bounds=(1720, 265, 350, 125),
)
label(
    "cache-manager-detail",
    1895,
    357,
    "Lookup · retain · evict",
    size=20,
    bounds=(1720, 265, 350, 125),
)
arrow("cache-to-storage", [(1895, 400), (1895, 545)], color=GREEN, both=True)
label(
    "storage-operation",
    1910,
    470,
    "Store / retrieve",
    size=19,
    color=MUTED,
    align="left",
)
box("storage-backends", 1710, 545, 370, 280, edge=GREEN, fill="white")
label(
    "storage-title",
    1895,
    586,
    "Storage backends",
    size=22,
    bold=True,
    bounds=(1720, 555, 350, 60),
)
for name, y, text in [
    ("dram", 622, "CPU DRAM"),
    ("ssd", 692, "NVMe SSD"),
    ("remote", 762, "Remote storage"),
]:
    box(
        f"backend-{name}",
        1745,
        y,
        300,
        52,
        edge=GREEN,
        fill=GREEN_PALE,
        width=1.2,
        radius=2,
    )
    label(
        f"backend-{name}-label",
        1895,
        y + 26,
        text,
        size=21,
        bounds=(1755, y + 4, 280, 44),
    )

# Legend carries implementation status independently of color. The qualifier
# applies to the combined path, not to the existence of LMCache itself.
arrow("existing-legend", [(85, 978), (180, 978)], color=MUTED)
label(
    "existing-legend-label",
    202,
    978,
    "Existing component capability",
    size=20,
    align="left",
)
arrow("proposed-legend", [(865, 978), (960, 978)], color=ORANGE, dashed=True, both=True)
label(
    "proposed-legend-label",
    983,
    978,
    "Proposed FragPack–LMCache integration",
    size=20,
    align="left",
)
label(
    "readiness-contract",
    1080,
    1040,
    "External KV becomes reusable only after transfer, successful import, "
    "and readiness confirmation.",
    size=20,
    color=MUTED,
    italic=True,
)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
violations: list[str] = []
for obj, bounds in labels:
    bbox = obj.get_window_extent(renderer).transformed(ax.transData.inverted())
    left, right = sorted((bbox.x0, bbox.x1))
    top, bottom = sorted((bbox.y0, bbox.y1))
    if min(left, top) < 0 or right > W or bottom > H:
        violations.append(f"{obj.get_gid()}: outside canvas")
    if bounds is not None:
        x, y, width, height = bounds
        if left < x or right > x + width or top < y or bottom > y + height:
            violations.append(f"{obj.get_gid()}: exceeds assigned text bounds")
if violations:
    raise RuntimeError("\n".join(violations))

stem = "fragpack-vllm-lmcache"
description = (
    "FragPack is nested in the vLLM GPU worker; native BF16 records become "
    "compressed closed blocks consumed by fused decode attention. The "
    "FragPack–LMCache adapter and external compressed-blob path are proposed. "
    "Vector shapes and editable SVG text; Liberation Serif font."
)
fig.savefig(
    OUT / f"{stem}.svg",
    metadata={
        "Title": "FragPack, vLLM, and LMCache",
        "Description": description,
        "Date": None,
    },
)
svg_path = OUT / f"{stem}.svg"
svg_path.write_text(
    "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
)
fig.savefig(
    OUT / f"{stem}.pdf",
    metadata={
        "Title": "FragPack, vLLM, and LMCache",
        "Subject": description,
        "Creator": "FragPack architecture renderer",
        "CreationDate": None,
        "ModDate": None,
    },
)
fig.savefig(BUILD / "preview.png", dpi=150)
plt.close(fig)

xml = ET.parse(OUT / f"{stem}.svg")
namespace = {"s": "http://www.w3.org/2000/svg"}
assert not xml.findall(".//s:image", namespace), "Raster image embedded in SVG"
texts = xml.findall(".//s:text", namespace)
assert len(texts) >= 30, "SVG text unexpectedly converted into paths"
assert "adapter (proposed)" in " ".join("".join(t.itertext()) for t in texts)
print(
    json.dumps(
        {
            "svg": str(OUT / f"{stem}.svg"),
            "pdf": str(OUT / f"{stem}.pdf"),
            "preview": str(BUILD / "preview.png"),
            "live_text_elements": len(texts),
            "raster_images": 0,
            "text_bounds": "passed",
        },
        indent=2,
    )
)
