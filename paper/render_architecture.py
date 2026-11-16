# SPDX-License-Identifier: Apache-2.0
"""Export a versioned architecture SVG to vector PDF and 400 dpi PNG.

Run with Python providing PyGObject, pycairo, and the Rsvg 2.0 typelib.
Poppler's pdftoppm renders the preview from the exported PDF.
"""

import argparse
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import cairo
import gi

gi.require_version("Rsvg", "2.0")
from gi.repository import Rsvg  # noqa: E402


def export_architecture(update4: bool = False) -> None:
    """Render the canonical SVG without rasterizing its text or geometry.

    The current Update4 B SVG specifies its width in mm and aspect ratio with a
    viewBox. ``update4`` is retained for existing invocation scripts; both forms
    export the current figure. Candidate exports stay under paper/build until
    both renders pass.
    This synchronous, CPU-only export does not invoke model or GPU code.
    """
    paper = Path(__file__).resolve().parent
    source = paper / "images/fig1_architecture_refined.svg"
    target = paper / "images/figure1-architecture-refined.pdf"
    build = paper / "build/kvfold_update4_b/fig1-qa"
    build.mkdir(parents=True, exist_ok=True)
    root = ET.parse(source).getroot()
    width_mm = root.attrib["width"]
    if not width_mm.endswith("mm"):
        raise ValueError("The architecture SVG must specify its width in mm")
    _, _, view_width, view_height = map(float, root.attrib["viewBox"].split())
    width = float(width_mm[:-2]) * 72 / 25.4
    height = width * view_height / view_width

    candidate = build / "architecture-export.pdf"
    handle = Rsvg.Handle.new_from_file(str(source))
    surface = cairo.PDFSurface(str(candidate), width, height)
    surface.set_metadata(cairo.PDF_METADATA_TITLE, "KVFold integration paths")
    context = cairo.Context(surface)
    viewport = Rsvg.Rectangle()
    viewport.x = viewport.y = 0
    viewport.width, viewport.height = width, height
    if not handle.render_document(context, viewport):
        raise RuntimeError("Rsvg failed to render the architecture")
    surface.finish()
    subprocess.run(
        [
            "pdftoppm",
            "-r",
            "400",
            "-singlefile",
            "-png",
            str(candidate),
            str(candidate.with_suffix("")),
        ],
        check=True,
    )
    candidate.replace(target)
    candidate.with_suffix(".png").replace(target.with_suffix(".png"))
    print(target.relative_to(paper))
    print(target.with_suffix(".png").relative_to(paper))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update4", action="store_true", help="Export Update4 B's Figure 1"
    )
    export_architecture(update4=parser.parse_args().update4)
