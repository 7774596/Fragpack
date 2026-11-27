#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export the paper's editable SVG figures to vector PDFs and 400 dpi PNGs.

With no arguments all three figures are exported. To render Figure 3 alone,
run `python figures/generate_update6_figs.py 3` from the paper directory.
Install Inkscape 1.x, or use a Python with PyGObject, librsvg and Pycairo
alongside Poppler's pdftocairo. SVG sources set the physical page dimensions.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

IMAGES = Path(__file__).resolve().parent.parent / "images"
FIGURES = {
    "1": "KVFold_Fig1_Architecture",
    "2": "KVFold_Fig2_Encoding",
    "3": "fig3_console_update6",
}
DPI = 400


def find_inkscape() -> str | None:
    """Locate the Inkscape CLI, including its default Windows install."""
    if os.name == "nt":
        standard_cli = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        standard_cli = standard_cli / "Inkscape" / "bin" / "inkscape.com"
        if standard_cli.is_file():
            return str(standard_cli)
        return shutil.which("inkscape.com") or shutil.which("inkscape")
    return shutil.which("inkscape")


def render_with_librsvg(source: Path, pdf: Path) -> None:
    """Render one SVG to a PDF in points and rasterize that PDF for its preview."""
    try:
        import cairo
        import gi

        gi.require_version("Rsvg", "2.0")
        from gi.repository import Rsvg
    except (ImportError, ValueError) as exc:
        raise SystemExit(
            "Install Inkscape, or run with a Python providing PyGObject, librsvg "
            "and Pycairo (for example /usr/bin/python3 on Linux)."
        ) from exc
    if not shutil.which("pdftocairo"):
        raise SystemExit("The librsvg export requires pdftocairo from Poppler.")

    handle = Rsvg.Handle.new_from_file(str(source))
    handle.set_dpi(72)
    has_size, width, height = handle.get_intrinsic_size_in_pixels()
    if not has_size or width <= 0 or height <= 0:
        raise ValueError(f"Set a positive physical width and height on {source.name}")
    page = Rsvg.Rectangle()
    page.width, page.height = width, height
    with cairo.PDFSurface(str(pdf), width, height) as surface:
        handle.render_document(cairo.Context(surface), page)
    subprocess.run(
        [
            "pdftocairo",
            "-png",
            "-singlefile",
            "-r",
            str(DPI),
            str(pdf),
            str(pdf.with_suffix("")),
        ],
        check=True,
    )


def main() -> None:
    """Export selected figures from source, staging outputs before publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "figures", nargs="*", metavar="N", help="Figure number (1, 2 or 3)"
    )
    args = parser.parse_args()
    invalid = set(args.figures).difference(FIGURES)
    if invalid:
        parser.error(f"unknown figure: {', '.join(sorted(invalid))}; choose 1, 2 or 3")

    inkscape = find_inkscape()
    with tempfile.TemporaryDirectory(dir=IMAGES, prefix=".render-") as tmp:
        outputs = []
        for number in dict.fromkeys(args.figures or FIGURES):
            source = IMAGES / f"{FIGURES[number]}.svg"
            pdf = Path(tmp) / source.with_suffix(".pdf").name
            targets = (pdf, pdf.with_suffix(".png"))
            if inkscape:
                for target in targets:
                    subprocess.run(
                        [
                            inkscape,
                            str(source),
                            "--export-area-page",
                            f"--export-dpi={DPI}",
                            f"--export-filename={target}",
                        ],
                        check=True,
                    )
            else:
                render_with_librsvg(source, pdf)
            for target in targets:
                if not target.is_file() or not target.stat().st_size:
                    raise RuntimeError(
                        f"Missing export for {source.name}: {target.name}"
                    )
            outputs.extend(targets)
        for result in outputs:
            result.replace(IMAGES / result.name)
            print(result.name)


if __name__ == "__main__":
    main()
