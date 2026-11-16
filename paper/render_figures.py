# SPDX-License-Identifier: Apache-2.0
"""Export manuscript figures as PDFs, 400 dpi PNGs, and optional editable SVGs.

The encoding example retains native vector labels and diagram objects.
SVG export requires latex and dvisvgm with WOFF2 support. It retains live text
and embeds the figure's fonts instead of outlining glyphs or substituting fonts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET


def _add_svg_font_fallbacks(text: str) -> str:
    """Keep embedded TeX fonts first, with editable fallbacks for new glyphs.

    dvisvgm embeds only used glyphs. Named fallbacks support later text edits
    and SVG editors that ignore embedded webfonts, without replacing the fonts
    used to display the exported labels in a browser.
    """
    fonts = {
        "phvr8r": ("'Nimbus Sans','Liberation Sans',Helvetica,Arial,sans-serif", 400),
        "phvb8r": ("'Nimbus Sans','Liberation Sans',Helvetica,Arial,sans-serif", 700),
        "pcrr8r": ("'Nimbus Mono PS','Liberation Mono','Courier New',monospace", 400),
    }
    for family, (fallbacks, weight) in fonts.items():
        text = text.replace(
            f"@font-face{{font-family:{family};",
            f"@font-face{{font-family:{family};font-weight:{weight};",
        )
        text = re.sub(
            rf"(text\.f\d+\s*\{{font-family:){family};",
            rf"\g<1>{family},{fallbacks};font-weight:{weight};",
            text,
        )
    return text


def export_editable_svg(wrapper: Path, *, paper: Path, target: Path) -> None:
    """Render a TikZ wrapper through DVI, retaining text and embedded WOFF2 fonts.

    Use the same source and PreviewBorder as the PDF export. DVI intermediates
    and diagnostics stay beside the wrapper under ``svg/``. ``latex`` and
    ``dvisvgm`` must be on PATH; compilation and conversion run synchronously.
    The final SVG replaces ``target`` only after a live-text check succeeds.
    """
    out = wrapper.parent / "svg"
    out.mkdir(exist_ok=True)
    svg_wrapper = out / wrapper.name
    svg_wrapper.write_text(
        wrapper.read_text(encoding="utf-8").replace(
            r"\begin{document}", "\\def\\FigureSVG{1}\n\\begin{document}", 1
        ),
        encoding="utf-8",
    )
    with (out / "build-console.log").open("wb") as stream:
        for _ in range(2):
            subprocess.run(
                [
                    "latex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-file-line-error",
                    f"-output-directory={out}",
                    str(svg_wrapper),
                ],
                cwd=paper,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        svg = out / f"{wrapper.stem}.svg"
        subprocess.run(
            [
                "dvisvgm",
                "--bbox=preview",
                "--font-format=woff2",
                "--precision=6",
                f"--output={svg}",
                str(out / f"{wrapper.stem}.dvi"),
            ],
            cwd=paper,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    svg.write_text(
        _add_svg_font_fallbacks(svg.read_text(encoding="utf-8")), encoding="utf-8"
    )
    root = ET.parse(svg).getroot()
    if not root.findall(".//{http://www.w3.org/2000/svg}text"):
        raise RuntimeError(f"The SVG export contains no editable text: {svg}")
    shutil.copyfile(svg, target)


def export_figures(
    numbers: list[int],
    *,
    source: str | None = None,
    output_stem: str | None = None,
    svg: bool = False,
) -> None:
    """Compile numbered figure inputs with the manuscript's own preamble.

    The preamble comes from the current kvfold_update4_b manuscript.
    ``numbers`` must select Figure 2, whose TikZ source produces the PDF
    included by the manuscript. Intermediates stay under paper/build/kvfold_update4_b/.
    Successful exports replace the selected PDF and PNG under images/. With
    ``svg=True``, also export SVG with editable text and embedded fonts.
    This command uses only CPU LaTeX tools and never runs model code.

    For a separately named revision, provide one figure number, a paper-relative
    TikZ ``source``, and an ``output_stem``. The manuscript and its normal
    exports are then preserved; the revision gets its own build directory.
    """
    paper = Path(__file__).resolve().parent
    document = paper / "kvfold_update4_b.tex"
    text = document.read_text(encoding="utf-8")
    preamble = text.split(r"\title{", 1)[0]
    inputs = {2: "figures/record_layout_map.tex"}
    build = paper / "build" / "kvfold_update4_b"
    images = paper / "images"
    images.mkdir(exist_ok=True)
    if svg:
        for tool in ("latex", "dvisvgm"):
            if shutil.which(tool) is None:
                raise FileNotFoundError(f"SVG export requires {tool} on PATH")

    if (source is None) != (output_stem is None):
        raise ValueError("--source and --output-stem must be provided together")
    if source is not None:
        if len(numbers) != 1:
            raise ValueError("A source override requires exactly one figure number")
        if not (paper / source).is_file():
            raise FileNotFoundError(paper / source)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", output_stem):
            raise ValueError("The output stem must use letters, digits, '_' or '-'")

    exports: list[tuple[str, str, str, str]] = []
    for number in numbers:
        if number not in inputs:
            raise ValueError(
                "This exporter handles Figure 2; "
                "use the SVG/PPTX exporters for Figures 1/3"
            )
        exports.append(
            (
                source if source is not None else inputs[number],
                output_stem if output_stem is not None else f"fig{number}",
                f"figure{number}",
                output_stem
                if output_stem is not None
                else "figure2-record-map",
            )
        )
    for source, directory, stem, target_stem in exports:
        out = build / directory
        out.mkdir(parents=True, exist_ok=True)
        wrapper = out / f"{stem}.tex"
        wrapper.write_text(
            preamble
            + "\n\\usepackage[active,tightpage]{preview}\n"
            + "\\setlength\\PreviewBorder{2pt}\n"
            + "\\PreviewEnvironment{tikzpicture}\n"
            + "\\begin{document}\n\\onecolumn\n\\noindent\n"
            + f"\\input{{{source}}}\n\\end{{document}}\n",
            encoding="utf-8",
        )
        # Preserve native compiler output bytes for diagnosis if it fails.
        log = out / "build-console.log"
        with log.open("wb") as stream:
            # The first clean pass writes Hyperref's outline file; resolve it
            # on the second pass before publishing the standalone PDF.
            for _ in range(2):
                subprocess.run(
                    [
                        "pdflatex",
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "-file-line-error",
                        f"-output-directory={out}",
                        str(wrapper),
                    ],
                    cwd=paper,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        target = images / f"{target_stem}.pdf"
        shutil.copyfile(out / f"{stem}.pdf", target)
        subprocess.run(
            [
                "pdftoppm",
                "-r",
                "400",
                "-singlefile",
                "-png",
                str(target),
                str(target.with_suffix("")),
            ],
            check=True,
        )
        print(target.relative_to(paper))
        print(target.with_suffix(".png").relative_to(paper))
        if svg:
            svg_target = target.with_suffix(".svg")
            export_editable_svg(wrapper, paper=paper, target=svg_target)
            print(svg_target.relative_to(paper))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        type=int,
        nargs="+",
        default=[2],
        help="TikZ figure index (default: 2)",
    )
    parser.add_argument(
        "--source", help="Paper-relative TikZ source for one figure revision"
    )
    parser.add_argument(
        "--output-stem", help="Separate PDF/PNG/SVG filename stem under images/"
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also export SVG with editable text and embedded fonts",
    )
    args = parser.parse_args()
    export_figures(
        args.figures, source=args.source, output_stem=args.output_stem, svg=args.svg
    )
