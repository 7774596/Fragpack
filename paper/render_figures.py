# SPDX-License-Identifier: Apache-2.0
"""Export manuscript figures as PDFs and 400 dpi PNGs.

Native labels and diagrams remain vector objects; UI screenshots remain raster.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess


def export_figures(version: int, numbers: list[int], *, dataflow: bool = False) -> None:
    """Compile numbered figure inputs with the manuscript's own preamble.

    ``version`` selects the preamble and numbered figure inputs; ``numbers``
    selects the one-based figure indices. ``dataflow`` additionally exports
    the native dataflow source without the paper's vertical placement scale.
    Intermediate wrappers and logs stay under paper/build/v<version>/.
    Successful exports replace the selected PDF and PNG under images/.
    This command uses only CPU LaTeX tools and never runs model code.
    """
    paper = Path(__file__).resolve().parent
    document = paper / f"fragpack_demo({version}).tex"
    text = document.read_text(encoding="utf-8")
    preamble = text.split(r"\title{", 1)[0]
    inputs = re.findall(r"\\input\{(figures/[^{}]+\.tex)\}", text)
    build = paper / "build" / f"v{version}"
    images = paper / "images"
    images.mkdir(exist_ok=True)

    exports: list[tuple[str, str, str, str]] = []
    for number in numbers:
        if number < 1 or number > len(inputs):
            raise ValueError(f"Figure {number} is outside 1..{len(inputs)}")
        exports.append(
            (
                inputs[number - 1],
                f"fig{number}",
                f"figure{number}",
                f"figure{number}({version})",
            )
        )
    if dataflow:
        exports.append(
            (
                "figures/demo_views_dataflow.tex",
                "dataflow",
                "figure3-dataflow",
                "figure3-dataflow",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        type=int,
        default=16,
        help="Manuscript preamble version (default: 16)",
    )
    parser.add_argument(
        "--figures",
        type=int,
        nargs="+",
        help="One-based figure indices (default: 1 2, unless --dataflow)",
    )
    parser.add_argument(
        "--dataflow",
        action="store_true",
        help="Export the native 180 mm dataflow figure",
    )
    args = parser.parse_args()
    numbers = (
        args.figures if args.figures is not None else ([] if args.dataflow else [1, 2])
    )
    export_figures(args.version, numbers, dataflow=args.dataflow)
