#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import textwrap
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


PAGE_WIDTH = 8.5
PAGE_HEIGHT = 11.0
TEXT_LEFT = 0.08
TEXT_TOP = 0.95
FONT_SIZE = 10
LINES_PER_PAGE = 52
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def split_text(text: str) -> list[str]:
    wrapped_lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").split("\n"):
        line = raw_line.rstrip()
        if not line:
            wrapped_lines.append("")
            continue
        pieces = textwrap.wrap(
            line,
            width=100,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped_lines.extend(pieces or [""])
    return [
        "\n".join(wrapped_lines[start : start + LINES_PER_PAGE])
        for start in range(0, len(wrapped_lines), LINES_PER_PAGE)
    ] or [""]


def sanitize_text(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\x00", "")


def add_text(pdf: PdfPages, title: str, text: str) -> None:
    for idx, chunk in enumerate(split_text(sanitize_text(text).strip())):
        fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(
            TEXT_LEFT,
            TEXT_TOP,
            title if idx == 0 else f"{title} (cont.)",
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            family="DejaVu Sans",
            parse_math=False,
        )
        ax.text(
            TEXT_LEFT,
            TEXT_TOP - 0.04,
            chunk,
            ha="left",
            va="top",
            fontsize=FONT_SIZE,
            family="DejaVu Sans Mono",
            parse_math=False,
        )
        pdf.savefig(fig)
        plt.close(fig)


def add_image(pdf: PdfPages, title: str, image: Image.Image) -> None:
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(
        TEXT_LEFT,
        TEXT_TOP,
        title,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        family="DejaVu Sans",
        parse_math=False,
    )
    img_w, img_h = image.size
    usable_w = 0.84
    usable_h = 0.82
    scale = min(usable_w / img_w, usable_h / img_h)
    width = img_w * scale
    height = img_h * scale
    left = (1 - width) / 2
    bottom = max(0.06, 0.48 - height / 2)
    img_ax = fig.add_axes([left, bottom, width, height])
    img_ax.imshow(image)
    img_ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def notebook_markdown(notebook: dict, cell_index: int) -> str:
    source = notebook["cells"][cell_index].get("source", "")
    return "".join(source) if isinstance(source, list) else source


def notebook_output_images(notebook: dict, cell_index: int) -> list[Image.Image]:
    images = []
    for output in notebook["cells"][cell_index].get("outputs", []):
        data = output.get("data", {})
        payload = data.get("image/png") or data.get("image/jpeg")
        if not payload:
            continue
        raw = base64.b64decode("".join(payload) if isinstance(payload, list) else payload)
        images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    return images


def notebook_stream_text(notebook: dict, cell_index: int) -> str:
    chunks = []
    for output in notebook["cells"][cell_index].get("outputs", []):
        if output.get("output_type") != "stream":
            continue
        text = output.get("text", "")
        chunks.append("".join(text) if isinstance(text, list) else text)
    return sanitize_text("\n".join(chunk.strip() for chunk in chunks if chunk.strip()))


def add_file_image_or_note(pdf: PdfPages, path: Path, title: str) -> None:
    if path.exists():
        add_image(pdf, title, Image.open(path).convert("RGB"))
    else:
        add_text(pdf, title, f"Missing so far: {path.as_posix()}")


def build_part2(pdf: PdfPages, notebook: dict, results_dir: Path) -> None:
    add_text(
        pdf,
        "Part 2",
        (
            "Filtered to the README submission checklist only. "
            "Includes written responses and required figure files when present."
        ),
    )

    add_text(pdf, "Problem 1.1 and 1.2", notebook_markdown(notebook, 6))

    p1_images = notebook_output_images(notebook, 7) + notebook_output_images(notebook, 9)
    if p1_images:
        for idx, image in enumerate(p1_images, start=1):
            add_image(pdf, f"Problem 1 Figure {idx}", image)
    else:
        add_text(
            pdf,
            "Problem 1 Figures",
            "Missing so far: the four beta-schedule figures are not yet saved in the notebook outputs.",
        )

    add_text(pdf, "Problem 1 Schedule Discussion", notebook_markdown(notebook, 10))

    add_file_image_or_note(pdf, results_dir / "p2_train_plot.png", "Problem 2: p2_train_plot.png")
    add_file_image_or_note(pdf, results_dir / "p2_toy_samples.png", "Problem 2: p2_toy_samples.png")

    add_text(pdf, "Problem 3 U-Net Architecture", notebook_markdown(notebook, 38))
    add_file_image_or_note(pdf, results_dir / "mnist_train_plot.png", "Problem 3: mnist_train_plot.png")
    for weight in ["0.0", "0.5", "1.0", "2.0", "4.0"]:
        add_file_image_or_note(
            pdf,
            results_dir / f"image_w{weight}.png",
            f"Problem 3: image_w{weight}.png",
        )
    add_text(pdf, "Problem 3 CFG Discussion", notebook_markdown(notebook, 45))

    add_file_image_or_note(
        pdf,
        results_dir / "sampling_comparison.png",
        "Problem 4: sampling_comparison.png",
    )
    add_text(pdf, "Problem 4 Sampler Discussion", notebook_markdown(notebook, 57))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final Part 2 writeup PDF.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--part2-notebook", type=Path, default=Path("part2/part2_combined_colab.ipynb"))
    args = parser.parse_args()

    part2_nb = json.loads(args.part2_notebook.read_text())
    results_dir = args.part2_notebook.parent / "results"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        add_text(
            pdf,
            "PSet 5 Writeup",
            "Final Part 2 writeup only, excluding code cells.",
        )
        build_part2(pdf, part2_nb, results_dir)

    print(args.output)


if __name__ == "__main__":
    main()
