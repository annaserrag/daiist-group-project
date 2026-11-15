"""
Small helpers for the exploratory data analysis in every chapter, so each chapter shows its data the same way:
the figure is saved next to the chapter's other outputs (artifacts/vision/<step>/) and, in a notebook, shown
inline as well.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LINE = "=" * 78
INK = "#2a6fdb"     # one colour per role, the same in every chapter
ACCENT = "#e07b00"
MUTED = "#9aa3ab"


def in_notebook() -> bool:
    return "ipykernel" in sys.modules


def plt():
    """matplotlib, inline in a notebook and without a window when run as a script."""
    import matplotlib
    if not in_notebook():
        matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    return pyplot


def show(table: pd.DataFrame | pd.Series, title: str | None = None, digits: int = 3) -> None:
    """Print a table with a title: rendered as a table in a notebook, as text in a terminal."""
    if title:
        print(f"\n{title}")
    if in_notebook():
        from IPython.display import display
        display(table.round(digits))
    else:
        print(table.round(digits).to_string())


def save_figure(fig, path: Path) -> None:
    """Save the figure, show it in a notebook, then close it so figures do not pile up in memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    if in_notebook():
        from IPython.display import display
        display(fig)
    plt().close(fig)
    print(f"figure: {path.name}")


def save_image(image, path: Path) -> None:
    """Save a PIL image (a contact sheet of pages) and show it in a notebook."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    if in_notebook():
        from IPython.display import display
        display(image)
    print(f"image: {path.name}")


def image_sizes(paths) -> pd.DataFrame:
    """Width, height and aspect ratio of each image, read from the file header only (no decoding)."""
    from PIL import Image
    rows = []
    for p in paths:
        with Image.open(p) as image:
            rows.append({"width": image.width, "height": image.height})
    out = pd.DataFrame(rows)
    out["aspect"] = out["height"] / out["width"]
    return out


def describe(series: pd.Series) -> pd.Series:
    """count, missing, min, median, mean and max of a numeric column, the summary used throughout."""
    s = pd.to_numeric(series, errors="coerce")
    return pd.Series({"count": s.notna().sum(), "missing": s.isna().sum(), "min": s.min(), "median": s.median(),
                      "mean": s.mean(), "max": s.max()})


def histograms(data: dict[str, pd.Series], xlabel: str, path: Path, title: str, bins: int = 30, log: bool = False) -> None:
    """One small histogram per group, side by side, sharing the x axis label."""
    pyplot = plt()
    fig, axes = pyplot.subplots(1, len(data), figsize=(3.6 * len(data), 2.8), squeeze=False)
    for ax, (name, values) in zip(axes[0], data.items()):
        values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        ax.hist(values, bins=bins, color=INK, alpha=0.85, log=log)
        ax.set_title(f"{name} (n={len(values)})", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        if len(values):
            ax.axvline(values.median(), color=ACCENT, linewidth=1)
    fig.suptitle(title, fontsize=10)
    save_figure(fig, path)


def contact_sheet(images: list, labels: list[str], path: Path, thumb: int = 260, columns: int = 4) -> None:
    """A grid of page thumbnails with a caption each, to look at the data before trusting any number about it."""
    from PIL import Image, ImageDraw
    head = 18
    rows = -(-len(images) // columns)
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + head)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (image, label) in enumerate(zip(images, labels)):
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        image.thumbnail((thumb, thumb))
        x, y = (i % columns) * thumb, (i // columns) * (thumb + head)
        sheet.paste(image, (x, y + head))
        draw.text((x + 3, y + 3), label[:40], fill="black")
    save_image(sheet, path)
