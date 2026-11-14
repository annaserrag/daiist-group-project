"""
The two public handwriting datasets the detectors are trained on, and why we split them the way we do.

Both come from Roboflow Universe and are used exactly as published (no fork), at a pinned version:

    answer sheets  answersheet/answer-sheet-component-detection v1   CC BY 4.0   -> question_no detector
    text lines     nucularvision/handwritten-text-line-dataset v1    CC BY 4.0   -> text_line detector

Nothing from the Mendeley exams is used here: all 50 of them are kept for the final product evaluation.

The leakage problem, in short. Roboflow exported every source photo several times (augmented copies, and
for the lines set also 2x2 tiles), and many different photos come from the same student's booklet or the
same photo session. A split by image would put copies of one page, or pages in one person's handwriting, on
both sides of the split, and the validation score would flatter the model. So every image is mapped back to
its source photo and to its group (booklet or session), and the split is made by group. The functions below
also produce the evidence for that decision (`similarity_evidence`), which train_detector.py prints.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# Downloads, built datasets and training runs live outside the repo: they are large, and deploy.py uploads
# everything inside the repo to the Space. Set VISION_WORK to put them somewhere else.
WORK = Path(os.environ.get("VISION_WORK", Path.home() / ".cache" / "daiist-vision"))
DATASETS = WORK / "datasets"

# workspace, project, version on Roboflow Universe
SOURCES = {
    "answersheet": ("answersheet", "answer-sheet-component-detection", 1),
    "lines": ("nucularvision", "handwritten-text-line-dataset", 1),
}

# The answer-sheet set has no validation or test split, so we hold out whole booklets, chosen by subject
# before looking at any result. Split 5/2/1: question numbers are scarce (about 280 in all 8 booklets), so
# as many booklets as possible go to train. Validation keeps two booklets (Science and Social Science)
# because it chooses between the grid runs; test keeps one (Social Science, 36 question numbers), which
# only checks the detector on unseen booklets. The real test is the Mendeley exams.
VAL_BOOKLETS = {"31-3-3-Science-Topper-Solution", "Social_Science_087"}
TEST_BOOKLETS = {"32-5-1-Social-Science-Topper-Solution"}

# All 22 WhatsApp photos were taken on 2024-04-05 between 13:41 and 14:04: one writer, one notebook.
WHATSAPP_SESSION = "WhatsApp-2024-04-05-session"


# --------------------------------------------------------------------------- download and index

def download(name: str) -> Path:
    """Download one dataset in YOLO format (once) and return its folder."""
    workspace, project, version = SOURCES[name]
    target = DATASETS / f"{project}-{version}"
    if (target / "data.yaml").is_file():
        return target
    from roboflow import Roboflow  # only needed for the download

    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise SystemExit("ROBOFLOW_API_KEY is not set. Add it to .env (see .env.example).")
    DATASETS.mkdir(parents=True, exist_ok=True)
    Roboflow(api_key=key).workspace(workspace).project(project).version(version).download(
        "yolov11", location=str(target), overwrite=True
    )
    return target


def source_of(file_name: str) -> str:
    """Roboflow names exported copies `<original>.rf.<hash>.jpg`; the part before `.rf.` is the source photo."""
    return file_name.split(".rf.")[0]


def group_of(dataset: str, source: str) -> str:
    """The unit that must not be split: a student's booklet, or a photo session."""
    if dataset == "answersheet":
        booklet = re.sub(r"[-_ ]?_?page_\d+.*$", "", source)
        return re.sub(r"_\d+_png$|_png$", "", booklet)
    if source.startswith("WhatsApp"):
        return WHATSAPP_SESSION
    return source


def index_images(name: str, folder: Path) -> pd.DataFrame:
    """One row per exported image: where it is, its Roboflow split, its source photo and its group."""
    rows = []
    for split in ("train", "valid", "test"):
        image_dir = folder / split / "images"
        if not image_dir.is_dir():
            continue
        for path in sorted(image_dir.iterdir()):
            source = source_of(path.name)
            label = folder / split / "labels" / f"{path.stem}.txt"
            rows.append(dict(
                dataset=name, roboflow_split=split, file=path.name, path=str(path), label=str(label),
                source=source, group=group_of(name, source),
                n_boxes=sum(1 for line in label.read_text().splitlines() if line.strip()) if label.is_file() else 0,
            ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- evidence for the split

def embed(paths: list[str], cache: Path) -> np.ndarray:
    """L2-normalised ResNet-18 (ImageNet) features. They still match a page after it has been cropped,
    rotated or brightened, which is what we need to find augmented copies and near-duplicates."""
    if cache.is_file():
        stored = np.load(cache, allow_pickle=True)
        if list(stored["paths"]) == paths:
            return stored["vectors"]
    import torch
    import torchvision
    from PIL import Image

    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    transform = weights.transforms()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(paths), 64):
            batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[start:start + 64]])
            vectors.append(torch.nn.functional.normalize(model(batch), dim=1).numpy())
    vectors = np.concatenate(vectors)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, paths=np.array(paths, dtype=object), vectors=vectors)
    return vectors


def similarity_evidence(index: pd.DataFrame, vectors: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compares every pair of images in a dataset and labels the pair:

        copy of the same photo   the same source exported twice (augmentation or tiling)
        same group               different photos from the same booklet or session
        different group          everything else

    If "same group" pairs are as similar as "copy" pairs, the group is what has to stay on one side of the
    split. Returns a summary table and, for each dataset, the closest pairs of different photos that
    Roboflow's own split put on different sides.
    """
    summary, crossing = [], []
    for name, part in index.groupby("dataset", sort=False):
        idx = part.index.to_numpy()
        sims = vectors[idx] @ vectors[idx].T
        source = part["source"].to_numpy()
        group = part["group"].to_numpy()
        split = part["roboflow_split"].to_numpy()
        upper = np.triu(np.ones_like(sims, dtype=bool), k=1)
        same_source = (source[:, None] == source[None, :]) & upper
        same_group = (group[:, None] == group[None, :]) & ~(source[:, None] == source[None, :]) & upper
        other = (group[:, None] != group[None, :]) & upper
        for label, mask in (("copy of the same photo", same_source), ("same group, different photo", same_group),
                            ("different group", other)):
            values = sims[mask]
            summary.append(dict(dataset=name, pair=label, pairs=int(mask.sum()),
                                median=float(np.median(values)) if values.size else np.nan,
                                p95=float(np.percentile(values, 95)) if values.size else np.nan))
        cross = (split[:, None] != split[None, :]) & ~(source[:, None] == source[None, :]) & upper
        rows, cols = np.where(cross)
        order = np.argsort(-sims[rows, cols])
        seen = set()
        for r, c in zip(rows[order], cols[order]):
            pair = tuple(sorted((source[r], source[c])))
            if pair in seen:
                continue
            seen.add(pair)
            crossing.append(dict(dataset=name, similarity=float(sims[r, c]),
                                 a=source[r], a_split=split[r], a_group=group[r], a_path=part.iloc[r]["path"],
                                 b=source[c], b_split=split[c], b_group=group[c], b_path=part.iloc[c]["path"],
                                 same_group=group[r] == group[c]))
            if len(seen) >= 10:
                break
    return pd.DataFrame(summary), pd.DataFrame(crossing)


def plot_similarity(index: pd.DataFrame, vectors: np.ndarray, out: Path) -> None:
    """Histogram of pair similarity per pair type, one panel per dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(index["dataset"].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 3.6), squeeze=False)
    colours = {"copy of the same photo": "#2a6fdb", "same group, different photo": "#e07b00", "different group": "#9aa3ab"}
    for ax, name in zip(axes[0], names):
        part = index[index["dataset"] == name]
        idx = part.index.to_numpy()
        sims = vectors[idx] @ vectors[idx].T
        source, group = part["source"].to_numpy(), part["group"].to_numpy()
        upper = np.triu(np.ones_like(sims, dtype=bool), k=1)
        masks = {
            "copy of the same photo": (source[:, None] == source[None, :]) & upper,
            "same group, different photo": (group[:, None] == group[None, :]) & (source[:, None] != source[None, :]) & upper,
            "different group": (group[:, None] != group[None, :]) & upper,
        }
        for label, mask in masks.items():
            if mask.any():
                ax.hist(sims[mask], bins=60, range=(0.3, 1.0), density=True, alpha=0.55, color=colours[label], label=label)
        ax.set_title(f"{name}: how alike are two images?")
        ax.set_xlabel("cosine similarity of ResNet-18 features")
        ax.set_yticks([])
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def contact_sheet(crossing: pd.DataFrame, out: Path, per_dataset: int = 3) -> None:
    """The closest pairs of different photos that Roboflow put in different splits, side by side."""
    from PIL import Image, ImageDraw

    picks = crossing.groupby("dataset", sort=False).head(per_dataset)
    size, pad, head = 300, 16, 22
    sheet = Image.new("RGB", (2 * size + pad, len(picks) * (size + head + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_i, (_, row) in enumerate(picks.iterrows()):
        y = row_i * (size + head + pad)
        for col, side in enumerate(("a", "b")):
            image = Image.open(row[f"{side}_path"]).convert("RGB").resize((size, size))
            sheet.paste(image, (col * (size + pad), y + head))
            draw.text((col * (size + pad), y + 4),
                      f"{row['dataset']} {row[f'{side}_split']}: {row[side][:30]}  sim {row['similarity']:.3f}", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


# --------------------------------------------------------------------------- the split

def make_split(index: pd.DataFrame) -> pd.DataFrame:
    """
    Our split, by group:

    - answer sheets: Roboflow put every image in train. We hold out whole booklets for validation
      (VAL_BOOKLETS) and for test (TEST_BOOKLETS), and there keep one copy of each page, so a page is not
      counted three times.
    - lines: Roboflow's split is by photo, which is fine except for the WhatsApp session, which is spread
      over train, valid and test. The whole session goes to train; the rest keeps Roboflow's split.

    Validation is for choosing (baselines, grid, thresholds), test is used once on the chosen models, like in
    Assignment 2. The Mendeley exams are a separate, later evaluation of the whole product.
    """
    out = index.copy()
    answers = out["dataset"] == "answersheet"
    out.loc[answers, "our_split"] = np.select(
        [out.loc[answers, "group"].isin(VAL_BOOKLETS), out.loc[answers, "group"].isin(TEST_BOOKLETS)],
        ["val", "test"], "train")
    lines = ~answers
    rf = out.loc[lines, "roboflow_split"].map({"train": "train", "valid": "val", "test": "test"})
    out.loc[lines, "our_split"] = np.where(out.loc[lines, "group"].eq(WHATSAPP_SESSION), "train", rf)
    # In val and test, keep one copy of each answer-sheet page: its copies are augmentations of the same page.
    # The lines set's copies are 4 tiles of a photo, i.e. different parts of it, so all stay.
    first_copy = ~out.duplicated(subset=["dataset", "source", "our_split"], keep="first")
    out["used"] = (out["our_split"] == "train") | first_copy | lines
    return out


def leakage_after_split(split: pd.DataFrame) -> pd.DataFrame:
    """Proof that the split holds: sources and groups that end up on both sides (should be none)."""
    rows = []
    for name, part in split[split["used"]].groupby("dataset", sort=False):
        for key in ("source", "group"):
            sides = part.groupby(key)["our_split"].nunique()
            rows.append(dict(dataset=name, unit=key, units=len(sides), in_more_than_one_split=int((sides > 1).sum())))
    return pd.DataFrame(rows)


def write_yolo(split: pd.DataFrame, dataset: str, class_map: dict[int, int], names: list[str], out: Path) -> Path:
    """Build a YOLO dataset folder from our split, keeping only the classes in class_map (renumbered)."""
    if out.exists():
        shutil.rmtree(out)
    for _, row in split[(split["dataset"] == dataset) & split["used"]].iterrows():
        image_dir = out / row["our_split"] / "images"
        label_dir = out / row["our_split"] / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], image_dir / row["file"])
        lines = []
        label = Path(row["label"])
        if label.is_file():
            for line in label.read_text().splitlines():
                parts = line.split()
                if parts and int(parts[0]) in class_map:
                    coords = parts[1:]
                    if len(coords) > 4:  # a polygon row (some WhatsApp labels): use its enclosing box,
                        # otherwise Ultralytics drops any image whose file mixes polygons and boxes
                        xs, ys = list(map(float, coords[0::2])), list(map(float, coords[1::2]))
                        coords = [f"{(min(xs) + max(xs)) / 2:.6f}", f"{(min(ys) + max(ys)) / 2:.6f}",
                                  f"{max(xs) - min(xs):.6f}", f"{max(ys) - min(ys):.6f}"]
                    lines.append(" ".join([str(class_map[int(parts[0])])] + coords))
        (label_dir / f"{Path(row['file']).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    yaml = out / "data.yaml"
    yaml.write_text(f"path: {out}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n"
                    + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    return yaml


def label_sheet(folder: Path, out: Path, split: str = "train", count: int = 6, thumb: int = 320) -> None:
    """A few built images with their labels drawn back on, the check from the Session 9 notebook:
    it catches swapped coordinates, wrong class maps and unconverted polygons before training."""
    from PIL import Image, ImageDraw

    paths = sorted((folder / split / "images").iterdir())
    paths = paths[:: max(len(paths) // count, 1)][:count]  # spread over the split, not the first booklet only
    columns = 3
    sheet = Image.new("RGB", (columns * thumb, -(-len(paths) // columns) * thumb), "black")
    for i, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        w, h = image.size
        draw = ImageDraw.Draw(image)
        label = folder / split / "labels" / f"{path.stem}.txt"
        for line in label.read_text().splitlines() if label.is_file() else []:
            cx, cy, bw, bh = map(float, line.split()[1:5])
            draw.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                           outline="lime", width=3)
        sheet.paste(image.resize((thumb, thumb)), ((i % columns) * thumb, (i // columns) * thumb))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
