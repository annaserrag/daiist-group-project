"""
The training data for fine-tuning the page reader (Qwen3-VL-2B) on exam-like handwriting. None of it comes from
the Mendeley exams (the final evaluation) or from the development set (vision/devset.py, which judges the
fine-tuned reader against the pretrained one), so both stay clean.

No public dataset has whole exam pages with transcriptions, so the set is put together from three sources:

    gnhk   GNHK, GoodNotes Handwriting Kollection (Lee, Chung and Lee, ICDAR 2021; CC BY 4.0): 687 phone photos
           of real English handwriting from Europe, the Americas, Africa and Asia (notes, diaries, shopping lists),
           every word boxed and transcribed. Its own train split trains and its test split validates, less the test pages that
           are a second photo of a training page (drop_leaks).
           -> real pages, many writers, lists
    exam   synthetic exam pages: question numbers and multiple-choice letters made of EMNIST characters (Cohen et
           al. 2017, from NIST SD 19; CC BY 4.0 on the Hugging Face copy), each followed by a letter or by lines
           of OpenHand-Synth handwriting (to-be/OpenHand-Synth, CC BY 4.0; single English lines only).
           -> the exam layout the reader meets: '7. B', '8) answer text', '2a) ...'

Each example is an image and the text the reader should write for it, in the format of reader.VLM_PAGE_PROMPT:
one line of text per handwritten line, top to bottom, question numbers kept at the start of their lines. Images
are grayscale and VLM_PAGE_WIDTH wide, as the reader sees them.

Run with `uv run --group train python -m vision.finetune_data` after downloading the data (vision/README.md).
Writes WORK/finetune/built/{train,val}.jsonl and their images, and artifacts/vision/finetune_data.json (the counts).

    Part 1  GNHK pages          text rebuilt from the word boxes; unreadable words and symbols painted out
    Part 2  Synthetic exams     EMNIST numbers and letters, OpenHand answer lines, on white or ruled paper
    Part 3  Save and check      the files, the counts, and a few sample pages to look at
"""

# %% Setup
import gzip
import hashlib
import io
import json
import random
import re
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageOps

from vision import datasets, reader

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "vision"
FINETUNE = datasets.WORK / "finetune"
GNHK = FINETUNE / "gnhk"          # radhakrishnadeshpande/gnhk-pagelevel: the original images and word annotations
OPENHAND = FINETUNE / "openhand"  # to-be/OpenHand-Synth, its validation split (6,808 images, 15 languages)
EMNIST = FINETUNE / "emnist"      # Heliosoph/EMNIST, the balanced split
BUILT = FINETUNE / "built"
SEED = 0
# Half as many synthetic pages as GNHK-sized pages would allow: the pretrained reader already reads them almost
# perfectly (validation loss 0.24, against 1.03 on GNHK), and each takes about 17 s to train on a laptop.
N_EXAM = {"train": 500, "val": 100}
VAL_WRITERS = 0.15  # share of OpenHand writers kept for validation pages, so no writer is in both splits
PAGE_W = reader.VLM_PAGE_WIDTH
PAGE_H = round(PAGE_W * 297 / 210)  # A4
PAINT_OUT = {"%NA%", "%math%"}  # unreadable words and drawn symbols (circled numbers...): not text, painted out
SKIP = {"%SC%"}                 # crossed-out words stay on the page but are not part of the answer
# EMNIST balanced: 0-9 digits, 10-35 A-Z, then the lowercase letters that differ from their capitals
EMNIST_CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("abdefghnqrt")
MCQ = "ABCD"
LINE = "=" * 78


def section(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def to_gray(data: bytes) -> np.ndarray:
    """Upright (118 GNHK photos carry a camera rotation tag, and their word boxes are drawn on the upright page)."""
    return np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("L"))


def fit_width(page: np.ndarray) -> np.ndarray:
    scale = min(1.0, PAGE_W / page.shape[1])
    return cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


# %% Part 1. GNHK pages
def gnhk_page(image: np.ndarray, annotations: list[dict]) -> tuple[np.ndarray, str]:
    """The page with unreadable words painted out, and its handwritten text: the words of each line (line_idx)
    left to right, the lines top to bottom. Printed words (type P) stay on the page but are not written, as the
    reader is asked for the handwriting only."""
    image = image.copy()
    background = int(np.median(image))
    lines = defaultdict(list)
    for word in annotations:
        xy = np.array([[word["polygon"][f"x{i}"], word["polygon"][f"y{i}"]] for i in range(4)], np.int32)
        if word["text"] in PAINT_OUT:
            cv2.fillPoly(image, [xy], background)
        elif word["type"] == "H" and word["text"] not in SKIP:
            lines[word["line_idx"]].append((xy[:, 0].mean(), xy[:, 1].mean(), word["text"]))
    rows = sorted((np.mean([w[1] for w in words]), " ".join(w[2] for w in sorted(words)))
                  for words in lines.values())
    return image, "\n".join(text for _, text in rows)


def gnhk_pages(split: str) -> list[dict]:
    out = []
    for path in sorted(GNHK.glob(f"{split}-*.parquet")):
        for row in pq.read_table(path, columns=["image", "filename", "annotations"]).to_pylist():
            image, text = gnhk_page(to_gray(row["image"]["bytes"]), json.loads(row["annotations"]))
            if text:
                out.append({"kind": "gnhk", "source": row["filename"], "image": fit_width(image), "text": text,
                            "md5": hashlib.md5(row["image"]["bytes"]).hexdigest()})
    return out


def text_lines(text: str) -> set[str]:
    """The page's lines of 20 characters or more, compared without case and punctuation."""
    return {norm for line in text.splitlines() if len(norm := re.sub(r"\W+", " ", line.lower()).strip()) >= 20}


def drop_leaks(train: list[dict], val: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """GNHK's own split is by photo, not by page: some notebook pages were photographed twice, once in each split
    (eng_AS_063 in test is eng_AS_044 in train, framed differently). A validation page that shares a line of 20
    characters or more with a training page is dropped, and so is the second copy of a file found twice in train."""
    seen, kept = set(), []
    for page in train:
        if page["md5"] not in seen:
            seen.add(page["md5"])
            kept.append(page)
    lines = set().union(*(text_lines(p["text"]) for p in kept))
    clean = [p for p in val if not text_lines(p["text"]) & lines]
    dropped = [p["source"] for p in val if text_lines(p["text"]) & lines]
    return kept, clean, dropped


# %% Part 2. Synthetic exam pages
def white_background(ink: np.ndarray) -> np.ndarray:
    """Some OpenHand lines carry scanner noise on a grey background, which would show as a grey box on the page:
    the background (the 90th percentile) is brought to white and the ink scaled with it."""
    return np.clip(ink.astype(np.float32) * 255 / max(np.percentile(ink, 90), 1), 0, 255).astype(np.uint8)


def openhand_lines() -> dict[str, list[tuple[np.ndarray, str]]]:
    """Single English lines of OpenHand-Synth as (image, text), split by writer."""
    rows = []
    for path in sorted(OPENHAND.glob("data-*.arrow")):
        with pa.ipc.open_stream(path) as stream:
            rows += stream.read_all().select(["image", "text", "writer_id", "language", "source"]).to_pylist()
    split = {"train": [], "val": []}
    for r in rows:
        if r["language"] == "eng" and r["source"] in ("wikipedia-simple", "tatoeba") and "\\n" not in r["text"]:
            val = zlib.crc32(str(r["writer_id"]).encode()) % 100 < VAL_WRITERS * 100
            split["val" if val else "train"].append((white_background(to_gray(r["image"]["bytes"])), r["text"].strip()))
    return split


def emnist_glyphs(split: str) -> dict[str, list[np.ndarray]]:
    """EMNIST digits, A-D and a-b as dark ink on white, cropped to the ink (the files store them transposed)."""
    def idx(name: str, offset: int) -> np.ndarray:
        return np.frombuffer(gzip.open(EMNIST / f"emnist-balanced-{split}-{name}.gz").read()[offset:], np.uint8)
    images = idx("images-idx3-ubyte", 16).reshape(-1, 28, 28).transpose(0, 2, 1)
    labels = idx("labels-idx1-ubyte", 8)
    glyphs = defaultdict(list)
    for image, label in zip(images, labels):
        char = EMNIST_CLASSES[label]
        if char.isdigit() or char in MCQ or char in "ab":
            ys, xs = np.nonzero(image > 40)
            glyphs[char].append(255 - image[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    return glyphs


def paste(page: np.ndarray, ink: np.ndarray, x: int, y: int) -> int:
    """Darken the page with this ink at (x, y), clipped to the page; returns the x after it."""
    y0, x0 = max(y, 0), max(x, 0)
    h, w = min(ink.shape[0] - (y0 - y), page.shape[0] - y0), min(ink.shape[1] - (x0 - x), page.shape[1] - x0)
    if h > 0 and w > 0:
        page[y0:y0 + h, x0:x0 + w] = np.minimum(page[y0:y0 + h, x0:x0 + w], ink[y0 - y:y0 - y + h, x0 - x:x0 - x + w])
    return x + ink.shape[1]


def scaled(ink: np.ndarray, height: int) -> np.ndarray:
    return cv2.resize(ink, (max(1, round(ink.shape[1] * height / ink.shape[0])), height), interpolation=cv2.INTER_AREA)


def write_label(page: np.ndarray, label: str, sep: str, glyphs: dict, x: int, y: int, height: int,
                rng: random.Random) -> int:
    """A question number ('12', '2a') from EMNIST characters, then its separator drawn as a pen mark."""
    for char in label:
        ink = scaled(rng.choice(glyphs[char]), height if char.isdigit() else round(height * 0.75))
        x = paste(page, ink, x, y + height - ink.shape[0]) + rng.randint(1, 5)
    thickness = max(2, height // 12)
    if sep == ".":
        cv2.circle(page, (x + thickness, y + height - thickness), thickness, 30, -1)
        return x + 3 * thickness
    cv2.ellipse(page, (x, y + height // 2), (height // 4, height // 2), 0, -70, 70, 30, thickness)
    return x + height // 4 + thickness


def exam_page(lines: list, glyphs: dict, rng: random.Random) -> tuple[np.ndarray, str]:
    """An exam answer page: questions numbered from a random start, each a multiple-choice letter or one to three
    lines of answer, written down the page on white or ruled paper until it is full."""
    page = np.full((PAGE_H, PAGE_W), 255, np.uint8)
    ruled = rng.random() < 0.5
    pitch = rng.randint(60, 80)  # distance between written rows
    if ruled:
        page[np.arange(150, PAGE_H - 60, pitch)] = 205
    height = round(pitch * rng.uniform(0.45, 0.6))  # height of a number; answer lines are drawn a little taller
    number = rng.randint(1, 40)
    sep = rng.choice(".)")
    margin = rng.randint(50, 110)
    y = 150 - round(pitch * 0.8) if ruled else rng.randint(80, 160)
    text = []
    while True:
        written = rng.random() < 0.6
        n_rows = rng.randint(1, 3) if written else 1
        if y + n_rows * pitch > PAGE_H - 80:
            break
        part = rng.choice("ab") if written and rng.random() < 0.15 else ""
        label = f"{number}{part}"
        mark = ")" if part else sep
        x = write_label(page, label, mark, glyphs, margin, y, height, rng) + rng.randint(15, 35)
        if written:
            rows = []
            for k in range(n_rows):
                ink, line_text = rng.choice(lines)
                ink = scaled(ink, round(height * 1.5))
                left = x if k == 0 else margin + rng.randint(0, 40)
                if ink.shape[1] > PAGE_W - left - 40:  # squeeze a long line into the page
                    ink = cv2.resize(ink, (PAGE_W - left - 40, ink.shape[0]), interpolation=cv2.INTER_AREA)
                paste(page, ink, left, y + k * pitch - round(height * 0.3))
                rows.append(line_text)
            text.append(f"{label}{mark} " + "\n".join(rows))
        else:
            letter = rng.choice(MCQ)
            paste(page, scaled(rng.choice(glyphs[letter]), height), x, y)
            text.append(f"{label}{mark} {letter}")
        y += n_rows * pitch + (0 if ruled else rng.randint(0, pitch // 2))
        number += 1
    noise = np.random.default_rng(rng.randint(0, 2**31))
    page = np.clip(page.astype(int) - noise.integers(0, 10, page.shape), 0, 255).astype(np.uint8)
    return page, "\n".join(text)


def exam_pages(split: str, lines: list, glyphs: dict) -> list[dict]:
    rng = random.Random(f"{SEED}-{split}")
    out = []
    for k in range(N_EXAM[split]):
        image, text = exam_page(lines, glyphs, rng)
        out.append({"kind": "exam", "source": f"synthetic_{split}_{k:04d}", "image": image, "text": text})
    return out


# %% Part 3. Save and check
def save(split: str, examples: list[dict]) -> pd.DataFrame:
    folder = BUILT / split
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, ex in enumerate(examples):
        path = folder / f"{ex['kind']}_{k:04d}.jpg"
        cv2.imwrite(str(path), ex["image"], [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({"image": str(path), "text": ex["text"], "kind": ex["kind"], "source": ex["source"]})
    with open(BUILT / f"{split}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    missing = [p for p in (GNHK, OPENHAND, EMNIST) if not any(p.glob("*"))]
    if missing:
        raise SystemExit(f"Download the fine-tuning data first (see vision/README.md): {missing}")

    section("Part 1. GNHK pages")
    train, val = gnhk_pages("train"), gnhk_pages("test")
    kept, clean, dropped = drop_leaks(train, val)
    print(f"train {len(train)} pages, {len(train) - len(kept)} duplicate files dropped -> {len(kept)}")
    print(f"validation {len(val)} pages, {len(dropped)} share text with a training page and are dropped -> {len(clean)}")
    print("dropped:", dropped)
    gnhk = {"train": kept, "val": clean}

    section("Part 2. Synthetic exam pages")
    lines = openhand_lines()
    print("OpenHand English lines:", {split: len(v) for split, v in lines.items()})
    exams = {split: exam_pages(split, lines[split], emnist_glyphs("train" if split == "train" else "test"))
             for split in ("train", "val")}
    print("\n--- one page's text ---\n" + exams["train"][0]["text"])

    section("Part 3. Save and check")
    summary = {}
    for split in ("train", "val"):
        df = save(split, gnhk[split] + exams[split])
        df["lines"] = df["text"].str.count("\n") + 1
        df["numbered_lines"] = df["text"].map(lambda t: sum(bool(reader.LEADING_NUMBER.match(l)) for l in t.splitlines()))
        table = df.groupby("kind").agg(pages=("image", "size"), lines=("lines", "sum"),
                                       numbered_lines=("numbered_lines", "sum"))
        print(f"\n{split}:\n{table.to_string()}")
        summary[split] = table.to_dict(orient="index")
    summary["gnhk_dropped"] = {"duplicate_train_files": len(train) - len(kept), "val_pages_sharing_text": dropped}
    samples = ARTIFACTS / "finetune_samples"
    samples.mkdir(parents=True, exist_ok=True)
    for k in range(3):  # synthetic pages only: GNHK photos are other people's notes and stay out of the repo
        cv2.imwrite(str(samples / f"exam_train_{k}.jpg"), exams["train"][k]["image"])
        (samples / f"exam_train_{k}.txt").write_text(exams["train"][k]["text"])
    (ARTIFACTS / "finetune_data.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {BUILT} and {ARTIFACTS / 'finetune_data.json'}")
