"""
How a reader is measured on the development set. Shared by step 4 (choosing the reader), step 6 (the fine-tuned
reader against the pretrained one) and eval/check_fresh_handwriting.py.

Three numbers answer the three things grading depends on:

    character error rate      how much of the text is read wrong (edit distance / length of the truth)
    question numbers found    how many of the numbers that open a line are found, with the reader's own rule
                              (reader.assign_lines); on answer sheets, the numbers found against the boxes drawn
    invented lines            lines written on short and blank pages beyond what is on them

Reading a page with a vision-language model takes seconds, so every read is cached in WORK/devsets/reads/ under a
key made of the image, the model and the generation settings. A second run of a chapter only reads the cache.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from vision import devset, reader

READS = devset.DEVSETS / "reads"
LINE_MODELS = {"TrOCR base, lines": "microsoft/trocr-base-handwritten",
               "TrOCR large, lines": "microsoft/trocr-large-handwritten",
               "Qwen3-VL-2B, lines": "Qwen/Qwen3-VL-2B-Instruct"}
PAGE_MODELS = {"Qwen3-VL-2B, page": "Qwen/Qwen3-VL-2B-Instruct", "Qwen3-VL-4B, page": "Qwen/Qwen3-VL-4B-Instruct"}


# --------------------------------------------------------------------------- reading, with a cache

def _page_key(image_path: str, generate: dict, model: str) -> str:
    # The 2B model was the first one compared, so its keys carry no model name. The keys must stay as they are,
    # or every cached read would be lost.
    name = "" if model == "Qwen/Qwen3-VL-2B-Instruct" else f"{model}|"
    return hashlib.sha1(f"page|{name}{image_path}|{json.dumps(generate, sort_keys=True)}".encode()).hexdigest()[:16]


def read_page(image_path: str, generate: dict, model: str = "Qwen/Qwen3-VL-2B-Instruct") -> list[tuple[str, float]]:
    """A page reader's lines (text, confidence) for one image, with these generation settings. The page is
    prepared exactly as reader.read_exam_page prepares it, so what is measured here is what the product does."""
    cache = READS / f"dec_{_page_key(image_path, generate, model)}.json"
    if not cache.is_file():
        reader.VLM_GENERATE = generate
        reader.PAGE_READER = model
        page = reader.load_pages(image_path)[0]
        scale = min(1.0, reader.VLM_PAGE_WIDTH / page.shape[1])
        image = Image.fromarray(cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)).convert("RGB")
        READS.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        lines = reader.vlm_read(image, reader.VLM_PAGE_PROMPT, model, 1024)
        cache.write_text(json.dumps(lines))
        (READS / f"{cache.stem}.seconds").write_text(str(round(time.perf_counter() - started, 2)))
    return [tuple(line) for line in json.loads(cache.read_text())]


def page_seconds(image_path: str, generate: dict, model: str) -> float:
    """How long that read took when it was made (nan for reads cached before timings were kept)."""
    path = READS / f"dec_{_page_key(image_path, generate, model)}.seconds"
    return float(path.read_text()) if path.is_file() else np.nan


def read_lines(label: str, image_path: str) -> dict:
    """A line reader's lines for one image: the line detector's boxes (reader.ocr_lines), each read on its own.
    Returns {"lines": [(text, confidence)], "seconds": s}."""
    key = hashlib.sha1(f"{label}|{image_path}|{reader.SPLIT_TALL}".encode()).hexdigest()[:16]
    cache = READS / f"{key}.json"
    if not cache.is_file():
        page = reader.load_pages(image_path)[0]
        started = time.perf_counter()
        boxes = reader.ocr_lines(page)
        lines = reader.read_crops([reader.crop(page, b) for b in boxes], LINE_MODELS[label])
        READS.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"lines": [list(line) for line in lines if line[0]],
                                     "seconds": round(time.perf_counter() - started, 2)}))
    return json.loads(cache.read_text())


# --------------------------------------------------------------------------- scoring

def normalise(text: str) -> str:
    """Compare text, not layout: one space between words, none before punctuation (SMHD puts spaces there)."""
    return re.sub(r"\s+([.,:;])", r"\1", " ".join(text.split()))


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: the fewest insertions, deletions and substitutions that turn a into b."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(read: str, truth: str) -> float:
    """Character error rate: 0 is a perfect read, 1 is as many errors as there are characters in the truth."""
    read, truth = normalise(read), normalise(truth)
    return edit_distance(read, truth) / max(len(truth), 1)


def numbers_read(lines: list) -> list[str]:
    """The question numbers the reader finds in these lines, with the same rule read_exam uses."""
    result = reader.assign_lines([[(text, conf, [0, 0, 0, 0]) for text, conf in lines]], "dev", time.perf_counter())
    return [q["number"] for q in result["questions"]]


def number_scores(found: list[str], row) -> tuple[float, int]:
    """Share of the expected numbers that were found, and how many extra numbers were read. An answer-sheet page
    has boxes but no text, so there it is the count of numbers against the count of boxes."""
    if row.kind == "answersheet":
        return min(len(found), row.n_boxes) / row.n_boxes, max(len(found) - row.n_boxes, 0)
    expected = Counter(row.numbers)
    hit = sum((Counter(found) & expected).values())
    return (hit / sum(expected.values()) if expected else np.nan), len(found) - hit


def score(dev: pd.DataFrame, lines_of) -> dict:
    """The development-set scores of one way of reading; lines_of(row) returns that row's lines."""
    out = {"smhd": [], "asap": [], "notebook": [], "numbers_smhd": [], "numbers_notebook": [], "numbers_answersheet": [],
           "extra": [], "invented": 0}
    for r in dev.itertuples():
        lines = lines_of(r)
        if r.kind in ("smhd", "asap", "notebook"):
            out[r.kind].append(cer("\n".join(t for t, _ in lines), r.truth))
        if r.kind == "answersheet" or (r.kind in ("smhd", "notebook") and r.numbers):
            found, extra = number_scores(numbers_read(lines), r)
            out[f"numbers_{r.kind}"].append(found)
            out["extra"].append(extra)
        if r.kind in ("short", "blank"):  # anything past the written lines is invented
            out["invented"] += max(len(lines) - len(r.truth.splitlines()), 0)
    return {"cer_smhd": np.mean(out["smhd"]), "cer_asap": np.mean(out["asap"]), "cer_notebook": np.mean(out["notebook"]),
            "numbers_found_smhd": np.mean(out["numbers_smhd"]), "numbers_found_notebook": np.mean(out["numbers_notebook"]),
            "numbers_found_answersheet": np.mean(out["numbers_answersheet"]),
            "extra_numbers_per_page": np.mean(out["extra"]), "invented_lines": out["invented"]}
