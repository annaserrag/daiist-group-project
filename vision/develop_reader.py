"""
Every design choice of the exam reader, made on the development set (vision/devset.py) and never on the
Mendeley exams, which stay a clean final evaluation (vision/evaluate_reader.py).

Run with `uv run python -m vision.develop_reader` after downloading the development data (vision/README.md).
The script is meant to be read top to bottom, like train_detector.py:

    Part 1  The development set     what is in it, and what each part is for
    Part 2  Which model reads       TrOCR base and large on the detected lines, Qwen3-VL-2B on the lines, and
                                    Qwen3-VL-2B and 4B on the whole page: character error rate, question numbers
                                    found, invented lines, speed; and whether to cut tall line boxes for TrOCR
    Part 3  Question numbers        the stray-mark prefix of the number rule, on and off
    Part 4  Blank pages and loops   the ink threshold, and the decoding setting that stops the page reader looping
    Part 5  Save                    artifacts/vision/dev_*.csv, dev_choices.json, and every line each try read
                                    (artifacts/vision/reads/dev_reads.csv)

The values chosen here are written into vision/reader.py by hand (BLANK_INK, VLM_GENERATE, SPLIT_TALL, the
number rule), with a comment pointing back to this script. Reads are cached per setting and image in
WORK/devsets/reads/, because the vision-language model takes a few seconds a page: the first run takes about an
hour on an Apple GPU, later runs a few minutes.
"""

# %% Setup
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from vision import devset, reader

pd.set_option("display.width", 140)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "vision"
READS = devset.DEVSETS / "reads"
LINE_MODELS = {"TrOCR base, lines": "microsoft/trocr-base-handwritten",
               "TrOCR large, lines": "microsoft/trocr-large-handwritten",
               "Qwen3-VL-2B, lines": "Qwen/Qwen3-VL-2B-Instruct"}
PAGE_MODELS = {"Qwen3-VL-2B, page": "Qwen/Qwen3-VL-2B-Instruct", "Qwen3-VL-4B, page": "Qwen/Qwen3-VL-4B-Instruct"}
# Once vision/finetune_reader.py has trained a LoRA adapter, the fine-tuned 2B is compared with the others here.
if (ROOT / "weights" / "page_reader_lora" / "adapter_config.json").is_file():
    PAGE_MODELS["Qwen3-VL-2B + LoRA, page"] = "Qwen/Qwen3-VL-2B-Instruct+weights/page_reader_lora"
# The page reader the product uses, and the one Parts 3 and 4 look at. 4B reads a little more accurately (Part 2),
# but finds fewer question numbers on SMHD, takes twice as long, and needs more memory than a free Space has.
PAGE_LABEL = "Qwen3-VL-2B, page"
DECODING = {"greedy": {}, "repetition_penalty 1.02": {"repetition_penalty": 1.02},
            "repetition_penalty 1.05": {"repetition_penalty": 1.05}}
LINE = "=" * 78
SECONDS: dict = {}


def section(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def read_page(image_path: str, generate: dict, model: str = "Qwen/Qwen3-VL-2B-Instruct") -> list[tuple[str, float]]:
    """A page reader's lines for one image with these generation settings (cached, with its seconds)."""
    name = "" if model == "Qwen/Qwen3-VL-2B-Instruct" else f"{model}|"
    key = hashlib.sha1(f"page|{name}{image_path}|{json.dumps(generate, sort_keys=True)}".encode()).hexdigest()[:16]
    cache = READS / f"dec_{key}.json"
    if not cache.is_file():
        reader.VLM_GENERATE = generate
        reader.PAGE_READER = model
        page = reader.load_pages(image_path)[0]
        scale = min(1.0, reader.VLM_PAGE_WIDTH / page.shape[1])
        image = Image.fromarray(cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)).convert("RGB")
        READS.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        lines = reader.vlm_read(image, reader.VLM_PAGE_PROMPT, model, 1024)
        SECONDS[cache.name] = round(time.perf_counter() - started, 2)
        cache.write_text(json.dumps(lines))
        (READS / f"{cache.stem}.seconds").write_text(str(SECONDS[cache.name]))
    return [tuple(line) for line in json.loads(cache.read_text())]


def page_seconds(image_path: str, generate: dict, model: str) -> float:
    name = "" if model == "Qwen/Qwen3-VL-2B-Instruct" else f"{model}|"
    key = hashlib.sha1(f"page|{name}{image_path}|{json.dumps(generate, sort_keys=True)}".encode()).hexdigest()[:16]
    path = READS / f"dec_{key}.seconds"
    return float(path.read_text()) if path.is_file() else np.nan


def read_lines(label: str, image_path: str) -> dict:
    """A line reader's lines for one image (the detected line boxes, read one by one), and its seconds (cached)."""
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


def normalise(text: str) -> str:
    """Compare text, not layout: one space between words, none before punctuation (SMHD puts spaces there)."""
    return re.sub(r"\s+([.,:;])", r"\1", " ".join(text.split()))


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(read: str, truth: str) -> float:
    read, truth = normalise(read), normalise(truth)
    return edit_distance(read, truth) / max(len(truth), 1)


def numbers_read(lines: list) -> list[str]:
    """The question numbers the reader finds in these lines (assign_lines, as in read_exam)."""
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
    """Error rate, numbers found and invented lines over the development set, for one way of reading."""
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


# %% Part 1. The development set
section("Part 1. The development set: public handwriting, no Mendeley exam")
dev = devset.build()
dev["ink"] = [reader.ink(reader.load_pages(p)[0]) for p in dev["image"]]
print(dev.groupby("kind").agg(images=("image", "size"), numbered_lines=("numbers", lambda s: sum(map(len, s))),
                              ink_min=("ink", "min"), ink_median=("ink", "median")).round(4).to_string())

# %% Part 2. Which model reads
section("Part 2. Which model reads: error rate, question numbers, speed")
chosen_generate = reader.VLM_GENERATE
rows = {label: {**score(dev, lambda r, label=label: [tuple(x) for x in read_lines(label, r.image)["lines"]]),
                "seconds_per_image": np.median([read_lines(label, p)["seconds"] for p in dev["image"]])}
        for label in LINE_MODELS}
for label, model in PAGE_MODELS.items():
    rows[label] = {**score(dev, lambda r, model=model: read_page(r.image, chosen_generate, model)),
                   "seconds_per_image": np.nanmedian([page_seconds(p, chosen_generate, model) for p in dev["image"]])}
reader.PAGE_READER = PAGE_MODELS[PAGE_LABEL]
readers = pd.DataFrame(rows).T.sort_values("cer_smhd")
print(readers.round(3).to_string())
best = readers.index[0]
print(f"\nLowest error rate: {best}. Used by the product: {PAGE_LABEL} (see PAGE_LABEL).")

reader.SPLIT_TALL = False
unsplit = score(dev, lambda r: [tuple(x) for x in read_lines("TrOCR base, lines", r.image)["lines"]])
reader.SPLIT_TALL = True
split_table = pd.DataFrame({"cut tall boxes": rows["TrOCR base, lines"], "as detected": unsplit}).T
print("\nTrOCR base, with tall line boxes cut into rows or read as detected:")
print(split_table[["cer_smhd", "cer_asap", "cer_notebook"]].round(3).to_string())

# %% Part 3. Question numbers
section("Part 3. Question numbers: the stray-mark prefix of the number rule")
with_prefix = reader.LEADING_NUMBER
reader.LEADING_NUMBER = re.compile(with_prefix.pattern.replace(r"^\W{0,3}(?:(?:q|question|ex|exercise)\s*\.?\s*|[a-z](?=\d))?",
                                                               r"^\s*(?:(?:q|question|ex|exercise)\s*\.?\s*)?"), re.IGNORECASE)
without = score(dev, lambda r: read_page(r.image, chosen_generate))
reader.LEADING_NUMBER = with_prefix
rules = pd.DataFrame({"with prefix": rows[PAGE_LABEL], "without prefix": without}).T
print(rules[["numbers_found_smhd", "numbers_found_notebook", "numbers_found_answersheet", "extra_numbers_per_page"]].round(3).to_string())

# %% Part 4. Blank pages and loops
section("Part 4. Blank pages, and decoding loops of the page reader")
written, blank = dev.loc[dev["kind"] != "blank", "ink"], dev.loc[dev["kind"] == "blank", "ink"]
blank_ink = round(float(written.min()) / 2, 4)  # half the lightest written page
assert blank.max() < blank_ink, "a blank page has more ink than the threshold"
print(f"ink: blank pages up to {blank.max():.4f}, written pages from {written.min():.4f} -> threshold {blank_ink}")
decoding = pd.DataFrame({name: score(dev, lambda r, g=g: read_page(r.image, g)) for name, g in DECODING.items()}).T
reader.VLM_GENERATE = chosen_generate
print(decoding.round(3).to_string())

# %% Part 5. Save
section("Part 5. Save")
ARTIFACTS.mkdir(parents=True, exist_ok=True)
readers.to_csv(ARTIFACTS / "dev_readers.csv")
split_table.to_csv(ARTIFACTS / "dev_line_boxes.csv")
rules.to_csv(ARTIFACTS / "dev_number_rule.csv")
decoding.to_csv(ARTIFACTS / "dev_decoding.csv")
choices = {"reader": PAGE_LABEL, "lowest_error_rate": best, "blank_ink": blank_ink, "generate": chosen_generate,
           "dev_set": dev.groupby("kind").size().to_dict()}
(ARTIFACTS / "dev_choices.json").write_text(json.dumps(choices, indent=2))

# What every try read, line by line, so each number above can be checked (public data only, no Mendeley exam).
tries = {label: lambda r, label=label: [tuple(x) for x in read_lines(label, r.image)["lines"]] for label in LINE_MODELS}
reader.SPLIT_TALL = False
tries["TrOCR base, lines as detected"] = {r.image: [tuple(x) for x in read_lines("TrOCR base, lines", r.image)["lines"]]
                                          for r in dev.itertuples()}.get
reader.SPLIT_TALL = True
tries |= {f"{label}, {name}": lambda r, m=model, g=g: read_page(r.image, g, m)
          for label, model in PAGE_MODELS.items() for name, g in DECODING.items()
          if model == PAGE_MODELS[PAGE_LABEL] or g == chosen_generate}
dump = []
for name, lines_of in tries.items():
    for r in dev.itertuples():
        lines = lines_of(r.image) if name == "TrOCR base, lines as detected" else lines_of(r)
        dump += [{"try": name, "kind": r.kind, "image": Path(r.image).name, "line": k, "text": text, "confidence": conf}
                 for k, (text, conf) in enumerate(lines, start=1)]
(ARTIFACTS / "reads").mkdir(exist_ok=True)
pd.DataFrame(dump).to_csv(ARTIFACTS / "reads" / "dev_reads.csv", index=False)
print(f"saved {len(dump)} read lines from {len(tries)} tries to artifacts/vision/reads/dev_reads.csv")
print(json.dumps(choices, indent=2))
