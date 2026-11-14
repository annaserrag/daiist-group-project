"""
Read a scanned exam: find the question numbers and the handwritten lines, read them, and
put each line under its question. This is the hand-off from the vision component to the
grading agent, and it assumes nothing about the exam: not how many questions there are,
not which are multiple choice, not the pen colour or the paper.

    read_exam(path) -> {
      "questions": [                       # in reading order, as found on the pages
        {"number": "21",                   # digits read in the question-number box, None if unreadable
         "part": None,                     # sub-part after the digits, e.g. "b" for "21b)" or "ii" for "21.ii"
         "number_text": "21.",             # what TrOCR read in that box
         "number_conf": 0.91,              # how sure the reader is of the number
         "text": "Supervised learning ...",
         "ocr_conf": 0.43,                 # lowest token probability over the answer's lines
         "n_lines": 3, "page": 1, "box": [x0, y0, x1, y1]},
        ...],
      "unassigned": [{"page": 1, "text": "...", "ocr_conf": 0.7}],   # lines before any question number
      "stats": {...}, "versions": {...},
    }

Which answers are multiple choice, and which questions should be there at all, comes
from the teacher's answer key, not from the page: `answers_by_number()` and
`choice_letter()` are what the pipeline uses to match the two.

Two modes, so the evaluation can compare them:
    detectors  the two YOLO models from train_detector.py (weights/*_detector.pt)
    rules      the no-learning baselines from vision/rules.py, for comparison

`uv run python -m vision.reader exam.pdf [rules|detectors]` prints what it reads.
"""

from __future__ import annotations

import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vision import rules

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
OCR_MODEL = "microsoft/trocr-base-handwritten"   # reads the crops in the 'detectors' and 'rules' modes
LINE_READER = "microsoft/trocr-base-handwritten"  # reads each line in the 'ocr' mode (a TrOCR or a vision-language model)
BASE_PAGE_READER = "Qwen/Qwen3-VL-2B-Instruct"
PAGE_READER = BASE_PAGE_READER + "+weights/page_reader_lora"
PAGE_ADAPTER_SHA256 = "12e3a113ff186a494aaaa41192a12cbbddba28814e3386b6d57c0c6112289fc4"
PAGE_READER_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"  # evaluated public checkpoint
PAGE_WIDTH = 2000   # pages are read at about this width; TrOCR resizes every crop to 384 px anyway
DETECT_SIZE = 640   # both detectors were trained on 640 x 640 images (see vision/README.md)
DETECT_CONF = 0.25  # boxes below this are dropped; train_detector.py reports precision and recall at it,
                    # and a sweep on validation (artifacts/vision/conf_sweep_val.csv) to revisit it
OCR_BATCH = 8
READER_VERSION = "reader-v3"  # v3: question sub-parts (3b, 3.ii) and the detection threshold as a constant


# ---------------------------------------------------------------- pages

def load_pages(path: str | Path) -> list[np.ndarray]:
    """Every page as a grayscale image about PAGE_WIDTH wide. PDFs and images (jpg, png...) both work."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium
        pages = []
        for page in pdfium.PdfDocument(str(path)):
            image = page.render(scale=PAGE_WIDTH / page.get_width()).to_pil().convert("L")
            pages.append(np.asarray(image))
        return pages
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Not an image or a PDF: {path.name}")
    scale = PAGE_WIDTH / image.shape[1]
    return [cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)]


# ---------------------------------------------------------------- finding boxes

def has_detectors() -> bool:
    return all((WEIGHTS / f"{name}_detector.pt").is_file() for name in ("question_no", "text_line"))


@lru_cache(maxsize=None)
def _detector(name: str):
    from ultralytics import YOLO
    return YOLO(str(WEIGHTS / f"{name}_detector.pt"))


@lru_cache(maxsize=1)
def device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _stretched(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, find) -> list[np.ndarray]:
    """Run `find` on a region stretched to DETECT_SIZE x DETECT_SIZE, the way the training images were
    exported, and map the boxes back to page coordinates."""
    region = image[y0:y1, x0:x1]
    sx, sy = region.shape[1] / DETECT_SIZE, region.shape[0] / DETECT_SIZE
    square = cv2.resize(region, (DETECT_SIZE, DETECT_SIZE), interpolation=cv2.INTER_AREA)
    return [np.array([b[0] * sx + x0, b[1] * sy + y0, b[2] * sx + x0, b[3] * sy + y0]) for b in find(square)]


def _yolo(name: str, conf: float = DETECT_CONF):
    def find(square: np.ndarray) -> list[np.ndarray]:
        result = _detector(name).predict(cv2.cvtColor(square, cv2.COLOR_GRAY2BGR), conf=conf, imgsz=DETECT_SIZE,
                                         device=device(), verbose=False)[0]
        return list(result.boxes.xyxy.cpu().numpy())
    return find


def question_boxes(page: np.ndarray, mode: str) -> list[np.ndarray]:
    """Question numbers, found on the whole page (the answer-sheet training images were whole pages)."""
    h, w = page.shape
    find = _yolo("question_no") if mode == "detectors" else rules.rule_question_numbers
    return sorted(_stretched(page, 0, 0, w, h, find), key=lambda b: (b[1], b[0]))


def line_boxes(page: np.ndarray, mode: str) -> list[np.ndarray]:
    """Handwritten lines. The lines training images were quarters of a photo, so the detector looks at
    the page in 2 x 2 overlapping tiles; a line cut by a tile edge is joined again by merge_rows."""
    h, w = page.shape
    if mode == "rules":
        return merge_rows(_stretched(page, 0, 0, w, h, rules.rule_lines))
    boxes, find = [], _yolo("text_line")
    ox, oy = int(0.1 * w), int(0.1 * h)
    for x0, x1 in ((0, w // 2 + ox), (w // 2 - ox, w)):
        for y0, y1 in ((0, h // 2 + oy), (h // 2 - oy, h)):
            boxes += _stretched(page, x0, y0, x1, y1, find)
    return merge_rows(boxes)


def merge_rows(boxes: list[np.ndarray]) -> list[np.ndarray]:
    """Join boxes that sit on the same row, so one physical line is read once (single-column pages)."""
    merged: list[np.ndarray] = []
    for box in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        if merged:
            last = merged[-1]
            overlap = min(box[3], last[3]) - max(box[1], last[1])
            if overlap > 0.5 * min(box[3] - box[1], last[3] - last[1]):
                merged[-1] = np.array([min(box[0], last[0]), min(box[1], last[1]),
                                       max(box[2], last[2]), max(box[3], last[3])])
                continue
        merged.append(np.asarray(box, float))
    return merged


# ---------------------------------------------------------------- reading

@lru_cache(maxsize=2)
def _ocr(name: str = OCR_MODEL):
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained(name)
    model = VisionEncoderDecoderModel.from_pretrained(name).to(device()).eval()
    return processor, model, torch


def read_crops(crops: list[Image.Image], name: str = OCR_MODEL) -> list[tuple[str, float]]:
    """TrOCR on each crop: the text and a confidence (the lowest token probability).

    The confidence ranks reads for review; it is not a calibrated probability, and the
    evaluation checks how well it separates right from wrong reads.
    """
    if not crops:
        return []
    if not name.startswith("microsoft/trocr"):
        return [vlm_read(c, VLM_LINE_PROMPT, name, max_new_tokens=64)[0] for c in crops]
    processor, model, torch = _ocr(name)
    out = []
    for i in range(0, len(crops), OCR_BATCH):
        with torch.no_grad():
            pixels = processor(images=crops[i:i + OCR_BATCH], return_tensors="pt").pixel_values.to(device())
            gen = model.generate(pixels, max_new_tokens=64, output_scores=True, return_dict_in_generate=True)
        texts = processor.batch_decode(gen.sequences, skip_special_tokens=True)
        probs = torch.stack([torch.softmax(s, -1).max(-1).values for s in gen.scores], dim=1).cpu()
        finished = (gen.sequences[:, 1:] == processor.tokenizer.pad_token_id).cpu()  # padding after a line ends
        probs = probs.masked_fill(finished[:, :probs.shape[1]], 1.0)
        out += [(text.strip(), round(float(p.min()), 3)) for text, p in zip(texts, probs)]
    return out


# ---------------------------------------------------------------- reading with a vision-language model

VLM_LINE_PROMPT = "Transcribe the handwritten text in this image exactly as written. Output only the text."
VLM_PAGE_PROMPT = ("Transcribe every handwritten line on this exam page, from top to bottom, one output line for each "
                   "written line, exactly as written. Keep question numbers such as '26.' or '7)' at the start of "
                   "their lines. Do not correct, translate, summarise or add anything. Output only the text.")
VLM_PAGE_WIDTH = 1400  # pages are shrunk to this width first: enough to read the writing, and about 2,500 image tokens
BLANK_INK = 0.001      # a page with less ink than this holds no answer and is not sent to the model (see ink):
                       # half the lightest written page of the development set (vision/develop_reader.py)
# Greedy decoding can fall into a loop on a page with little writing ('Cell', 'Cell', 'Cell'...): on the development
# set it wrote 703 lines that are not on the page. A small repetition penalty stopped it (12 lines) without
# raising the error rate (vision/develop_reader.py, Part 4).
VLM_GENERATE = {"repetition_penalty": 1.05}


def ink(page: np.ndarray) -> float:
    """Share of pixels at least 60 grey levels darker than the page's own background (its median): handwriting
    counts, even in a light pen, while ruled lines and scanner noise on an empty page do not."""
    return float((page.astype(np.int16) < np.median(page) - 60).mean())


@lru_cache(maxsize=2)
def _vlm(name: str):
    """name is a Hugging Face model, or 'model+folder' for that model with a LoRA adapter from this folder (relative
    to the repo; vision/finetune_reader.py). The adapter is merged in, so reading is as fast as without it."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    base, _, adapter = name.partition("+")
    if adapter == "weights/page_reader_lora":
        import hashlib
        path = WEIGHTS.parent / adapter / "adapter_model.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained page reader: {path}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != PAGE_ADAPTER_SHA256:
                raise ValueError("The page reader adapter differs from the evaluated checkpoint.")
    revision = PAGE_READER_REVISION if base == "Qwen/Qwen3-VL-2B-Instruct" else None
    processor = AutoProcessor.from_pretrained(base, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(base, revision=revision, dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, WEIGHTS.parent / adapter).merge_and_unload()
    return processor, model.to(device()).eval(), torch


def vlm_read(image: Image.Image, prompt: str, name: str, max_new_tokens: int) -> list[tuple[str, float]]:
    """A vision-language model's transcription, one (text, confidence) per output line. As with TrOCR, the
    confidence is the lowest token probability on that line, greedy decoding so the read is repeatable."""
    processor, model, torch = _vlm(name)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt").to(device())
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, output_scores=True,
                             return_dict_in_generate=True, **VLM_GENERATE)
    tokens = gen.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    probs = [float(torch.softmax(score[0].float(), -1).max()) for score in gen.scores]
    lines, text, low = [], "", 1.0
    for token, prob in zip(tokens, probs):
        piece = processor.tokenizer.decode([token], skip_special_tokens=True)
        for k, part in enumerate(piece.split("\n")):
            if k:  # a newline ends the line
                lines.append((text.strip(), round(low, 3)))
                text, low = "", 1.0
            text += part
            low = min(low, prob) if part.strip() else low
    lines.append((text.strip(), round(low, 3)))
    return [line for line in lines if line[0]] or [("", round(min(probs, default=0.0), 3))]


def crop(page: np.ndarray, box, pad: int = 6) -> Image.Image:
    h, w = page.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    return Image.fromarray(page[max(y0 - pad, 0):min(y1 + pad, h), max(x0 - pad, 0):min(x1 + pad, w)]).convert("RGB")


def number_on_line(line: np.ndarray, numbers: list[np.ndarray]) -> int | None:
    """Index of the question-number box that sits on this line, if any."""
    for k, q in enumerate(numbers):
        centre = (q[1] + q[3]) / 2
        if line[1] <= centre <= line[3] and q[0] < line[0] + 0.5 * (line[2] - line[0]):
            return k
    return None


SUB_PART = r"(?:[a-h]|i{1,3}|iv|vi{0,3}|ix|x)"  # 3b, 3(b), 3.ii ...


def parse_number(text: str) -> tuple[str | None, str | None]:
    """The question number read in a number box, and its sub-part if one follows it: 'Q3.' -> ('3', None),
    '3b)' -> ('3', 'b'), '12(ii)' -> ('12', 'ii'). Exams number questions differently, so nothing here
    assumes a format beyond 'digits first'; the teacher's key decides later which labels exist."""
    match = re.search(rf"(\d+)\s*[.)]?\s*\(?({SUB_PART})?\)?(?![a-z])", text.lower())
    if not match:
        return None, None
    return str(int(match.group(1))), match.group(2)


# A question number at the start of a read line: up to three stray marks before it ('"', '#', '(', or one letter
# stuck to the digits as in 'p16.'), an optional "Q"/"Question"/"Ex", the digits, an optional sub-part written
# against them (3b, 3(b)), then a separator. A digit right after the separator means it is not a question ("1.5").
# On the development set the stray-mark prefix found more numbers (89% vs 84%); repairing misread numbers from the
# sequence, splitting lines at a later number and accepting a number with no separator did not, so they are not
# used (vision/develop_reader.py, Part 3).
LEADING_NUMBER = re.compile(
    rf"^\W{{0,3}}(?:(?:q|question|ex|exercise)\s*\.?\s*|[a-z](?=\d))?(\d{{1,3}})(?:\(?({SUB_PART})\)?)?"
    rf"\s*[.):,\-](?!\d)\s*(.*)$", re.IGNORECASE)


def leading_number(text: str) -> tuple[str | None, str | None, str]:
    """('26', None, 'Distinguish ...') for '26. Distinguish ...'; (None, None, text) when the line does not
    open with a question number."""
    match = LEADING_NUMBER.match(text)
    if not match:
        return None, None, text
    return str(int(match.group(1))), (match.group(2) or "").lower() or None, match.group(3).strip()


# Splitting a line box much taller than a normal line lowered TrOCR's error rate on the development set
# (25.1% vs 26.4%), so the 'ocr' mode does it (vision/develop_reader.py, Part 2).
SPLIT_TALL = True


def ocr_lines(page: np.ndarray) -> list[np.ndarray]:
    """The line detector's boxes, prepared for reading the question number from the text:
    - a box much taller than a normal line usually holds two or more lines written close together, and TrOCR
      reads only one line per crop, so it is cut into rows at the emptiest bands between them."""
    boxes = line_boxes(page, "detectors")
    if not boxes or not SPLIT_TALL:
        return [np.asarray(b, float) for b in boxes]
    height = float(np.median([b[3] - b[1] for b in boxes]))
    ink = page < 128
    out = []
    for x0, y0, x1, y1 in boxes:
        rows = max(1, int(round((y1 - y0) / height)))
        cuts = [y0]
        profile = ink[int(y0):int(y1), int(x0):int(x1)].sum(axis=1)
        for k in range(1, rows):
            centre = int((y1 - y0) * k / rows)
            lo, hi = max(centre - int(height / 3), 1), min(centre + int(height / 3), len(profile) - 1)
            cuts.append(y0 + lo + int(np.argmin(profile[lo:hi])) if hi > lo else y0 + centre)
        cuts.append(y1)
        out += [np.array([x0, a, x1, b]) for a, b in zip(cuts[:-1], cuts[1:])]
    return out


def assign_lines(pages: list[list[tuple[str, float, list]]], mode: str, started: float) -> dict:
    """Put read lines (text, confidence, box) under questions, page by page in reading order: a line that opens
    with a question number starts a new question. It only assumes that a question's number opens its first
    line, which is how most exams are written."""
    questions: list[dict] = []
    unassigned: list[dict] = []
    current: dict | None = None

    def start(number: str, part: str | None, number_text: str, conf: float, page_no: int, box) -> dict:
        return {"number": number, "part": part, "number_text": number_text, "number_conf": conf, "text": "",
                "ocr_conf": None, "n_lines": 0, "page": page_no, "box": [round(float(v)) for v in box]}

    for page_no, lines in enumerate(pages, start=1):
        for text, conf, box in lines:
            number, part, rest = leading_number(text)
            if number is not None:
                current = start(number, part, text[:len(text) - len(rest)].strip(), conf, page_no, box)
                questions.append(current)
                text = rest
                if not text:
                    continue
            if current is None:
                unassigned.append({"page": page_no, "text": text, "ocr_conf": conf})
                continue
            current["text"] = f"{current['text']} {text}".strip()
            current["ocr_conf"] = conf if current["ocr_conf"] is None else min(current["ocr_conf"], conf)
            current["n_lines"] += 1

    return {
        "questions": questions,
        "unassigned": unassigned,
        "stats": {"pages": len(pages), "lines": sum(len(lines) for lines in pages), "questions_found": len(questions),
                  "numbers_unreadable": 0, "seconds": round(time.perf_counter() - started, 1), "device": device()},
        "versions": versions(mode),
    }


def read_exam_ocr_numbers(path: str | Path) -> dict:
    """The 'ocr' mode: no question-number detector. The trained line detector finds the lines, LINE_READER
    reads them, and the question numbers are read from the text (assign_lines)."""
    started = time.perf_counter()
    pages = []
    for page in load_pages(path):
        lines = ocr_lines(page)
        reads = read_crops([crop(page, line) for line in lines], LINE_READER)
        pages.append([(text, conf, line) for line, (text, conf) in zip(lines, reads)])
    return assign_lines(pages, "ocr", started)


def read_exam_page(path: str | Path) -> dict:
    """The 'page' mode: PAGE_READER, a vision-language model, transcribes each whole page line by line, and the
    question numbers are read from the text the same way (assign_lines)."""
    started = time.perf_counter()
    pages = []
    for page in load_pages(path):
        h, w = page.shape
        if ink(page) < BLANK_INK:
            pages.append([])
            continue
        scale = min(1.0, VLM_PAGE_WIDTH / w)
        image = Image.fromarray(cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)).convert("RGB")
        reads = vlm_read(image, VLM_PAGE_PROMPT, PAGE_READER, max_new_tokens=1024)
        pages.append([(text, conf, [0, 0, w, h]) for text, conf in reads])
    return assign_lines(pages, "page", started)


def read_exam(path: str | Path, mode: str | None = None) -> dict:
    """Read every page. mode: 'page' (the default: a vision-language model reads whole pages), 'ocr' (the line
    detector and TrOCR, question numbers read from the text), 'detectors' (both YOLO models) or 'rules' (no
    learning). The Mendeley evaluation compares all four (vision/README.md)."""
    mode = mode or "page"
    if mode == "ocr":
        return read_exam_ocr_numbers(path)
    if mode == "page":
        return read_exam_page(path)
    started = time.perf_counter()
    questions: list[dict] = []
    unassigned: list[dict] = []
    current: dict | None = None
    n_lines = 0
    page_no = 0

    for page_no, page in enumerate(load_pages(path), start=1):
        numbers = question_boxes(page, mode)
        lines = line_boxes(page, mode)
        n_lines += len(lines)
        number_reads = read_crops([crop(page, q) for q in numbers])

        # A line that carries a question number starts that question; the number is cut off before reading.
        owner = [number_on_line(line, numbers) for line in lines]
        text_boxes = []
        for line, k in zip(lines, owner):
            if k is not None and numbers[k][2] + 4 < line[2]:
                text_boxes.append(np.array([numbers[k][2] + 4, line[1], line[2], line[3]]))
            else:
                text_boxes.append(line if k is None else None)
        reads = read_crops([crop(page, b) for b in text_boxes if b is not None])
        reads_iter = iter(reads)

        # Numbers with no detected line still start a question (its answer may be missing or unread).
        started_here = {k for k in owner if k is not None}
        events = [(float(lines[i][1]), "line", i) for i in range(len(lines))]
        events += [(float(numbers[k][1]), "number", k) for k in range(len(numbers)) if k not in started_here]
        for _, kind, i in sorted(events):
            k = owner[i] if kind == "line" else i
            if k is not None:
                number_text, number_conf = number_reads[k]
                number, part = parse_number(number_text)
                current = {"number": number, "part": part, "number_text": number_text,
                           "number_conf": number_conf, "text": "", "ocr_conf": None, "n_lines": 0,
                           "page": page_no, "box": [round(float(v)) for v in numbers[k]]}
                questions.append(current)
            if kind == "number" or text_boxes[i] is None:
                continue
            text, conf = next(reads_iter)
            if current is None:
                unassigned.append({"page": page_no, "text": text, "ocr_conf": conf})
                continue
            current["text"] = f"{current['text']} {text}".strip()
            current["ocr_conf"] = conf if current["ocr_conf"] is None else min(current["ocr_conf"], conf)
            current["n_lines"] += 1

    return {
        "questions": questions,
        "unassigned": unassigned,
        "stats": {"pages": page_no, "lines": n_lines, "questions_found": len(questions),
                  "numbers_unreadable": sum(q["number"] is None for q in questions),
                  "seconds": round(time.perf_counter() - started, 1), "device": device()},
        "versions": versions(mode),
    }


def versions(mode: str) -> dict:
    """What produced a read; logged with every request and used to key the evaluation cache."""
    ocr = {"ocr": LINE_READER, "page": PAGE_READER}.get(mode, OCR_MODEL)
    detectors = ({name: _weights_id(name) for name in ("question_no", "text_line")} if mode == "detectors"
                 else {"text_line": _weights_id("text_line")} if mode == "ocr" else None)
    if mode in ("ocr", "page"):
        detectors = {**(detectors or {}), "numbers": "leading-number-v5"}
    if mode == "page":
        detectors = {**detectors, "generate": VLM_GENERATE, "blank_ink": BLANK_INK}
    return {"reader": READER_VERSION, "mode": mode, "ocr": ocr, "detectors": detectors}


def _weights_id(name: str) -> str:
    path = WEIGHTS / f"{name}_detector.pt"
    return f"{path.name}:{path.stat().st_size}:{int(path.stat().st_mtime)}" if path.is_file() else "missing"


# ---------------------------------------------------------------- matching with the answer key

def answers_by_number(result: dict, expected: set[str] | None = None) -> dict[str, dict]:
    """The answers keyed by the label read ('3', or '3b' when a sub-part was read). `expected` is the set of
    labels in the teacher's key: a sub-part the key does not have falls back to the plain number, so '21.'
    misread as '21a' still matches question 21 on an exam without sub-parts. If a label was found twice
    (the student came back to a question, or a misread), the texts are joined and marked `duplicate`."""
    out: dict[str, dict] = {}
    for q in result["questions"]:
        if q["number"] is None:
            continue
        number = q["number"] + (q.get("part") or "")
        if expected is not None and number not in expected and q["number"] in expected:
            number = q["number"]
        if number in out:
            out[number] = {**out[number], "text": f"{out[number]['text']} {q['text']}".strip(), "duplicate": True,
                           "ocr_conf": min(c for c in (out[number]["ocr_conf"], q["ocr_conf"]) if c is not None)
                           if (out[number]["ocr_conf"], q["ocr_conf"]) != (None, None) else None}
        else:
            out[number] = {**q, "duplicate": False}
    return out


def choice_letter(text: str, options: str = "ABCD") -> str | None:
    """The option a student chose, if exactly one option letter stands on its own in the answer;
    None when there is none or more than one, which sends the answer to review. In a longer answer
    only capitals count, so the article "a" in a sentence is not read as option A."""
    words = re.findall(r"[A-Za-z]+", text)
    short = len(words) <= 3
    found = {w.upper() for w in words if len(w) == 1 and w.upper() in options and (short or w.isupper())}
    return found.pop() if len(found) == 1 else None


if __name__ == "__main__":
    result = read_exam(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    for q in result["questions"]:
        conf = f"{q['ocr_conf']:.2f}" if q["ocr_conf"] is not None else "  - "
        print(f"{str(q['number']):>4} ({q['number_text']!r:>8}, {q['number_conf']:.2f})  ocr {conf} | {q['text'][:90]}")
    for line in result["unassigned"]:
        print(f"   unassigned p{line['page']}: {line['text'][:90]}")
    print(result["stats"], result["versions"]["mode"])
