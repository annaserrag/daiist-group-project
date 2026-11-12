"""
The development set for the exam reader: public handwriting that is not from the Mendeley exams, so every
design choice of the reader (which model reads, how question numbers are found, the blank-page and loop
safeguards) is made here, and the 50 Mendeley exams stay a clean evaluation.

No public dataset has what Mendeley has (whole exam pages with numbered answers and a key), so the set is put
together from the parts that exist:

    smhd         Student Messy Handwritten Dataset (RMIT, CC BY-NC 4.0): student pages written in exam-like
                 conditions, with a line-by-line transcription. Every page with a numbered list ('1)', '2:')
                 and a random sample of the others.                     -> reading accuracy, numbers at line starts
    asap         Handwritten ASAP-SAS (Gold and Zesch 2020, non-commercial research use): short answers to one
                 exam prompt, rewritten by hand, with the original text. -> reading accuracy on exam answers
    notebook     School Notebooks EN (ai-forever, MIT): photos of English school notebooks, every word boxed and
                 transcribed by line. Every page with a numbered list and a random sample of the others; teacher
                 comments are left out of the text, as exams arrive uncorrected. -> a second source of numbered lists
    answersheet  the Roboflow answer-sheet pages of the detectors' test booklet, one per photo, with the
                 number of question-number boxes (no text).             -> how many question numbers are found
    short        SMHD pages whitened below their second line: the reader must stop after two lines.
    blank        empty pages, white and lined.                           -> invented text, decoding loops

Downloads (about 690 MB, into WORK/devsets/; see vision/README.md):
    SMHD-forms.rar                  https://doi.org/10.25439/rmt.24312715.v1
    Handwritten-ASAP-SAS-V1.0.zip   https://zenodo.org/records/8088866
    school_notebooks_EN             https://huggingface.co/datasets/ai-forever/school_notebooks_EN (images.zip and
                                    annotations_*.json)
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import json
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from vision import datasets

DEVSETS = datasets.WORK / "devsets"
SMHD = DEVSETS / "smhd" / "x" / "SMHD-forms"
ASAP = DEVSETS / "asap" / "x" / "Handwritten ASAP SAS"
NOTEBOOKS = DEVSETS / "school_notebooks"
SYNTHETIC = DEVSETS / "synthetic"
SEED = 0
N_SMHD_OTHER, N_ASAP, N_NOTEBOOK_OTHER, N_SHORT, N_BLANK = 18, 40, 10, 10, 4
STUDENT_TEXT = {0, 1}  # pupil_text and pupil_comment; 2 is teacher_comment, 3 the paper, 4 the line outlines

# A numbered line in a transcription: '1)', '2:', '3.', 'Q4)'; the number must be followed by a separator.
NUMBERED = re.compile(r"^\s*(?:q\.?\s*)?(\d{1,3})\s*[.):]", re.IGNORECASE)


def clean_truth(text: str) -> str:
    """SMHD marks crossed-out words with '#'; they are not part of the answer."""
    return "\n".join(re.sub(r"\s+", " ", line.replace("#", " ")).strip() for line in text.splitlines()).strip()


def numbers_in(text: str) -> list[str]:
    """The numbers that open a line of the transcription, in order."""
    return [str(int(m.group(1))) for line in text.splitlines() if (m := NUMBERED.match(line))]


def smhd_pages(rng: random.Random) -> list[dict]:
    pages = []
    for txt in sorted(SMHD.glob("*/*.txt")):
        image = next((txt.with_suffix(ext) for ext in (".jpg", ".jpeg") if txt.with_suffix(ext).is_file()), None)
        if image is not None:
            truth = clean_truth(txt.read_text(errors="ignore"))
            pages.append({"kind": "smhd", "image": str(image), "truth": truth, "numbers": numbers_in(truth)})
    numbered = [p for p in pages if len(p["numbers"]) >= 2]
    others = [p for p in pages if len(p["numbers"]) < 2]
    return numbered + rng.sample(others, N_SMHD_OTHER)


def asap_answers() -> list[dict]:
    info = pd.read_csv(ASAP / "information" / "prompt-3.txt", sep="\t", comment="#", header=None,
                       names=["file", "asap_id", "text", "pen", "status"], quoting=3)
    info = info[(info["status"] == "ok") & info["file"].map(lambda f: (ASAP / "prompt-3" / f).is_file())]
    rows = info.sample(N_ASAP, random_state=SEED)
    return [{"kind": "asap", "image": str(ASAP / "prompt-3" / r.file), "truth": str(r.text).strip(), "numbers": []}
            for r in rows.itertuples()]


def notebook_pages(rng: random.Random) -> list[dict]:
    """Each word carries the id of its line (group_id): the words of a line, left to right, and the lines, top to
    bottom, give the page's text."""
    pages = []
    for split in ("train", "val", "test"):
        annotations = json.loads((NOTEBOOKS / f"annotations_{split}.json").read_text())
        lines: dict = defaultdict(list)
        for a in annotations["annotations"]:
            if a["category_id"] in STUDENT_TEXT and a.get("group_id") is not None and a["attributes"].get("translation"):
                xy = np.array(a["segmentation"][0]).reshape(-1, 2)
                lines[(a["image_id"], a["group_id"])].append((xy[:, 0].mean(), xy[:, 1].mean(), a["attributes"]["translation"]))
        for image in annotations["images"]:
            rows = sorted((np.mean([w[1] for w in words]), " ".join(w[2] for w in sorted(words)))
                          for (image_id, _), words in lines.items() if image_id == image["id"])
            truth = clean_truth("\n".join(text for _, text in rows))
            path = NOTEBOOKS / "x" / "images" / image["file_name"]
            if truth and path.is_file():
                pages.append({"kind": "notebook", "image": str(path), "truth": truth, "numbers": numbers_in(truth)})
    numbered = [p for p in pages if len(p["numbers"]) >= 2]
    others = [p for p in pages if len(p["numbers"]) < 2]
    return numbered + rng.sample(others, N_NOTEBOOK_OTHER)


def answersheet_pages() -> list[dict]:
    """One image per photo of the test booklet (the detector never trained on it), with its number of boxes."""
    split = pd.read_csv(datasets.ROOT / "artifacts" / "vision" / "split.csv")
    test = split[(split["dataset"] == "answersheet") & (split["our_split"] == "test")].drop_duplicates("source")
    workspace, project, version = datasets.SOURCES["answersheet"]
    folder = datasets.DATASETS / f"{project}-{version}"
    out = []
    for r in test.itertuples():
        image = next(folder.glob(f"*/images/{r.file}"), None)
        if image is not None:
            out.append({"kind": "answersheet", "image": str(image), "truth": None, "numbers": [],
                        "n_boxes": int(r.n_boxes)})
    return out


def short_pages(smhd: list[dict], rng: random.Random) -> list[dict]:
    """Real pages cut after their second line: everything below is whitened. The second line's bottom is found
    with the text_line detector, so the page keeps its paper, ruling and ink as scanned."""
    from vision import reader
    SYNTHETIC.mkdir(parents=True, exist_ok=True)
    out = []
    for page in rng.sample([p for p in smhd if len(p["truth"].splitlines()) >= 4], N_SHORT):
        image = reader.load_pages(page["image"])[0]
        lines = reader.line_boxes(image, "detectors")
        if len(lines) < 2:
            continue
        cut = int(lines[1][3]) + 4
        image = image.copy()
        image[cut:] = 255
        path = SYNTHETIC / f"short_{Path(page['image']).parent.name[:4]}_{Path(page['image']).stem}.png"
        cv2.imwrite(str(path), image)
        truth = "\n".join(page["truth"].splitlines()[:2])
        out.append({"kind": "short", "image": str(path), "truth": truth, "numbers": numbers_in(truth)})
    return out


def blank_pages() -> list[dict]:
    """White pages and lined pages like SMHD's paper (light lines every 40 pt at 300 dpi), with scanner noise."""
    SYNTHETIC.mkdir(parents=True, exist_ok=True)
    noise = np.random.default_rng(SEED)
    out = []
    for k in range(N_BLANK):
        page = np.full((2829, 2000), 255, np.uint8)
        if k % 2:
            for y in range(300, 2700, 118):
                page[y:y + 3] = 200
        page = np.clip(page.astype(int) - noise.integers(0, 12, page.shape), 0, 255).astype(np.uint8)
        path = SYNTHETIC / f"blank_{k}.png"
        cv2.imwrite(str(path), page)
        out.append({"kind": "blank", "image": str(path), "truth": "", "numbers": []})
    return out


def build() -> pd.DataFrame:
    """The development set as one table: kind, image, truth (text, or None), numbers (expected at line starts)."""
    missing = [p for p in (SMHD, ASAP, NOTEBOOKS / "x") if not p.is_dir()]
    if missing:
        raise SystemExit(f"Download and extract the development data first (see vision/README.md): {missing}")
    rng = random.Random(SEED)
    smhd = smhd_pages(rng)
    rows = (smhd + asap_answers() + notebook_pages(random.Random(SEED)) + answersheet_pages()
            + short_pages(smhd, rng) + blank_pages())
    return pd.DataFrame(rows)
