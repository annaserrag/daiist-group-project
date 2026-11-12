"""
The reader on real exams: all 50 Mendeley exams, used here for evaluation only.
`uv run python -m vision.evaluate_reader`

It compares the reader's modes: the rules baseline, the two trained detectors, the line
detector with question numbers read from the text, and the vision-language model reading
whole pages (vision/reader.py). The detectors were trained on Roboflow data only. A first run
of the rules and detectors modes on these exams (artifacts/vision/reader_eval_*_v1_all50.csv)
showed that the question-number detector does not transfer; every choice made after that was
made on the development set (vision/develop_reader.py), not on these exams, which are read once
the reader is frozen.

What is checked, against the answer key and the teacher's anonymised sheet:
  - questions found: every question in the key that the reader found by its number
  - multiple choice: the teacher's sheet has the letter each student wrote, so the letter read
    is checked one by one
  - written answers: there is no transcription, so only whether each answered question was found
  - numbers read that are not in the key (misreads, or page furniture taken for a question)

Reads are cached per exam and per reader version in WORK/reader_cache/, because a full exam
takes minutes on a laptop CPU. Writes artifacts/vision/reader_eval_*.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vision import datasets, reader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "vision"
MENDELEY = datasets.WORK / "mendeley"
CACHE = datasets.WORK / "reader_cache"
# Cells of the teacher's sheet that mean "no answer": NC is "not corrected" (blank or skipped) in the dataset's
# README; NM and "-" appear the same way. Anything else counts as an answer.
NOT_MARKED = {"NC", "NM", "-", ""}
# The page mode with each vision-language model compared on the development set (vision/develop_reader.py).
PAGE_MODELS = {"page": "Qwen/Qwen3-VL-2B-Instruct", "page-4b": "Qwen/Qwen3-VL-4B-Instruct"}


def mendeley_files() -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """The answer key, the teacher's marks and the raw student PDFs (CC BY 4.0), downloaded by hand.

    Student_MCQ.csv and the student list are not needed and not used: they hold real names, and
    the anonymised teacher sheet already has the letters.
    """
    key_path = MENDELEY / "answerkey.txt"
    marks_path = MENDELEY / "Teacher_manual_marks_Anonymized.csv"
    pdfs = sorted((MENDELEY / "pdfs").glob("Student_*.pdf"), key=lambda p: int(p.stem.split("_")[1]))
    if not key_path.is_file() or not marks_path.is_file() or not pdfs:
        raise SystemExit(
            f"Download the dataset from https://data.mendeley.com/datasets/sf3kvjwknt/1 and put answerkey.txt and "
            f"Teacher_manual_marks_Anonymized.csv in {MENDELEY}, and the raw Student_<n>.pdf files in {MENDELEY / 'pdfs'}.")
    key = pd.read_csv(key_path)
    key["Question_Number"] = key["Question_Number"].astype(str)
    # As text: the sheet mixes letters, marks and codes in the same columns, and pandas would otherwise turn
    # some of them into numbers or booleans.
    return key, pd.read_csv(marks_path, dtype=str).set_index("Student_ID"), pdfs


def read_cached(pdf: Path, mode: str) -> dict:
    """mode is a reader mode, or 'page-4b' for the page mode with the 4B model."""
    reader_mode = "page" if mode in PAGE_MODELS else mode
    if mode in PAGE_MODELS:
        reader.PAGE_READER = PAGE_MODELS[mode]
    cache = CACHE / mode / f"{pdf.stem}.json"
    if cache.is_file():
        saved = json.loads(cache.read_text())
        if saved.get("versions") == reader.versions(reader_mode):
            return saved
    result = reader.read_exam(pdf, reader_mode)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    return result


def evaluate(mode: str, key: pd.DataFrame, marks: pd.DataFrame, pdfs: list[Path]) -> tuple[pd.DataFrame, list]:
    rows, extra, seconds = [], [], []
    for n, pdf in enumerate(pdfs, start=1):
        print(f"[{mode}] {n}/{len(pdfs)} {pdf.name}", flush=True)
        result = read_cached(pdf, mode)
        seconds.append(result["stats"]["seconds"])
        found = reader.answers_by_number(result, expected=set(key["Question_Number"]))
        truth = marks.loc[pdf.stem]
        for _, q in key.iterrows():
            number, is_mcq = q["Question_Number"], q["Type"] == "MCQ"
            answer = found.get(number)
            teacher = truth.get(number)
            row = {"mode": mode, "exam": pdf.stem, "q": int(number), "type": "mcq" if is_mcq else "written",
                   "found": answer is not None, "ocr_conf": answer["ocr_conf"] if answer else np.nan,
                   "number_conf": answer["number_conf"] if answer else np.nan,
                   "duplicate": bool(answer and answer["duplicate"])}
            teacher = None if pd.isna(teacher) or str(teacher).strip().upper() in NOT_MARKED else str(teacher).strip()
            if is_mcq:
                wrote = teacher.upper() if teacher is not None else None
                options = str(q["Options"]).strip().upper() if "Options" in key and pd.notna(q["Options"]) else "ABCD"
                read = reader.choice_letter(answer["text"], options) if answer else None
                row |= {"answered": wrote is not None, "wrote": wrote, "read": read, "letter_right": read == wrote}
            else:
                row |= {"answered": teacher is not None}  # the teacher gave a mark, so there is an answer
            rows.append(row)
        extra.append(sum(number not in set(key["Question_Number"]) for number in found))
    return pd.DataFrame(rows), [seconds, extra]


def summarise(mode: str, rows: pd.DataFrame, seconds: list, extra: list) -> list[dict]:
    mcq = rows[(rows["type"] == "mcq") & rows["answered"]]
    written = rows[(rows["type"] == "written") & rows["answered"]]
    letters_read = mcq["read"].notna()
    out = {
        "exams": rows["exam"].nunique(),
        "questions_in_key_found": rows.loc[rows["answered"], "found"].mean(),
        "mcq_answers": len(mcq),
        "mcq_letter_read_rate": letters_read.mean(),
        "mcq_letter_accuracy_when_read": mcq.loc[letters_read, "letter_right"].mean(),
        "mcq_letter_accuracy_overall": mcq["letter_right"].mean(),
        "written_answers": len(written),
        "written_found_rate": written["found"].mean(),
        "numbers_not_in_key_per_exam": float(np.mean(extra)),
        "duplicates_per_exam": rows.groupby("exam")["duplicate"].sum().mean(),
        "seconds_per_exam_median": float(np.median(seconds)),
    }
    return [{"mode": mode, "metric": k, "value": v} for k, v in out.items()]


def save_reads(modes: list[str], pdfs: list[Path]) -> None:
    """What each mode read under each question number, so every number in the summary can be checked.

    The text stays on this computer (WORK/reads/), outside the repo: every page of a Mendeley exam has a
    handwritten header with the student's name and ID number, and on later pages it is read into the previous
    answer. artifacts/vision/reads/ gets the same rows without the text (the number found, its page, the
    confidence and the length of the answer)."""
    out = []
    for mode in modes:
        for pdf in pdfs:
            out += [{"mode": mode, "exam": pdf.stem, "page": q["page"], "number": q["number"], "part": q.get("part"),
                     "text": q["text"], "chars": len(q["text"]), "confidence": q["ocr_conf"]}
                    for q in read_cached(pdf, mode)["questions"]]
    reads = pd.DataFrame(out)
    (datasets.WORK / "reads").mkdir(parents=True, exist_ok=True)
    reads.to_csv(datasets.WORK / "reads" / "mendeley_reads.csv", index=False)
    (OUT / "reads").mkdir(parents=True, exist_ok=True)
    reads.drop(columns="text").to_csv(OUT / "reads" / "mendeley_reads.csv", index=False)


def main(limit: int | None = None) -> None:
    """All 50 exams, once the reader is frozen. `limit` only checks that the code runs on the first few exams."""
    key, marks, pdfs = mendeley_files()
    pdfs = pdfs[:limit] if limit else pdfs
    modes = ["rules"] + (["detectors", "ocr"] if reader.has_detectors() else []) + list(PAGE_MODELS)
    if "detectors" not in modes:
        print("No trained detectors in weights/ yet: only the rules baseline is evaluated.")
    all_rows, summary = [], []
    for mode in modes:
        rows, (seconds, extra) = evaluate(mode, key, marks, pdfs)
        all_rows.append(rows)
        summary += summarise(mode, rows, seconds, extra)
    rows = pd.concat(all_rows, ignore_index=True)
    save_reads(modes, pdfs)
    summary = pd.DataFrame(summary).pivot(index="metric", columns="mode", values="value")
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT / "reader_eval_questions.csv", index=False)
    summary.to_csv(OUT / "reader_eval_summary.csv")
    print(summary.to_string())

    # Does the OCR confidence separate right from wrong letters? It decides what goes to review.
    mcq = rows[(rows["type"] == "mcq") & rows["read"].notna()]
    if len(mcq):
        print("\nOCR confidence of the letters read, right vs wrong:")
        print(mcq.groupby(["mode", "letter_right"])["ocr_conf"].describe()[["count", "mean", "50%"]].round(3).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
