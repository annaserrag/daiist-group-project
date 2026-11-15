"""Bridge Anna's page reader to the bundled, numbered Mendeley exam."""
import json
import os
import re
import subprocess
from pathlib import Path

from .catalog import OPTIONS, OPTION_TEXTS, QUESTIONS
from .handoff import VisionExam

VERSION = "vision-finetune/reader-v3/page/bridge-v12"
_ANSWER_SHEET_HEADER = "data science examination | answer sheet"
_BARE_CHOICE = re.compile(r"([1-9][0-9]?)\s+([A-Da-d])")
_CHOICE_WITH_TEXT = re.compile(r"([A-Da-d])(?:\s*[,.:)]\s*|\s+)(.+)", re.DOTALL)
_WORKER = """
import contextlib, json, sys
with contextlib.redirect_stdout(sys.stderr):
    sys.path.insert(1, sys.argv[2])
    from exam_grader.page_confidence import read_exam
    from vision import reader
    if getattr(reader, "PAGE_READER", None) != "Qwen/Qwen3-VL-2B-Instruct":
        raise RuntimeError("GRADE_VISION_ROOT must point to a checkout whose page reader is the pretrained Qwen3-VL-2B.")
    if len(sys.argv) > 3 and sys.argv[3] == "review-lines":
        from exam_grader.selected_line_crops import from_pdf
        from base64 import b64encode
        from io import BytesIO
        images, detail = from_pdf(sys.argv[1], json.load(sys.stdin))
        result = {"images": [], "detail": detail}
        for image, label in images:
            buf = BytesIO()
            image.save(buf, format="PNG")
            result["images"].append([b64encode(buf.getvalue()).decode("ascii"), label])
    elif len(sys.argv) > 3:
        from exam_grader.selected_reader import read_pdf
        result = read_pdf(sys.argv[1], json.load(sys.stdin))
    else:
        result = read_exam(sys.argv[1])
json.dump(result, sys.stdout)
"""


def _score(value):
    return value if type(value) in (int, float) and 0 <= value <= 1 else 0.0


def _option_text(text):
    return " ".join(text.casefold().split()).removesuffix(".")


def _observation(row, question_id):
    text = row["text"]
    confidence = row.get("answer_token_conf", row.get("ocr_conf"))
    if question_id in QUESTIONS and question_id not in OPTIONS:
        confidence = max(_score(confidence), _score(row.get("confirmed_answer_conf")))
    if not isinstance(text, str):
        raise ValueError("Expected a transcription string from the page reader.")
    if question_id in OPTIONS:
        labelled = _CHOICE_WITH_TEXT.fullmatch(text.strip())
        if labelled:
            label = labelled[1].upper()
            option = OPTION_TEXTS[question_id].get(label)
            if option is not None and _option_text(labelled[2]) == _option_text(option):
                text = label
        arrow = re.fullmatch(r"\s*→\s*([A-Da-d])\s*", text)
        if arrow and arrow[1].upper() in OPTIONS[question_id]:
            text = arrow[1].upper()
        # Preserve ambiguous or unmatched text for teacher review.
        letter = re.fullmatch(r"\s*(?:\(([A-Da-d])\)|([A-Da-d])[.)]?)\s*", text)
        if letter:
            text = (letter[1] or letter[2]).upper()
        if text in OPTIONS[question_id]:
            confidence = row.get("choice_token_conf", confidence)
    return {"question_id": question_id, "text": text,
            "confidence": _score(confidence),
            "id_confidence": max(_score(row.get("number_conf")), _score(row.get("confirmed_number_conf")))
                             if question_id is not None else 0.0,
            "state": "read" if text.strip() else "unreadable"}


def is_heading(row, first_page):
    """A line the reader put before the first question number that is part of the page header (subject, name, ID
    number, date, "Answer"), not an answer. The reader only leaves a line unassigned when no question number came
    before it in the whole exam, so these lines sit above the answers by construction. Only short lines on the page
    where the answers start count; a bare option letter, or anything longer, stays for the teacher to check, because
    it could be an answer whose number was not read. So does a line that opens with a one- to three-digit number
    ("3 A or D"); a long digit run, such as a student ID number, does not protect a line."""
    text = " ".join(str(row.get("text", "")).split())
    if not text or row.get("page", first_page) != first_page:
        return False
    if re.fullmatch(r"\(?[A-Da-d][.)]?", text) or re.match(r"^\W*\d{1,3}(?!\d)", text):
        return False  # an option letter, or a line opening with a question-sized number: maybe an answer
    return len(text.split()) <= HEADING_WORDS and len(text) <= HEADING_CHARS


# A header line is at most this long. Chosen before looking at any exam: a subject, a name and an ID number fit; a
# written answer of one sentence does not. Checked on development exams only (eval/header_rule_dev.py).
HEADING_WORDS, HEADING_CHARS = 6, 40


def adapt(raw):
    """Keep duplicate detections and unassigned lines for the grader's review gates."""
    questions, loose = raw["questions"], raw["unassigned"]
    if not isinstance(questions, list) or not isinstance(loose, list):
        raise ValueError("Expected lists of questions and unassigned lines.")
    answers = []
    for row in questions:
        label = row["number"]
        number = int(label) if isinstance(label, str) and re.fullmatch(r"[0-9]{1,2}", label) else None
        question_id = number if number in QUESTIONS and not row.get("part") else None
        answers.append(_observation(row, question_id))
    # Anna's parser expects punctuation after a number. Some answer sheets
    # have a standalone "1  D" line. Only recover it on a positively headed
    # page, keeping the model's original confidence for both tokens.
    def is_header(row):
        return " ".join(row["text"].casefold().split()) == _ANSWER_SHEET_HEADER

    headed_pages = {row["page"] for row in loose if is_header(row) and type(row.get("page")) is int}
    pages = [row["page"] for row in questions if type(row.get("page")) is int]
    first_page = min(pages) if pages else 1
    for row in loose:
        if row.get("page") in headed_pages:
            if is_header(row):
                continue
            choice = _BARE_CHOICE.fullmatch(row["text"].strip())
            if choice and int(choice[1]) in OPTIONS:
                answers.append(_observation({**row, "text": choice[2], "number_conf": row.get("ocr_conf")},
                                            int(choice[1])))
                continue
        if is_heading(row, first_page):
            continue  # page header: neither an answer nor something to send on (it may hold the student's name)
        answers.append(_observation(row, None))
    readable = any(a["question_id"] is not None and a["state"] == "read" for a in answers)
    payload = {"schema_version": "vision.exam.v1", "exam_id": "mendeley-sf3kvjwknt-v1",
               "exam_status": "in_scope" if readable else "unreadable",
               "answers": answers, "usage": {"cost_eur": 0.0}}
    if raw.get("confirmed_other_document") is True:
        payload.update(exam_status="out_of_scope", exam_id=None, answers=[])
    stats = raw.get("stats", {})
    features = {field: stats[key] for key, field in (("lines", "line_count"), ("ink_density", "ink_density"))
                if key in stats}
    if features:
        payload["features"] = features
    VisionExam.model_validate(payload)
    return payload


def read_exam(path):
    pdf = Path(path).resolve(strict=True)
    root = os.getenv("GRADE_VISION_ROOT")
    if not root:
        from .page_confidence import read_exam as read_page
        return adapt(read_page(pdf))

    return adapt(_external_raw(pdf, root))


def read_selected_exam(path, *, definition):
    pdf = Path(path).resolve(strict=True)
    root = os.getenv("GRADE_VISION_ROOT")
    if root:
        return _external_raw(pdf, root, definition)
    from .selected_reader import read_pdf
    return read_pdf(pdf, definition)


def read_review_lines(pdf, item):
    root = os.getenv("GRADE_VISION_ROOT")
    if not root:
        from .selected_line_crops import from_pdf
        return from_pdf(pdf, item)
    from base64 import b64decode
    from io import BytesIO
    from PIL import Image
    data = _external_raw(Path(pdf).resolve(strict=True), root, item, mode="review-lines")
    return [(Image.open(BytesIO(b64decode(raw))).convert("RGB"), label)
            for raw, label in data["images"]], data["detail"]


def _external_raw(pdf, root, definition=None, *, mode="selected"):
    # The two branches can keep their own dependency locks and Python environments.
    checkout = Path(root).expanduser().resolve(strict=True)
    python = checkout / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    result = subprocess.run([str(python), "-X", "utf8", "-c", _WORKER, str(pdf), str(Path(__file__).resolve().parents[1]),
                             *([mode] if definition is not None else [])],
                            input=json.dumps(definition).encode() if definition is not None else None,
                            cwd=checkout, stdout=subprocess.PIPE, check=True)
    return json.loads(result.stdout)


def assign_unresolved(payload, assignments):
    """Apply explicit, one-based teacher assignments to a copy. Preserve duplicate detections."""
    from copy import deepcopy
    checked = deepcopy(payload)
    for position, number in assignments:
        if position < 1 or position > len(checked["answers"]) or number not in QUESTIONS:
            raise ValueError("Observation or question number is outside the allowed range.")
        answer = checked["answers"][position-1]
        if answer["question_id"] is not None:
            raise ValueError("Only an unresolved observation can be assigned.")
        answer.update(question_id=number, id_confidence=1.0)
    VisionExam.model_validate(checked)
    return checked


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export a local reader handoff, or assign an unresolved observation after checking the scan.")
    parser.add_argument("input", type=Path, help="Source PDF or a previously saved JSON handoff")
    parser.add_argument("--output", type=Path, required=True, help="New private JSON file")
    parser.add_argument("--assign", type=int, nargs=2, action="append", default=[], metavar=("OBSERVATION", "QUESTION"),
                        help="One-based observation and visually checked question ID; preserves the answer confidence")
    args = parser.parse_args()
    try:
        if args.input.suffix.lower() == ".json":
            payload = json.loads(args.input.read_bytes())
        elif args.input.suffix.lower() == ".pdf":
            import env
            env.load_env()
            payload = read_exam(args.input)
        else:
            parser.error("Choose a source PDF or JSON handoff.")
        checked = assign_unresolved(payload, args.assign)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(checked, stream, ensure_ascii=False, indent=2)
        for i, observation in enumerate(checked["answers"], 1):
            if observation["question_id"] is None:
                print(f"Observation {i} needs its question number checked.")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
