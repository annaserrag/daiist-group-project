"""Bridge Anna's page reader to the bundled, numbered Mendeley exam."""
import json
import os
import re
import subprocess
from pathlib import Path

from .catalog import OPTIONS, OPTION_TEXTS, QUESTIONS
from .handoff import VisionExam

VERSION = "vision-finetune/reader-v3/page/bridge-v6"
_ANSWER_SHEET_HEADER = "data science examination | answer sheet"
_BARE_CHOICE = re.compile(r"([1-9][0-9]?)\s+([A-Da-d])")
_CHOICE_WITH_TEXT = re.compile(r"([A-Da-d])(?:\s*[,.:)]\s*|\s+)(.+)", re.DOTALL)
_WORKER = """
import contextlib, json, sys
with contextlib.redirect_stdout(sys.stderr):
    sys.path.insert(1, sys.argv[2])
    from exam_grader.page_confidence import read_exam
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
    for row in loose:
        if row.get("page") in headed_pages:
            if is_header(row):
                continue
            choice = _BARE_CHOICE.fullmatch(row["text"].strip())
            if choice and int(choice[1]) in OPTIONS:
                answers.append(_observation({**row, "text": choice[2], "number_conf": row.get("ocr_conf")},
                                            int(choice[1])))
                continue
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

    # The two branches can keep their own dependency locks and Python environments.
    checkout = Path(root).expanduser().resolve(strict=True)
    python = checkout / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    result = subprocess.run([str(python), "-X", "utf8", "-c", _WORKER, str(pdf), str(Path(__file__).resolve().parents[1])],
                            cwd=checkout, stdout=subprocess.PIPE, check=True)
    return adapt(json.loads(result.stdout))
