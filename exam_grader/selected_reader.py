"""Read a teacher-selected written paper without using answer keys for OCR."""
import json
import re

from .handoff import VisionExam
from .page_confidence import confirmed_text_score, page_lines, span_score, word_scores

_TRANSCRIBE = (
    "Transcribe the student's handwritten responses verbatim. Keep the visible question or part label "
    "before each answer on a new line. Preserve spelling; mark illegible words [?]. Skip printed prompts, "
    "instructions, sample codes and page numbers. Never invent a missing label, fill a blank, or answer a question."
)
_VERIFY = (
    "Independently copy all handwriting from the answer area in reading order. Start each student's answer "
    "on a fresh line with its actual visible label. Preserve wording and uncertainty [?]. Exclude printed "
    "text. Do not infer labels or content."
)
_LABEL = re.compile(r"^\s*(?:[Qq](?:uestion)?\s*)?\(?([1-9][0-9]?[A-Za-z]?|[A-Za-z])\s*[).:\-]\s*(.*)$")


def observations(lines, identifiers):
    rows = []
    for text, scores in lines:
        if not text.strip():
            continue
        match = _LABEL.fullmatch(text)
        if match:
            label = match[1].casefold()
            rows.append(dict(label=label, question_id=identifiers.get(label), lines=[],
                             id_confidence=span_score(text, scores, *match.span(1))))
            start = match.start(2)
            text, scores = text[start:], scores[start:]
        elif not rows:
            rows.append(dict(label=None, question_id=None, lines=[], id_confidence=0.0))
        if text.strip():
            rows[-1]["lines"].append((text, scores))
    return rows


def compare_reads(first, second, identifiers):
    rows, checks = [observations(lines, identifiers) for lines in (first, second)]
    same_labels = [r["label"] for r in rows] == [r["label"] for r in checks]
    answers = []
    for i, row in enumerate(rows):
        parts = row["lines"]
        text = " ".join(s.strip() for s, _ in parts)
        confidence = id_confidence = 0.0
        if same_labels and row["question_id"] is not None:
            check = checks[i]
            id_confidence = min(row["id_confidence"], check["id_confidence"])
            scores = [w for s, values in parts for w in word_scores(s, values)]
            confidence = confirmed_text_score(text, scores, check["lines"]) or 0.0
        answers.append(dict(question_id=row["question_id"], text=text, confidence=confidence,
                            id_confidence=id_confidence, state="read" if text else "unreadable"))
    if not same_labels:
        # The second view may see an additional answer. Keep it as an unresolved observation.
        answers.append(dict(question_id=None, text="", confidence=0.0, id_confidence=0.0, state="unreadable"))
    return answers


def page_scope(image, reader, definition):
    if definition.get("booklet_question") is not None:
        prompt = ("Read only the filled question-number bubble or explicitly marked question-number box. "
                  "Output only the selected numeral. If absent, multiple or unclear return ?.")
        result = page_lines(image, reader, prompt=prompt, limit=12)
        visible = result[0][0].strip() if len(result) == 1 else ""
        if not visible.isascii() or not visible.isdecimal() or span_score(*result[0]) < .85:
            return "unknown"
        if visible != str(definition["booklet_question"]):
            return "different"
    description = {"title": definition["title"], "questions": [q["question"] for q in definition["questions"]]}
    prompt = ("Identify whether this page belongs to the described exam. Use a matching visible title, "
              "matching printed questions, or handwritten content clearly addressing the specific questions. "
              "Shared numbering or general subject alone is insufficient. Reply SAME, DIFFERENT (clear "
              "contradiction or another document), or UNKNOWN. Description: " + json.dumps(description))
    result = page_lines(image, reader, prompt=prompt, limit=10)
    if len(result) == 1 and span_score(*result[0]) >= .85:
        return {"SAME": "same", "DIFFERENT": "different"}.get(result[0][0].strip(), "unknown")
    return "unknown"


def read_pdf(path, definition):
    from PIL import Image
    from vision import reader

    identifiers = {q["label"].casefold(): q["id"] for q in definition["questions"]}
    answers, scopes = [], []
    for page in reader.load_pages(path):
        if reader.ink(page) < reader.BLANK_INK:
            continue
        image = Image.fromarray(page).convert("RGB")
        preview = image.copy()
        preview.thumbnail((reader.VLM_PAGE_WIDTH, 2200))
        scope = page_scope(preview, reader, definition)
        scopes.append(scope)
        if scope != "same":
            continue
        image.thumbnail((1800, 2600))
        first = page_lines(preview, reader, prompt=_TRANSCRIBE, limit=1536, flag_truncation=True)
        second = page_lines(image, reader, prompt=_VERIFY, limit=1536, flag_truncation=True)
        if any(lines and lines[-1] == ("[?]", [0.0] * 3) for lines in (first, second)):
            scopes.append("unknown")
        else:
            answers.extend(compare_reads(first, second, identifiers))
    status = "unreadable"
    if "different" in scopes:
        status = "out_of_scope"
    elif scopes and all(s == "same" for s in scopes) and any(a["question_id"] is not None for a in answers):
        status = "in_scope"
    payload = dict(schema_version="vision.exam.v1", exam_status=status,
                   exam_id=definition["exam_id"] if status == "in_scope" else None,
                   answers=answers if status == "in_scope" else [], usage={"cost_eur": 0.0})
    VisionExam.model_validate(payload)
    return payload
