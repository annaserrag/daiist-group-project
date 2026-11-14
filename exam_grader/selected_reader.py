"""Read a teacher-selected written paper without using answer keys for OCR."""
import json
import re

from .handoff import VisionExam
from .page_confidence import _TEXT_CHECK_THRESHOLD, confirmed_text_score, page_lines, span_score, word_scores

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

# An inline part must follow a completed sentence. A period in an initial or
# title ("J. B. Smith", "Dr. B. Smith") does not end the preceding answer.
_INLINE = re.compile(r"""(?P<end>[.!?])["'”’)]*\s+(?P<marker>\(?(?P<label>[A-Za-z])[.)])(?=\s+\S)""")
_ABBREVIATION = re.compile(r"(?:\b[A-Za-z]|\b(?:Dr|Mr|Mrs|Ms|Prof|St|vs|etc|e\.g|i\.e))$", re.I)


def part_lines(lines, identifiers):
    """Split only configured, explicit letter labels, keeping their original character scores."""
    active = False
    for text, scores in lines:
        head = _LABEL.fullmatch(text)
        if head:
            active = head[1].casefold() in identifiers
        start = 0
        if active:
            for marker in _INLINE.finditer(text):
                if marker["label"].casefold() not in identifiers:
                    continue
                if marker["end"] == "." and _ABBREVIATION.search(text[:marker.start("end")]):
                    continue
                stop = marker.start("marker")
                yield text[start:stop], scores[start:stop]
                start = stop
        yield text[start:], scores[start:]


def observations(lines, identifiers):
    rows = []
    for text, scores in part_lines(lines, identifiers):
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


def prose_case_only(first, second):
    """Allow a tie-break only on the initial letter of ordinary-looking prose words."""
    if len(first) != len(second) or first == second:
        return False
    for a, b in zip(first, second, strict=True):
        if a == b:
            continue
        if not (re.fullmatch(r"[A-Za-z][a-z]{3,}[.,;:?!]?", a)
                and re.fullmatch(r"[A-Za-z][a-z]{3,}[.,;:?!]?", b) and a.lower() == b.lower()):
            return False
    return True


def confirmed_case_read(first, second, witness):
    """Use only scores for the exact word at the same position, including its case."""
    if witness["id_confidence"] < .85:
        return None
    readings = [" ".join(s.strip() for s, _ in row["lines"]) for row in (first, second, witness)]
    words = [text.split() for text in readings]
    if words[2] not in words[:2] or "[?]" in readings[2] or "\ufffd" in readings[2]:
        return None
    scores = [[w for s, v in row["lines"] for w in word_scores(s, v)] for row in (first, second, witness)]
    confidence = min(max(values[i] for values, tokens in zip(scores, words) if tokens[i] == word)
                     for i, word in enumerate(words[2]))
    if confidence >= _TEXT_CHECK_THRESHOLD:
        return readings[2], confidence
    return None


def compare_reads(first, second, identifiers, *, reread=None, review_targets=None):
    rows, checks = [observations(lines, identifiers) for lines in (first, second)]
    same_labels = [r["label"] for r in rows] == [r["label"] for r in checks]
    answers = []
    third_rows = None
    for i, row in enumerate(rows):
        parts = row["lines"]
        text = " ".join(s.strip() for s, _ in parts)
        confidence = id_confidence = 0.0
        if same_labels and row["question_id"] is not None:
            check = checks[i]
            id_confidence = min(row["id_confidence"], check["id_confidence"])
            scores = [w for s, values in parts for w in word_scores(s, values)]
            confidence = confirmed_text_score(text, scores, check["lines"]) or 0.0
            if not confidence and reread and id_confidence >= .85:
                other_text = " ".join(s.strip() for s, _ in check["lines"])
                if prose_case_only(text.split(), other_text.split()):
                    if third_rows is None:
                        try:
                            witness_lines = reread()
                        except (OSError, RuntimeError):
                            witness_lines = []
                        third_rows = ([] if witness_lines and witness_lines[-1] == ("[?]", [0.0] * 3)
                                      else observations(witness_lines, identifiers))
                    if [r["label"] for r in third_rows] == [r["label"] for r in rows]:
                        confirmed = confirmed_case_read(row, check, third_rows[i])
                        if confirmed:
                            text, confidence = confirmed
            if review_targets is not None and confidence < _TEXT_CHECK_THRESHOLD and id_confidence >= .85:
                words = text.split()
                other = [w for line, _ in check["lines"] for w in line.split()]
                other_scores = [s for line, v in check["lines"] for s in word_scores(line, v)]
                if len(words) == len(other):
                    targets = {str(pos): other[pos] for pos, word in enumerate(words)
                               if word != other[pos] or max(scores[pos], other_scores[pos]) < _TEXT_CHECK_THRESHOLD}
                    if targets:
                        review_targets.append(dict(question_id=row["question_id"], words=words, targets=targets))
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
    answers, scopes, review_pages = [], [], []
    for page_number, page in enumerate(reader.load_pages(path), 1):
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
            def reread():
                detail = Image.fromarray(page).convert("RGB")
                return page_lines(detail, reader,
                                  prompt="Copy the handwritten answers exactly as written, including capitalization, "
                                         "punctuation and original part labels. Omit printed instructions. "
                                         "Use [?] for any illegible words. Do not correct or solve.",
                                  limit=1536, flag_truncation=True)
            targets = []
            answers.extend(compare_reads(first, second, identifiers, reread=reread, review_targets=targets))
            if targets:
                parts = [[word for line, _ in r["lines"] for word in line.split()]
                         for r in observations(first, identifiers)]
                review_pages.append(dict(page=page_number, parts=parts, answers=targets))
    status = "unreadable"
    if "different" in scopes:
        status = "out_of_scope"
    elif scopes and all(s == "same" for s in scopes) and any(a["question_id"] is not None for a in answers):
        status = "in_scope"
    payload = dict(schema_version="vision.exam.v1", exam_status=status,
                   exam_id=definition["exam_id"] if status == "in_scope" else None,
                   answers=answers if status == "in_scope" else [], usage={"cost_eur": 0.0})
    VisionExam.model_validate(payload)
    if status == "in_scope" and review_pages:
        payload["_review_pages"] = review_pages
    return payload
