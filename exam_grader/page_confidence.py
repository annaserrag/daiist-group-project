"""Retain token scores until the page reader has split the question label from the answer."""
import re
import time
from collections import Counter


def scored_lines(pieces):
    """A token can cover several characters, or cross a newline."""
    chars, scores = [], []
    for piece, score in pieces:
        for char in piece:
            if char == "\n":
                if chars:
                    yield "".join(chars), scores
                chars, scores = [], []
            else:
                chars.append(char)
                scores.append(score)
    if chars:
        yield "".join(chars), scores


def span_score(text, scores, start=0, end=None):
    return round(min((p for ch, p in zip(text[start:end], scores[start:end], strict=True)
                      if not ch.isspace()), default=0.0), 3)


_LABEL_PROMPT = ("Read only the handwritten question labels at the start of answers on this page. "
                 "Return one label per line in reading order, preserving repeats. Output only its digits "
                 "and any subpart, or ? for an unclear label. Ignore numbers inside answers. Do not fill gaps.")
_CROP_PROMPT = ("Transcribe only the handwritten question label in this crop. "
                "Preserve the digits exactly. If unclear return ?.")
_ID_CHECK_THRESHOLD = 0.85  # The bundled grader's default ID review threshold.
_TEXT_CHECK_THRESHOLD = 0.80
_LAYOUT_PROMPT = ("Transcribe the entire page including printed and handwritten question numbers and answers. "
                  "Keep every visible number on its answer line. Preserve line breaks. Copy only what is visible.")
_SCOPE_PROMPT = (
    "Classify this page using both its heading and body. A = record of a booking, stock, transaction, "
    "or other administrative business. B = worksheet or exam explicitly labelled as a different subject "
    "from data science. C = a data science answer sheet, an answer fragment, a blank, or uncertain. "
    "Reply with only the letter A, B or C."
)
_ADMIN_TITLE = re.compile(r"\b(?:inventory|stock count|reservation|booking|invoice|receipt|purchase order|manifest)\b")
_EXAM_TITLE = re.compile(r"\b(?:exam(?:ination)?|worksheet|quiz|assessment|test)\b")
_ANSWER_PROMPT = ("Read only the student's handwritten answer labeled {number} on this page. "
                  "Copy the answer verbatim, including all words, spelling and punctuation. "
                  "Omit the question number and printed page headings. Do not solve, correct or paraphrase. "
                  "Mark illegible text with [?].")


def page_lines(image, reader, *, prompt, limit):
    processor, model, torch = reader._vlm(reader.PAGE_READER)
    content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
    message = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(message, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt").to(reader.device())
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=limit, do_sample=False, output_scores=True,
                                   return_dict_in_generate=True, **reader.VLM_GENERATE)
    tokens = generated.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    probs = [float(torch.softmax(logits[0].float(), -1)[token])
             for token, logits in zip(tokens, generated.scores, strict=True)]
    pieces = ((processor.tokenizer.decode([token], skip_special_tokens=True), prob)
              for token, prob in zip(tokens, probs, strict=True))
    lines = [(line, scores) for line, scores in scored_lines(pieces) if line.strip()]
    return lines or [("", [min(probs, default=0.0)])]


def confirm_numbers(rows, lines):
    """A second transcription must agree at the same position on a page with the same label count."""
    labels = [(s, scores) for s, scores in lines if s.strip()]
    if len(labels) != len(rows):
        return
    for row, (line, scores) in zip(rows, labels, strict=True):
        label = re.fullmatch(r"\s*([0-9]{1,3})[.),]?\s*", line)
        if label and not row["part"] and label[1] == row["number"]:
            row["confirmed_number_conf"] = min(row["number_token_conf"], span_score(line, scores, *label.span(1)))


def headed_choices(layout, verification):
    """Normalize an unpunctuated number only for a letter row on an explicitly headed answer sheet."""
    from .vision_finetune import _ANSWER_SHEET_HEADER, _BARE_CHOICE
    if not layout or " ".join(layout[0][0].casefold().split()) != _ANSWER_SHEET_HEADER:
        return layout, verification
    def normalize(lines):
        out = []
        for line, scores in lines:
            match = _BARE_CHOICE.fullmatch(line.strip())
            if match:
                pos = len(line) - len(line.lstrip()) + match.end(1)
                line = line[:pos] + "." + line[pos:]
                scores = [*scores[:pos], scores[pos - 1], *scores[pos:]]
            out.append((line, scores))
        return out
    return normalize(layout), normalize(verification)


def recover_layout(first, layout, verification, reader):
    """Recover labels only when two reads place the same labels before the same answer words."""
    def split(lines):
        header, groups = [], []
        for line, scores in lines:
            match = reader.LEADING_NUMBER.match(line.strip())
            if match:
                offset = len(line) - len(line.lstrip())
                label = match[1], match[2]
                confidence = span_score(line, scores, offset + match.start(1), offset + match.end(1))
                groups.append((label, confidence, match[3].split()))
            elif groups:
                groups[-1][2].extend(line.split())
            else:
                header.extend(line.split())
        return header, groups

    layout, verification = headed_choices(layout, verification)
    header, groups = split(layout)
    _, checked = split(verification)
    if not groups or len(groups) != len(checked):
        return first, []
    confirmed = []
    for (label, score, words), (other, second_score, other_words) in zip(groups, checked, strict=True):
        confidence = min(score, second_score)
        if label != other or not words or (other_words and words != other_words):
            return first, []
        confirmed.append(confidence)
    body = [word for _, _, words in groups for word in words]
    first_words = [word for line, _ in first for word in line.split()]
    from .catalog import OPTIONS
    # The grader compares choice labels and printed option text without case. Preserve case for written answers.
    choices = all(int(label[0]) in OPTIONS and label[1] is None for label, _, _ in groups)
    def comparable(words):
        return [w.casefold() for w in words] if choices else words
    if comparable(first_words) not in (comparable(body), comparable(header + body)):
        return first, []
    # Identical words are aligned across the original handwriting read and the layout read.
    # Ignore header-only words, and never transfer answer confidence to a question label.
    first_scores = [score for line, scores in first for score in word_scores(line, scores)]
    word_iter = iter(first_scores[-len(body):])
    aligned = []
    in_answer = False
    for line, scores in layout:
        match = reader.LEADING_NUMBER.match(line.strip())
        start = 0
        if match:
            in_answer = True
            start = len(line) - len(line.lstrip()) + match.start(3)
        scores = list(scores)
        if in_answer:
            for word in _WORD.finditer(line, start):
                confidence = next(word_iter)
                for i in range(word.start(), word.end()):
                    scores[i] = max(scores[i], confidence)
        aligned.append((line, scores))
    return aligned, confirmed


_WORD = re.compile(r"\S+")


def word_scores(text, scores, start=0):
    return [span_score(text, scores, *m.span()) for m in _WORD.finditer(text, start)]


def confirmed_text_score(text, first_scores, second_lines):
    """Align whole words exactly, allowing only differences in whitespace."""
    words = text.split()
    if not words or "[?]" in text or "\ufffd" in text or len(words) != len(first_scores):
        return None
    second_words = [word for line, _ in second_lines for word in line.split()]
    if words != second_words:
        return None
    second_scores = [p for line, scores in second_lines for p in word_scores(line, scores)]
    # A whole word must be confident in at least one read. Never combine
    # confident parts of two uncertain readings of the same word.
    return min(max(a, b) for a, b in zip(first_scores, second_scores, strict=True))


def verify_crop_words(text, first_scores, page_text, crops):
    """Credit a reread only for an exact, unique multiword span on this page."""
    words = text.split()
    if len(words) != len(first_scores) or not words:
        return None
    scores = list(first_scores)
    for lines in crops:
        span = [word for line, _ in lines for word in line.split()]
        if len(span) < 3 or "[?]" in span:
            continue
        n = len(span)
        if sum(page_text[i:i+n] == span for i in range(len(page_text)-n+1)) != 1:
            continue
        for i in range(len(words)-n+1):
            if words[i:i+n] == span:
                read = [v for line, values in lines for v in word_scores(line, values)]
                scores[i:i+n] = [max(a, b) for a, b in zip(scores[i:i+n], read, strict=True)]
                break
    return min(scores)


def assemble(pages, reader, started):
    """Let Anna assign every line. Attach scores for the exact spans her parser used."""
    original = []
    field_scores = []
    for page_no, (lines, box) in enumerate(pages, start=1):
        normal = []
        for line, scores in lines:
            text = line.strip()
            if not text:
                normal.append(("", round(min(scores, default=0.0), 3), box))
                continue
            offset = len(line) - len(line.lstrip())
            scores = scores[offset:offset + len(text)]
            normal.append((text, span_score(text, scores), box))
            match = reader.LEADING_NUMBER.match(text)
            start = 0
            if match:
                number_score = span_score(text, scores, *match.span(1))
                field_scores.append({"number_token_conf": number_score, "_word_scores": []})
                start = match.start(3)
            if not field_scores or not text[start:].strip():
                continue
            fields = field_scores[-1]
            fields["_word_scores"].extend(word_scores(text, scores, start))
            fields["_last_page"] = page_no
            score = span_score(text, scores, start)
            if "answer_token_conf" in fields:
                fields.pop("choice_token_conf", None)
                score = min(score, fields["answer_token_conf"])
            else:
                letter = re.match(r"\s*(?:→\s*|\(\s*)?([A-Da-d])(?=\W|$)", text[start:])
                if letter:
                    fields["choice_token_conf"] = span_score(text, scores, start+letter.start(1), start+letter.end(1))
            fields["answer_token_conf"] = score
        original.append(normal)
    result = reader.assign_lines(original, "page", started)
    for row, fields in zip(result["questions"], field_scores, strict=True):
        row.update(fields)
    return result


def _title_text(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def confirms_other_document(image, reader, lines):
    """Only a title-like first line and a corroborating page classification can veto grading."""
    nonempty = [(text, scores) for text, scores in lines if text.strip()]
    if len(nonempty) < 2:
        return False
    line, scores = nonempty[0]
    title = _title_text(line)
    if (not line.lstrip()[0].isalpha() or not 2 <= len(title.split()) <= 12
            or "data science" in title or span_score(line, scores) < _TEXT_CHECK_THRESHOLD):
        return False
    if _EXAM_TITLE.search(title):
        expected_label = "B"
    elif _ADMIN_TITLE.search(title):
        expected_label = "A"
    else:
        return False
    try:
        reply = page_lines(image, reader, prompt=_SCOPE_PROMPT, limit=9)
    except (OSError, RuntimeError):
        return False
    if len(reply) != 1:
        return False
    text, scores = reply[0]
    if text.strip() != expected_label:
        return False
    if span_score(text, scores) >= 0.90:
        return True
    if "+" not in getattr(reader, "PAGE_READER", "") or span_score(text, scores) <= 0.5:
        return False
    # LoRA is trained to transcribe. Verify an explicit title a second time when the
    # categorical answer is weak. Generic exam/worksheet headings cannot override review.
    if expected_label == "B" and not re.search(r"\b(?:geography|history|chemistry|physics|biology|literature|mathematics)\b", title):
        return False
    try:
        again = page_lines(image, reader, prompt="Transcribe the printed title at the top of the page exactly. Output only the title.", limit=64)
    except (OSError, RuntimeError):
        return False
    return (len(again) == 1 and _title_text(again[0][0]) == title
            and span_score(*again[0]) >= 0.90)


def read_exam(path):
    from vision import reader
    from PIL import Image
    import cv2
    started = time.perf_counter()
    pages = []
    images = []
    scope_pages = []
    recovered = {}
    originals = reader.load_pages(path)
    ink_pixels = 0.0
    total_pixels = 0
    for page in originals:
        h, w = page.shape
        lines = []
        image = None
        scope_lines = None
        ink_share = reader.ink(page)
        ink_pixels += ink_share * page.size
        total_pixels += page.size
        if ink_share >= reader.BLANK_INK:
            scale = min(1.0, reader.VLM_PAGE_WIDTH / w)
            image = Image.fromarray(cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)).convert("RGB")
            lines = page_lines(image, reader, prompt=reader.VLM_PAGE_PROMPT, limit=1024)
            if "+" in reader.PAGE_READER and any(s.strip() for s, _ in lines) and not any(
                    reader.LEADING_NUMBER.match(s.strip()) for s, _ in lines):
                layout = page_lines(image, reader, prompt=_LAYOUT_PROMPT, limit=1024)
                scope_lines = layout
                normalized, _ = headed_choices(layout, [])
                if any(reader.LEADING_NUMBER.match(s.strip()) for s, _ in normalized):
                    verification = page_lines(image, reader, prompt=_LABEL_PROMPT, limit=1024)
                    lines, scores = recover_layout(lines, layout, verification, reader)
                    if scores:
                        recovered[len(pages) + 1] = scores
        pages.append((lines, [0, 0, w, h]))
        scope_pages.append(scope_lines if scope_lines is not None else lines)
        images.append(image)
    result = assemble(pages, reader, started)
    for page_no, scores in recovered.items():
        rows = [row for row in result["questions"] if row["page"] == page_no]
        for row, confidence in zip(rows, scores, strict=True):
            row["number_conf"] = min(row["number_conf"], confidence)
            row["number_token_conf"] = confidence
            row["confirmed_number_conf"] = confidence
    if total_pixels:
        result["stats"]["ink_density"] = ink_pixels / total_pixels
    # A cover or annex alone cannot disqualify a multi-page submission.
    if images and all(image is not None and confirms_other_document(image, reader, lines)
                      for image, lines in zip(images, scope_pages, strict=True)):
        result["confirmed_other_document"] = True
        result["stats"]["seconds"] = round(time.perf_counter() - started, 1)
        return result
    for page_no, image in enumerate(images, start=1):
        rows = [r for r in result["questions"] if r["page"] == page_no]
        if not any(max(r["number_conf"], r.get("confirmed_number_conf", 0)) < _ID_CHECK_THRESHOLD
                   <= r["number_token_conf"] for r in rows):
            continue
        if len(rows) == 1:
            # On an isolated answer the page model can repeat the same wrong digit.
            # Require a separately localized number, with no answer text for context.
            if not reader.has_detectors():
                continue
            page = originals[page_no - 1]
            boxes = reader.question_boxes(page, "detectors")
            if len(boxes) == 1:
                crop = reader.crop(page, boxes[0])
                crop.thumbnail((320, 160))
                confirm_numbers(rows, page_lines(crop, reader, prompt=_CROP_PROMPT, limit=16))
        else:
            secondary = page_lines(image, reader, prompt=_LABEL_PROMPT, limit=256)
            if "+" in reader.PAGE_READER:
                # LoRA sometimes copies the answer after each standalone label.
                # Require exact answer words for every group before filtering those words out.
                labels, bodies = [], []
                for line, scores in secondary:
                    match = re.fullmatch(r"\s*(?:[Qq]\s*)?([0-9]{1,3})[.):]?\s*", line)
                    if match:
                        labels.append((match[1], scores[match.start(1):match.end(1)]))
                        bodies.append([])
                    elif bodies:
                        bodies[-1].extend(line.split())
                    elif line.strip():
                        break
                else:
                    if any(bodies) and bodies == [r["text"].split() for r in rows]:
                        secondary = labels
            confirm_numbers(rows, secondary)
            if "+" in reader.PAGE_READER and any(max(r["number_conf"], r.get("confirmed_number_conf", 0))
                                                 < _ID_CHECK_THRESHOLD <= r["number_token_conf"] for r in rows):
                layout = page_lines(image, reader, prompt=_LAYOUT_PROMPT, limit=1024)
                check = assemble([(layout, [0, 0, 0, 0])], reader, started)["questions"]
                if [(r["number"], r["part"], r["text"]) for r in rows] == [
                        (r["number"], r["part"], r["text"]) for r in check]:
                    for row, repeat in zip(rows, check, strict=True):
                        row["confirmed_number_conf"] = max(row.get("confirmed_number_conf", 0),
                            min(row["number_token_conf"], repeat["number_token_conf"]))
    counts = Counter(row["number"] for row in result["questions"])
    verification_images = {}
    crop_reads = {}
    for row in result["questions"]:
        first_scores = row.pop("_word_scores")
        last_page = row.pop("_last_page", None)
        number = row["number"]
        page_no = row["page"]
        if (not number or not number.isascii() or not number.isdecimal() or not 21 <= int(number) <= 35
                or row["part"] or counts[number] != 1 or page_no != last_page
                or max(row["number_conf"], row.get("confirmed_number_conf", 0)) < _ID_CHECK_THRESHOLD
                or row.get("answer_token_conf", 1) >= _TEXT_CHECK_THRESHOLD or not first_scores):
            continue
        if page_no not in verification_images:
            detail = Image.fromarray(originals[page_no - 1]).convert("RGB")
            detail.thumbnail((1750, 2200))
            verification_images[page_no] = detail
        try:
            lines = page_lines(verification_images[page_no], reader,
                               prompt=_ANSWER_PROMPT.format(number=number), limit=768)
        except (OSError, RuntimeError):
            # A failed optional inference leaves the original reading available for review.
            continue
        score = confirmed_text_score(row["text"], first_scores, lines)
        if (page_no in recovered and score is not None and score < _TEXT_CHECK_THRESHOLD
                and reader.has_detectors()):
            if page_no not in crop_reads:
                page = originals[page_no - 1]
                crop_reads[page_no] = []
                for box in reader.line_boxes(page, "detectors"):
                    crop = reader.crop(page, box, pad=12)
                    crop.thumbnail((1400, 550))
                    try:
                        crop_reads[page_no].append(page_lines(crop, reader, prompt=reader.VLM_PAGE_PROMPT, limit=94))
                    except (OSError, RuntimeError):
                        continue
            page_words = [word for line, _ in pages[page_no - 1][0] for word in line.split()]
            crop_score = verify_crop_words(row["text"], first_scores, page_words, crop_reads[page_no])
            if crop_score is not None:
                score = max(score, crop_score)
        if score is not None and score >= _TEXT_CHECK_THRESHOLD:
            row["confirmed_answer_conf"] = score
    result["stats"]["seconds"] = round(time.perf_counter() - started, 1)
    return result
