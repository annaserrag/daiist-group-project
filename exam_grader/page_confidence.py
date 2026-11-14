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
            read_score = min(row["number_token_conf"], span_score(line, scores, *label.span(1)))
            row["confirmed_number_conf"] = max(row.get("confirmed_number_conf", 0), read_score)


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


def course_term_words(words):
    """Case variants of these full course terms carry the same meaning. Keep symbols and variables exact."""
    normalized = list(words)
    for i, word in enumerate(words):
        if word.lower() in {"k-means", "kmeans"}:
            normalized[i] = "k-means"
        elif word.lower() == "supervised" and i+1 < len(words) and words[i+1].lower() == "learning":
            normalized[i:i+2] = ["supervised", "learning"]
    return normalized


def verify_crop_words(text, first_scores, page_text, crops, *, course_terms=False):
    """Credit a reread only for an exact, unique multiword span on this page."""
    words = text.split()
    if len(words) != len(first_scores) or not words:
        return None
    normalize = course_term_words if course_terms else list
    words, page_text = normalize(words), normalize(page_text)
    scores = list(first_scores)
    for lines in crops:
        span = normalize([word for line, _ in lines for word in line.split()])
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


def confirm_isolated_written_id(row, page, reader):
    """Require a strong page digit and a single agreeing label from a separate crop."""
    number = row["number"]
    if (not number or not number.isascii() or not number.isdecimal() or not 21 <= int(number) <= 35
            or row["part"] or not row["number_conf"] < _ID_CHECK_THRESHOLD <= row["number_token_conf"]):
        return False
    try:
        labels = local_labels(page, reader)
    except (OSError, RuntimeError):
        return False
    if len(labels) != 1 or labels[0][0] != number or labels[0][1] < _ID_CHECK_THRESHOLD:
        return False
    row["confirmed_number_conf"] = min(row["number_token_conf"], labels[0][1])
    return True


def verify_isolated_answer(row, first_scores, page, reader):
    """Align complete words from separate line images against the sole answer on this page."""
    text = row["text"]
    words = text.split()
    if not words or "[?]" in text or "\ufffd" in text:
        return None
    crops = []
    for x0, y0, x1, y1 in reader.line_boxes(page, "detectors"):
        # Reduce ink from adjacent lines in loose detector boxes.
        inset = (y1 - y0) * .10
        crop = reader.crop(page, [x0, y0+inset, x1, y1-inset], pad=16)
        for line, scores in page_lines(crop, reader, prompt=reader.VLM_PAGE_PROMPT, limit=120):
            label = reader.LEADING_NUMBER.match(line)
            if label:
                if label[1] != row["number"] or label[2]:
                    continue
                line, scores = line[label.start(3):], scores[label.start(3):]
            # A terminal word whose punctuation changed keeps its original score.
            last = re.search(r"(\S+)\s*$", line)
            if last and last[1].isalpha() and last[1] + "." == words[-1]:
                line, scores = line[:last.start()], scores[:last.start()]
            crops.append([(line, scores)])
    return verify_crop_words(text, first_scores, words, crops, course_terms=True)


def is_sheet_header(text):
    from .vision_finetune import _ANSWER_SHEET_HEADER
    return " ".join(text.casefold().split()) == _ANSWER_SHEET_HEADER


def page_body(lines, number, reader):
    """Strip only a leading sheet header and the expected first label. Reject other labels."""
    lines = list(lines)
    if lines and is_sheet_header(lines[0][0]):
        lines = lines[1:]
    if not lines:
        return []
    body = []
    for i, (line, scores) in enumerate(lines):
        label = reader.LEADING_NUMBER.match(line)
        if i == 0 and number is not None:
            if not label or label[1] != number or label[2]:
                return []
            start = label.start(3)
            line, scores = line[start:], scores[start:]
        elif label or is_sheet_header(line):
            return []
        if line.strip():
            body.append((line, scores))
    return body


def verify_page_bodies(row, pages, originals, reader):
    """Re-read the ink of each page and align every word within its original page."""
    import numpy as np
    from PIL import Image
    bodies = [page_body(lines, row["number"] if i == 0 else None, reader)
              for i, (lines, _) in enumerate(pages)]
    if not all(bodies):
        return None
    if len(bodies) == 2:
        # A continuation must carry on a visible bullet list across adjacent pages.
        bullet = re.compile(r"^\s*([-•])\s+\S")
        continuation = bullet.match(bodies[1][0][0])
        if not continuation or not any(line.lstrip().startswith(continuation[1] + " ") for line, _ in bodies[0]):
            return None
    text = " ".join(line for body in bodies for line, _ in body)
    if text.split() != row["text"].split():
        return None
    verified = []
    crops = []
    for i, (body, page) in enumerate(zip(bodies, originals, strict=True)):
        ink = page < 140
        xs, ys = np.flatnonzero(ink.any(axis=0)), np.flatnonzero(ink.any(axis=1))
        if not len(xs):
            return None
        crop = reader.crop(page, [xs[0], ys[0], xs[-1]+1, ys[-1]+1], pad=22)
        crop.thumbnail((2000, 2200))
        crops.append(crop)
        prompt = _LAYOUT_PROMPT if len(bodies) == 2 else reader.VLM_PAGE_PROMPT
        repeated = page_body(page_lines(crop, reader, prompt=prompt, limit=768),
                             row["number"] if i == 0 else None, reader)
        scores = [score for line, values in body for score in word_scores(line, values)]
        score = confirmed_text_score(" ".join(line for line, _ in body), scores, repeated)
        if score is None:
            return None
        verified.append(score)
    if len(crops) == 2 and min(verified) >= _TEXT_CHECK_THRESHOLD:
        # A matching bullet alone does not establish that both pages answer one question.
        joined = Image.new("RGB", (max(im.width for im in crops), sum(im.height for im in crops)+30), "white")
        joined.paste(crops[0], (0, 0))
        joined.paste(crops[1], (0, crops[0].height+30))
        prompt = ("These are two consecutive pages. Transcribe the single handwritten response only if the second "
                  "page continues the same question as the first. If another question starts or the connection "
                  "is unclear, output [?]. Preserve the original question number, both list items and their exact "
                  "words. Omit printed headers.")
        joint_body = page_body(page_lines(joined, reader, prompt=prompt, limit=768), row["number"], reader)
        if [w for line, _ in joint_body for w in line.split()] != text.split():
            return None
    return min(verified)


def local_labels(page, reader):
    """Read the leading ink of separated horizontal writing bands in their physical order."""
    import numpy as np
    from PIL import Image
    ink = page < 100
    ys = np.flatnonzero(ink.sum(axis=1) >= 4)
    groups = np.split(ys, np.flatnonzero(np.diff(ys) > max(12, round(page.shape[1] / 80))) + 1)
    if len(groups) > 40:
        return []
    labels = []
    for band in groups:
        if len(band) < 10:
            continue
        top, bottom = band[0], band[-1]
        xs = np.flatnonzero(ink[top:bottom + 1].sum(axis=0) > 1)
        if not len(xs):
            continue
        left = xs[0]
        crop = Image.fromarray(page[max(0, top-15):bottom+16,
                                    max(0, left-12):left+round(.09 * page.shape[1])]).convert("RGB")
        result = page_lines(crop, reader, prompt=_CROP_PROMPT, limit=12)
        if len(result) != 1:
            continue
        text, scores = result[0]
        match = re.match(r"^\s*(?:[Qq]\s*)?(\d{1,3})(?=\s*[.):,]|\s*$)", text)
        if match:
            labels.append((match[1], span_score(text, scores, *match.span(1))))
    return labels


def enhance_faint_label(row, page, reader):
    """On faint ink, match a contrast-enhanced body and its label crop to the original."""
    import numpy as np
    from PIL import ImageOps
    if (row["part"] or row["number_token_conf"] < _ID_CHECK_THRESHOLD
            or row["number_conf"] >= _ID_CHECK_THRESHOLD):
        return
    width = page.shape[1]
    ink = page < 240
    ys = np.flatnonzero(ink.sum(axis=1) > max(4, width // 400))
    gaps = np.flatnonzero(np.diff(ys) > .026 * width)
    if not len(gaps):
        return
    heading, body = ys[:gaps[0]+1], ys[gaps[0]+1:]
    if heading[-1]-heading[0] > .045*width or not len(body):
        return
    lo, hi = body[0], body[-1]
    area = page[lo:hi+1]
    light = area[area < 240]
    if not len(light) or np.percentile(light, 10) < 140:
        return
    xs = np.flatnonzero(ink[lo:hi+1].sum(axis=0) > 2)
    if not len(xs):
        return
    crop = ImageOps.autocontrast(reader.crop(page, [xs[0], lo, xs[-1], hi], pad=18).convert("L")).convert("RGB")
    lines = page_lines(crop, reader, prompt=_LAYOUT_PROMPT, limit=768)
    if not lines:
        return
    text, scores = lines[0]
    label = re.match(r"^([0-9]{1,3})[.):,]+\s+", text)
    if not label or label[1] != row["number"]:
        return
    words = (text[label.end():] + " " + " ".join(s for s, _ in lines[1:])).split()
    if words != row["text"].split():
        return
    prefix = crop.crop((0, 0, min(crop.width, round(width*.10)), min(crop.height, round(width*.10))))
    isolated = page_lines(prefix, reader, prompt=_CROP_PROMPT, limit=16)
    if len(isolated) == 1:
        line, values = isolated[0]
        ident = re.fullmatch(r"\s*([0-9]{1,3})[.):,]*\s*", line)
        if ident and ident[1] == label[1]:
            confidence = min(span_score(line, values, *ident.span(1)),
                             span_score(text, scores, *label.span(1)), row["number_token_conf"])
            if confidence >= _ID_CHECK_THRESHOLD:
                row["confirmed_number_conf"] = confidence


def gutter_regions(page, rows, reader):
    """Match left-margin ink bands by physical order on a multi-answer written page."""
    import numpy as np
    if len(rows) < 2 or any(r["part"] or not r["number"].isdecimal() or int(r["number"]) <= 20 for r in rows):
        return {}
    width = page.shape[1]
    ink = page[:, :round(width*.12)] < 180
    ys = np.flatnonzero(ink.sum(axis=1) > 3)
    bands = [b for b in np.split(ys, np.flatnonzero(np.diff(ys) > width*.014)+1) if len(b) >= 10]
    if len(bands) == len(rows)+1:
        # A printed page heading is allowed only when separately identified.
        header = reader.crop(page, [0, bands[0][0], width, bands[0][-1]], pad=12)
        lines = page_lines(header, reader, prompt=_LAYOUT_PROMPT, limit=64)
        if len(lines) != 1 or not is_sheet_header(lines[0][0]):
            return {}
        bands = bands[1:]
    if len(bands) != len(rows):
        return {}
    verified = []
    for row, band in zip(rows, bands, strict=True):
        top, bottom = band[0], band[-1]
        xs = np.flatnonzero(ink[top:bottom+1].sum(axis=0) > 1)
        if not len(xs):
            return {}
        digit = reader.crop(page, [xs[0], top, xs[-1], bottom], pad=14)
        reads = page_lines(digit, reader, prompt=_CROP_PROMPT, limit=16)
        confidence = 0
        if len(reads) == 1:
            line, values = reads[0]
            label = re.fullmatch(r"\s*([0-9]{1,3})[.):,]*\s*", line)
            if label and label[1] == row["number"]:
                confidence = min(row["number_token_conf"], span_score(line, values, *label.span(1)))
        verified.append(confidence >= _ID_CHECK_THRESHOLD)
        if verified[-1]:
            row["confirmed_number_conf"] = max(row.get("confirmed_number_conf", 0), confidence)
    return {row["number"]: (max(0, band[0]-round(.035*width)),
                            bands[i+1][0]-round(.035*width) if i+1 < len(bands) else page.shape[0])
            for i, (row, band) in enumerate(zip(rows, bands, strict=True))
            if verified[i] and (i+1 == len(bands) or verified[i+1])}


def assemble(pages, reader, started):
    """Let Anna assign every line. Attach scores for the exact spans her parser used."""
    original = []
    field_scores = []
    for page_no, (lines, box) in enumerate(pages, start=1):
        normal = []
        for i, (line, scores) in enumerate(lines):
            # A printed heading on a new page is outside the preceding answer.
            if page_no > 1 and i == 0 and is_sheet_header(line):
                continue
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
    local_by_page = {}
    unresolved_cells = []
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
            if "+" in reader.PAGE_READER:
                from .choice_columns import columns, verify
                from .vision_finetune import _ANSWER_SHEET_HEADER
                geometry = columns(page)
                if geometry:
                    def read(im, prompt, limit):
                        return page_lines(im, reader, prompt=prompt, limit=limit)
                    heading, cells = geometry
                    title = read(heading, "Transcribe the printed heading exactly.", 64)
                    if len(title) == 1 and " ".join(title[0][0].lower().split()) == _ANSWER_SHEET_HEADER:
                        verified = verify(cells, read)
                        if verified and any(number for number, _, _, _ in verified):
                            lines = []
                            scores = []
                            for number, score, choice, confidence in verified:
                                if number is None:
                                    unresolved_cells.append({"page": len(pages)+1, "text": choice, "ocr_conf": confidence})
                                else:
                                    lines.append((f"{number}. {choice}", [score]*(len(number)+2) + [confidence]))
                                    scores.append(score)
                            recovered[len(pages)+1] = scores
            if "+" in reader.PAGE_READER and any(s.strip() for s, _ in lines) and not any(
                    reader.LEADING_NUMBER.match(s.strip()) for s, _ in lines):
                layout = page_lines(image, reader, prompt=_LAYOUT_PROMPT, limit=1024)
                scope_lines = layout
                normalized, _ = headed_choices(layout, [])
                if not any(reader.LEADING_NUMBER.match(s.strip()) for s, _ in normalized):
                    # Some handwritten labels have only a space before the answer. The
                    # digit must be present in both full reads and in an isolated image crop.
                    bare = re.compile(r"^(\d{1,2})\s+(?=[A-Za-z])")
                    candidates = [bare.match(s) for s, _ in lines if bare.match(s)]
                    if len(candidates) == 1:
                        number = candidates[0][1]
                        local = local_labels(page, reader)
                        local_by_page[len(pages) + 1] = local
                        tagged = [(s, sc) for s, sc in layout if bare.match(s)]
                        if (len(local) == 1 and local[0][0] == number and local[0][1] >= _ID_CHECK_THRESHOLD
                                and len(tagged) == 1 and bare.match(tagged[0][0])[1] == number
                                and [s.split() for s, _ in layout[-len(lines):]] == [s.split() for s,_ in lines]):
                            normalized = []
                            for s, sc in lines:
                                m = bare.match(s)
                                if m:
                                    pos = m.end(1)
                                    s = s[:pos] + "." + s[pos:]
                                    sc = sc[:pos] + [sc[pos-1]] + sc[pos:]
                                normalized.append((s, sc))
                            lines = normalized
                            recovered[len(pages)+1] = [local[0][1]]
                if len(pages)+1 not in recovered and any(reader.LEADING_NUMBER.match(s.strip()) for s, _ in normalized):
                    verification = page_lines(image, reader, prompt=_LABEL_PROMPT, limit=1024)
                    lines, scores = recover_layout(lines, layout, verification, reader)
                    if scores:
                        recovered[len(pages) + 1] = scores
        pages.append((lines, [0, 0, w, h]))
        scope_pages.append(scope_lines if scope_lines is not None else lines)
        images.append(image)
    result = assemble(pages, reader, started)
    result["unassigned"].extend(unresolved_cells)
    result["stats"]["lines"] += len(unresolved_cells)
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
    crop_verified_pages = set()
    regions = {}
    only_headers = all(is_sheet_header(item["text"]) for item in result["unassigned"])
    for page_no, image in enumerate(images, start=1):
        rows = [r for r in result["questions"] if r["page"] == page_no]
        if reader.PAGE_READER == reader.BASE_PAGE_READER and only_headers and rows:
            try:
                if len(images) == len(rows) == 1 and rows[0]["number"].isdecimal() and int(rows[0]["number"]) > 20:
                    enhance_faint_label(rows[0], originals[page_no-1], reader)
                elif any(min(r["number_conf"], r.get("answer_token_conf", 1)) < _ID_CHECK_THRESHOLD for r in rows):
                    regions[page_no] = gutter_regions(originals[page_no-1], rows, reader)
            except (OSError, RuntimeError):
                pass
        if ("+" in reader.PAGE_READER and rows and not any(r["part"] for r in rows)
                and any(max(r["number_conf"], r.get("confirmed_number_conf", 0)) < _ID_CHECK_THRESHOLD for r in rows)):
            labels = local_by_page.get(page_no)
            if labels is None:
                labels = local_labels(originals[page_no-1], reader)
            if [number for number, _ in labels] == [r["number"] for r in rows]:
                for row, (_, score) in zip(rows, labels, strict=True):
                    if score >= _ID_CHECK_THRESHOLD:
                        row["confirmed_number_conf"] = max(row.get("confirmed_number_conf", 0), score)
        if not any(max(r["number_conf"], r.get("confirmed_number_conf", 0)) < _ID_CHECK_THRESHOLD
                   <= r["number_token_conf"] for r in rows):
            continue
        if len(rows) == 1:
            if (reader.PAGE_READER == reader.BASE_PAGE_READER and len(images) == 1
                    and only_headers
                    and confirm_isolated_written_id(rows[0], originals[page_no-1], reader)):
                crop_verified_pages.add(page_no)
                continue
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
                or row["part"] or counts[number] != 1
                or max(row["number_conf"], row.get("confirmed_number_conf", 0)) < _ID_CHECK_THRESHOLD
                or row.get("answer_token_conf", 1) >= _TEXT_CHECK_THRESHOLD or not first_scores):
            continue
        continued = page_no != last_page
        if (reader.PAGE_READER == reader.BASE_PAGE_READER and len(result["questions"]) == 1
                and only_headers and page_no == 1
                and ((len(pages) == 2 and last_page == 2)
                     or (len(pages) == 1 and page_no in crop_verified_pages and result["unassigned"]))):
            try:
                score = verify_page_bodies(row, pages, originals, reader)
            except (OSError, RuntimeError):
                score = None
            if score is not None and score >= _TEXT_CHECK_THRESHOLD:
                row["confirmed_answer_conf"] = score
                continue
        if continued:
            continue
        bounds = regions.get(page_no, {}).get(number)
        if bounds:
            lo, hi = bounds
            region = originals[page_no-1][lo:hi]
            try:
                crop = Image.fromarray(region).convert("RGB")
                reread = page_body(page_lines(crop, reader, prompt=_LAYOUT_PROMPT, limit=768), number, reader)
                score = confirmed_text_score(row["text"], first_scores, reread)
                if score is not None and score >= _TEXT_CHECK_THRESHOLD:
                    row["confirmed_answer_conf"] = score
                    continue
            except (OSError, RuntimeError):
                pass
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
        if page_no in crop_verified_pages and (score is None or score < _TEXT_CHECK_THRESHOLD) and reader.has_detectors():
            try:
                crop_score = verify_isolated_answer(row, first_scores, originals[page_no-1], reader)
            except (OSError, RuntimeError):
                crop_score = None
            if crop_score is not None:
                score = max(score or 0, crop_score)
        second_words = [word for line, _ in lines for word in line.split()]
        same_terms = ("+" in reader.PAGE_READER and "[?]" not in row["text"] and "\ufffd" not in row["text"]
                      and course_term_words(row["text"].split()) == course_term_words(second_words))
        if ((page_no in recovered and score is not None or same_terms)
                and (score is None or score < _TEXT_CHECK_THRESHOLD) and reader.has_detectors()):
            if page_no not in crop_reads:
                page = originals[page_no - 1]
                crop_reads[page_no] = []
                for box in reader.line_boxes(page, "detectors"):
                    crop = reader.crop(page, box, pad=12)
                    crop.thumbnail((1400, 550))
                    try:
                        crop_reads[page_no].append(page_lines(crop, reader, prompt=reader.VLM_PAGE_PROMPT, limit=94))
                        if same_terms and crop.width > 5*crop.height:
                            # The leading phrase alone can preserve small distinctions lost on a resized line.
                            prefix = crop.crop((0, 0, round(.60*crop.width), crop.height))
                            crop_reads[page_no].append(page_lines(prefix, reader, prompt=reader.VLM_PAGE_PROMPT, limit=64))
                            wider = reader.crop(page, box, pad=22)
                            crop_reads[page_no].append(page_lines(wider, reader, prompt=reader.VLM_PAGE_PROMPT, limit=94))
                    except (OSError, RuntimeError):
                        continue
            page_words = [word for line, _ in pages[page_no - 1][0] for word in line.split()]
            crop_score = verify_crop_words(row["text"], first_scores, page_words, crop_reads[page_no], course_terms=same_terms)
            if crop_score is not None:
                score = max(score or 0, crop_score)
        if score is not None and score >= _TEXT_CHECK_THRESHOLD:
            row["confirmed_answer_conf"] = score
    result["stats"]["seconds"] = round(time.perf_counter() - started, 1)
    return result
