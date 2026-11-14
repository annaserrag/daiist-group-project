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


def read_exam(path):
    from vision import reader
    from PIL import Image
    import cv2
    started = time.perf_counter()
    pages = []
    images = []
    originals = reader.load_pages(path)
    for page in originals:
        h, w = page.shape
        lines = []
        image = None
        if reader.ink(page) >= reader.BLANK_INK:
            scale = min(1.0, reader.VLM_PAGE_WIDTH / w)
            image = Image.fromarray(cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)).convert("RGB")
            lines = page_lines(image, reader, prompt=reader.VLM_PAGE_PROMPT, limit=1024)
        pages.append((lines, [0, 0, w, h]))
        images.append(image)
    result = assemble(pages, reader, started)
    for page_no, image in enumerate(images, start=1):
        rows = [r for r in result["questions"] if r["page"] == page_no]
        if not any(r["number_conf"] < _ID_CHECK_THRESHOLD <= r["number_token_conf"] for r in rows):
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
            confirm_numbers(rows, page_lines(image, reader, prompt=_LABEL_PROMPT, limit=256))
    counts = Counter(row["number"] for row in result["questions"])
    verification_images = {}
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
        if score is not None and score >= _TEXT_CHECK_THRESHOLD:
            row["confirmed_answer_conf"] = score
    result["stats"]["seconds"] = round(time.perf_counter() - started, 1)
    return result
