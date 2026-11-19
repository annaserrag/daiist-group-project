"""Locate and display original scan lines for a teacher reviewing uncertain OCR."""

from .page_confidence import page_lines, word_scores

def handwriting_bands(page):
    """Locate text lines by their horizontal ink density inside a ruled response box."""
    import cv2
    import numpy as np

    h, w = page.shape
    contours, _ = cv2.findContours(np.uint8(page < 160), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frames = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if (width > .6*w and height > .3*h and width*height < .98*w*h
                and cv2.contourArea(contour) > .94*width*height):
            frames.append((x, y, width, height))
    if not frames:
        return []
    # Use the inside edge of the largest response frame (printed outlines have two edges).
    x, y, width, height = max(frames, key=lambda b: b[2]*b[3])
    pad = max(4, round(w*.008))
    left, right = x+pad, x+width-pad
    top, bottom = y+pad, y+height-pad
    profile = (page[top:bottom, left:right] < 160).sum(axis=1)
    kernel = max(3, round(w*.0075))
    density = np.convolve(profile, np.ones(kernel)/kernel, mode="same")
    local = cv2.dilate(density[:, None], np.ones((round(w*.02)|1, 1), np.uint8)).ravel()
    peaks = np.flatnonzero((density >= local) & (density > .028*w))
    groups = np.split(peaks, np.flatnonzero(np.diff(peaks) > 1)+1)
    centers = [int(g[len(g)//2]) for g in groups if len(g)]
    if not 1 <= len(centers) <= 48:
        return []
    boxes = []
    radius = round(w*.024)
    for i, center in enumerate(centers):
        low, high = max(0, center-radius), min(len(profile), center+radius+1)
        if i:
            low = max(low, (centers[i-1]+center)//2-3)
        if i+1 < len(centers):
            high = min(high, (centers[i+1]+center)//2+3)
        boxes.append((left, top+low, right, top+high))
    return boxes


def anchors(word):
    # Only location uses folded words. The actual evidence token keeps every character.
    return word.strip(".,:;!?()").casefold()


def crop_evidence(words, page_parts, targets, lines):
    """Locate a target by at least two ordered neighbors unique to this part on the page."""
    evidence = {}
    page = [[anchors(w) for w in part] for part in page_parts]
    expected = [anchors(w) for w in words]
    for line, scores in lines:
        if "[?]" in line or "\ufffd" in line:
            continue
        tokens = line.split()
        keys = [anchors(w) for w in tokens]
        conf = word_scores(line, scores)
        for i in targets:
            for j, token in enumerate(tokens):
                left = right = 0
                while left < min(i, j, 3) and keys[j-left-1] == expected[i-left-1]:
                    left += 1
                while right < min(len(tokens)-1-j, len(words)-1-i, 3) and keys[j+right+1] == expected[i+right+1]:
                    right += 1
                if left+right < 2:
                    continue
                before, after = expected[i-left:i], expected[i+1:i+right+1]
                matches = sum(part[pos-left:pos] == before and part[pos+1:pos+1+right] == after
                              for part in page for pos in range(left, len(part)-right))
                if matches == 1:
                    hits = evidence.setdefault(i, {})
                    hits[token] = max(conf[j], hits.get(token, 0))
    return evidence


def review_lines(page, reader, words, other_words, page_parts, page_number, part_label):
    """Return original line images and both OCR readings as teacher-only evidence."""
    from PIL import Image

    targets = {int(i): word for i, word in other_words.items()}
    found = set()
    images, details = [], []
    prompt = ("Transcribe only the visible handwriting verbatim. Preserve case, punctuation and spelling. "
              "Use [?] for unreadable words. Do not correct anything.")
    for line_number, (left, top, right, bottom) in enumerate(handwriting_bands(page), 1):
        image = Image.fromarray(page[top:bottom, left:right]).convert("RGB")
        lines = page_lines(image, reader, prompt=prompt, limit=160, flag_truncation=True)
        if lines and lines[-1] == ("[?]", [0.0]*3):
            continue
        hits = crop_evidence(words, page_parts, targets.keys(), lines)
        if not hits:
            continue
        location = f"Part {part_label}, page {page_number}, line {line_number}"
        labels = []
        for pos, candidates in sorted(hits.items()):
            found.add(pos)
            word = words[pos]
            context = " ".join([*words[max(pos-3, 0):pos], f"[{word}]", *words[pos+1:pos+4]])
            labels.append(word)
            page_read = word if targets[pos] == word else f"{word} / {targets[pos]}"
            crop_read = ", ".join(f"{value} ({score:.2f})" for value, score in candidates.items())
            details.append(f"{location}, word {pos+1}, {context}\nPage OCR: {page_read}   Line OCR: {crop_read}")
        images.append((image, location + ", " + ", ".join(labels)))
    for pos in sorted(targets.keys() - found):
        details.append(f"Word {pos+1} [{words[pos]}]: no line could be located unambiguously.")
    return images, "\n\n".join(details)


def from_pdf(path, item):
    from vision import reader
    pages = reader.load_pages(path)
    number = item["page"]
    if not 1 <= number <= len(pages):
        raise ValueError("Unknown review page")
    return review_lines(pages[number-1], reader, item["words"], item["targets"],
                        item["parts"], number, item["label"])
