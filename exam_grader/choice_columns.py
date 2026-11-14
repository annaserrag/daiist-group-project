"""Verify spatially separated number / option cells on a headed two-column answer sheet."""
import re

import cv2
import numpy as np
from PIL import Image, ImageOps


def bands(positions, gap):
    return [group for group in np.split(positions, np.flatnonzero(np.diff(positions) > gap) + 1) if len(group)]


def cell(image, mask):
    yy, xx = np.nonzero(mask)
    crop = image[yy.min():yy.max()+1, xx.min():xx.max()+1].copy()
    crop[~mask[yy.min():yy.max()+1, xx.min():xx.max()+1]] = 255
    return ImageOps.expand(Image.fromarray(crop), border=10, fill=255)


def split_row(image):
    mask = cv2.GaussianBlur(image, (3, 3), 0) < 220
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"))
    shapes = [(i, *map(int, box[:4])) for i, box in enumerate(stats[1:], 1) if box[4] >= 12]
    # The slash is a near-straight descending stroke separating two groups of ink.
    candidates = []
    for i, x, y, w, h in shapes:
        if h < .58 * image.shape[0] or w > .46*h:
            continue
        yy, xx = np.nonzero(labels == i)
        slope, intercept = np.polyfit(yy, xx, 1)
        if not -.48 < slope < .04 or np.std(xx - slope*yy - intercept) > max(2.2, .036*h):
            continue
        left = [j for j, a, b, c, d in shapes if a+c <= x+2]
        right = [j for j, a, b, c, d in shapes if a >= x+w-2
                 and not (d <= .30*h and a <= x+w+.45*h)]
        if not left or not any(stats[j, cv2.CC_STAT_HEIGHT] >= .32*h for j in right):
            continue
        # Ignore only small separator dots/fragments adjacent to the slash. Other ink
        # belongs to the cell, including crossings and a second choice.
        rest = [j for j, a, b, c, d in shapes if j != i and j not in left+right
                and not (d <= .30*h and x-2 <= a <= x+w+.45*h)]
        if rest:
            continue
        candidates.append((cell(image, np.isin(labels, left)), cell(image, np.isin(labels, right)),
                           sum(stats[j, cv2.CC_STAT_WIDTH] >= .22*h for j in right) == 1))
    return candidates[0] if len(candidates) == 1 else None


def columns(page):
    """Return every physical row or decline the page. Never infer IDs from position."""
    mask = cv2.GaussianBlur(page, (3, 3), 0) < 220
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"))
    h, w = page.shape
    rules = [i for i, (x, y, dx, dy, area) in enumerate(stats[1:], 1) if dy > .35*h and dx < .012*w]
    if rules:
        mask[np.isin(labels, rules)] = False
    occupied = np.flatnonzero(mask.sum(axis=1) >= 3)
    stripes = bands(occupied, max(5, w//120))
    if len(stripes) < 4 or stripes[0][-1] > .18*h:
        return []
    body_start = stripes[1][0]
    xs = np.flatnonzero(mask[body_start:].any(axis=0))
    blocks = bands(xs, .16*w)
    if len(blocks) != 2 or any(block[-1]-block[0] > .26*w for block in blocks):
        return []
    result = []
    for block in blocks:
        a, b = block[0], block[-1]+1
        visible = mask[body_start:, a:b].sum(axis=1)
        rows = bands(np.flatnonzero(visible > 0), max(5, w//160))
        if not 3 <= len(rows) <= 25:
            return []
        cells = [split_row(page[body_start+ys[0]:body_start+ys[-1]+1, a:b]) for ys in rows]
        if any(item is None for item in cells):
            return []
        result.append(cells)
    header = Image.fromarray(page[max(0, stripes[0][0]-12):stripes[0][-1]+13]).convert("RGB")
    return header, result


def stack(images):
    width = max(im.width for im in images) + 30
    result = Image.new("L", (width, sum(im.height + 12 for im in images)), 255)
    top = 0
    for im in images:
        result.paste(im, (15, top))
        top += im.height + 12
    return result.convert("RGB")


def read_cell(image, pattern, prompt, read):
    from .page_confidence import span_score
    lines = read(image.convert("RGB"), prompt, 12)
    if len(lines) == 1:
        line, scores = lines[0]
        match = re.fullmatch(r"\s*(" + pattern + r")\s*", line)
        if match:
            return match[1].upper(), span_score(line, scores, *match.span(1))
    return None, 0.0


def verify(rows_by_column, read):
    """Compare each pair in its column with two isolated cell reads. Preserve unresolved rows."""
    from .page_confidence import span_score
    from .catalog import OPTIONS
    results = []
    for rows in rows_by_column:
        joined = []
        number_width = max(row[0].width for row in rows)
        width = number_width + max(row[1].width for row in rows) + 24
        for number, letter, _ in rows:
            view = Image.new("L", (width, max(number.height, letter.height) + 6), 255)
            view.paste(number, (4, 0))
            view.paste(letter, (number_width+20, 0))
            joined.append(view)
        context = read(stack(joined), "Copy the handwritten numbers and letters exactly. Return each pair on its own line. Never fill gaps.", 256)
        if len(context) != len(rows):
            return []
        for (number, letter, clean), (line, scores) in zip(rows, context, strict=True):
            pair = re.fullmatch(r"\s*([1-9][0-9]?)\s*[.):]?\s+(.+?)\s*", line)
            single = pair and re.fullmatch(r"([A-Da-d])[.]?", pair[2])
            n, ns = read_cell(number, r"[1-9][0-9]?", "Transcribe the handwritten number exactly. Output only the digits.", read)
            label = n if pair and pair[1] == n and int(n) in OPTIONS else None
            id_score = min(ns, span_score(line, scores, *pair.span(1))) if label else 0.0
            choice, cs = read_cell(letter, r"[A-Da-d][.]?", "Transcribe the handwritten letter exactly. Use ? for an ambiguous or crossed-out letter.", read) if clean else (None, 0.0)
            if choice:
                choice = choice.rstrip(".")
            if not single or choice != single[1].upper():
                choice, cs = "?", 0.0
            else:
                # Case is immaterial for an MCQ letter. Require two agreeing reads, with
                # a majority token probability on the isolated glyph and retain the pair's score.
                cs = scores[pair.start(2)] if cs > .5 else 0.0
            results.append((label, id_score, choice, cs))
    return results
