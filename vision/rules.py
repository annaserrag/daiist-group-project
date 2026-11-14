"""
The rule baselines (no learning), shared by train_detector.py and the reader.

They answer "how far do simple image rules get without any training?", the same role the
trivial baseline had in Assignment 2. Their few parameters were chosen on the Roboflow
validation pages, never on the Mendeley exams.
"""

from __future__ import annotations

import cv2
import numpy as np


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Pen strokes only: a local (adaptive) threshold that ignores faint watermarks and paper
    texture, then the ruled lines and isolated specks removed. Works for any pen colour."""
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 25)
    h, w = gray.shape
    ruled = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 6, 1), 1)))
    ruled |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 6, 1))))
    ink = cv2.subtract(ink, cv2.dilate(ruled, np.ones((3, 3), np.uint8)))
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def rule_lines(gray: np.ndarray) -> list[np.ndarray]:
    """Rows with ink, grouped into bands; each band's ink extent is one line (x0, y0, x1, y1)."""
    ink = ink_mask(gray)
    rows = np.convolve((ink > 0).sum(axis=1), np.ones(5) / 5, mode="same")
    floor = 0.15 * np.percentile(rows[rows > 0], 90) if (rows > 0).any() else 3.0
    on = rows > max(3.0, floor)
    boxes, start = [], None
    for y, value in enumerate(list(on) + [False]):
        if value and start is None:
            start = y
        elif not value and start is not None:
            if y - start >= 6:
                cols = np.where((ink[start:y] > 0).any(axis=0))[0]
                if len(cols):
                    boxes.append(np.array([cols.min(), start, cols.max(), y], float))
            start = None
    return boxes


def rule_question_numbers(gray: np.ndarray) -> list[np.ndarray]:
    """The first word of each line, if it starts between 10% and 30% of the page width and is narrow:
    a question number is usually the first thing on its line, just right of the margin."""
    h, w = gray.shape
    merged = cv2.dilate(ink_mask(gray), np.ones((3, 5), np.uint8))  # letters of a word touch
    _, _, stats, _ = cv2.connectedComponentsWithStats(merged)
    words = [s for s in stats[1:] if s[4] > 30 and 6 <= s[3] <= 0.12 * h and s[2] < 0.5 * w]
    out = []
    for x, y, bw, bh, _ in words:
        if not (0.10 * w <= x <= 0.30 * w and bw < 0.12 * w):
            continue
        has_word_to_the_left = any(
            x2 + bw2 <= x and x2 >= 0.10 * w and min(y + bh, y2 + bh2) - max(y, y2) > 0.5 * min(bh, bh2)
            for x2, y2, bw2, bh2, _ in words)
        if not has_word_to_the_left:
            out.append(np.array([x, y, x + bw, y + bh], float))
    return out
