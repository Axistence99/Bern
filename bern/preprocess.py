"""Preprocessing for real photographs: perspective, ruled paper, and columns.

Everything in this module addresses a failure mode that only appears on a **photo**.
Composed test images are flat, level and unruled; a phone photo is none of those, and the
segmentation in :mod:`bern.handwriting` assumes all three. Each function states the defect
it removes and stays deliberately conservative: when a correction cannot be detected with
confidence it does nothing, because a wrong straightening is worse than a crooked page.

All functions take and return grayscale (or binary) arrays and have no side effects.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Beyond this many degrees a rotation is a sideways photo, not a tilt.
MAX_DESKEW_DEGREES = 20.0


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    Perspective warping needs a consistent corner order. Sorting by angle around the
    centroid is stable for convex quadrilaterals and avoids the classic "sum and difference"
    hack that fails on strongly rotated pages.
    """
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    ordered = points[np.argsort(angles)]
    # argsort by angle starts anywhere; rotate so the top-left corner comes first.
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def correct_perspective(gray: np.ndarray, min_area_ratio: float = 0.25) -> tuple[np.ndarray, bool]:
    """Straighten a page photographed at an angle.

    A phone held over a notebook produces a quadrilateral, not a rectangle. The character
    normaliser rescales each glyph to a square, so a trapezoid stretch makes letters in one
    corner 30% taller than those in another — enough to lose the classifier.

    The page outline is found as the largest bright quadrilateral. The correction is applied
    only if that quadrilateral covers at least ``min_area_ratio`` of the frame, which rejects
    the common case of a photo where no page edge is visible.

    Returns:
        ``(image, corrected)``. ``corrected`` is False when no confident page outline was
        found and the input is returned untouched.
    """
    if gray.ndim != 2:
        raise ValueError("correct_perspective expects a grayscale image")

    height, width = gray.shape
    # Blur hard: we want the page edge, not the handwriting near it.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray, False

    frame_area = float(height * width)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(contour)
        if area < frame_area * min_area_ratio:
            continue
        perimeter = cv2.arcLength(contour, True)
        for epsilon in (0.02, 0.03, 0.05):
            polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(polygon) != 4:
                continue
            corners = order_corners(polygon)
            top_left, top_right, bottom_right, bottom_left = corners

            target_width = int(max(
                np.linalg.norm(bottom_right - bottom_left),
                np.linalg.norm(top_right - top_left),
            ))
            target_height = int(max(
                np.linalg.norm(top_right - bottom_right),
                np.linalg.norm(top_left - bottom_left),
            ))
            if target_width < 40 or target_height < 40:
                continue

            destination = np.float32([[0, 0], [target_width, 0],
                                      [target_width, target_height], [0, target_height]])
            matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
            warped = cv2.warpPerspective(gray, matrix, (target_width, target_height),
                                         flags=cv2.INTER_LINEAR, borderValue=255)
            return warped, True

    return gray, False


def remove_ruled_lines(binary_mask: np.ndarray, min_length_ratio: float = 0.55) -> tuple[np.ndarray, int]:
    """Erase notebook rules from a binary ink mask, keeping the handwriting.

    The distinction is length: a rule spans most of the page width, while a letter does not.
    An opening with a horizontal kernel longer than ``min_length_ratio`` of the mask width
    extracts only the long horizontal runs, and subtracting them removes the rules.

    Vertical margin lines are handled the same way with a vertical kernel. Both removals are
    guarded: if they would delete more than 35% of the ink, the mask is probably cursive
    writing with long horizontal strokes and the operation is abandoned.

    Args:
        binary_mask: White-ink-on-black mask, as produced by ``ink_mask``.
        min_length_ratio: Minimum fraction of the width a run must span to count as a rule.

    Returns:
        ``(cleaned_mask, lines_removed)``.
    """
    if binary_mask.ndim != 2:
        raise ValueError("remove_ruled_lines expects a single-channel mask")

    height, width = binary_mask.shape
    horizontal_length = max(40, int(width * min_length_ratio))
    vertical_length = max(40, int(height * min_length_ratio))

    horizontal = cv2.morphologyEx(
        binary_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_length, 1))
    )
    vertical = cv2.morphologyEx(
        binary_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length))
    )
    rules = cv2.bitwise_or(horizontal, vertical)

    if not rules.any():
        return binary_mask, 0

    ink_total = float((binary_mask > 0).sum())
    rule_total = float((rules > 0).sum())
    if rule_total > ink_total * 0.35:
        # Too much of the page would disappear: these are strokes, not rules.
        return binary_mask, 0

    cleaned = cv2.bitwise_and(binary_mask, cv2.bitwise_not(rules))
    contours, _ = cv2.findContours(rules, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cleaned, len(contours)


def _count_lines(binary_mask: np.ndarray) -> int:
    """Count text lines by horizontal projection.

    A local helper rather than an import from :mod:`bern.handwriting`, which imports this
    module; going the other way would be circular.
    """
    if binary_mask.ndim != 2 or not binary_mask.any():
        return 0
    rows = (binary_mask > 0).sum(axis=1)
    is_ink = rows > max(1, binary_mask.shape[1] * 0.005)
    lines = 0
    previous = False
    for current in is_ink:
        if current and not previous:
            lines += 1
        previous = bool(current)
    return lines


def detect_columns(binary_mask: np.ndarray) -> list[tuple[int, int]]:
    """Find vertical column bands in a page mask.

    Two-column lecture notes are common, and naive top-to-bottom segmentation reads across
    the gutter — producing lines made of the left column's word fragment plus the right
    column's, which is worse than useless.

    A gutter must be empty over the **full vertical extent of the ink**, which is why the
    mask is first cropped to its ink bounding box. Testing the raw page instead makes the
    white space above and below a single line of text look like a gutter, and a one-column
    page gets split in two.

    Word spaces are the other trap: a generous gap between two words on a single line is
    wider than any tuned constant, so width alone cannot separate a gutter from a space.
    Two structural facts can, and both are used here:

    * A column layout has several lines. One line of text is never two columns, so the page
      must yield at least three lines before columns are considered at all.
    * A gutter is a large outlier among the page's horizontal gaps, not merely a wide one.
      It must reach 2.5x the median gap width.

    Returns:
        ``[(x_start, x_end), ...]`` left to right. A single full-width span is returned when
        no gutter is present.
    """
    if binary_mask.ndim != 2 or binary_mask.size == 0:
        return []

    height, width = binary_mask.shape
    if height < 40 or width < 80:
        return [(0, width)]

    # Work only within the band that actually contains ink.
    rows = np.nonzero((binary_mask > 0).any(axis=1))[0]
    if rows.size == 0:
        return [(0, width)]
    band = binary_mask[rows.min():rows.max() + 1, :]
    band_height = band.shape[0]

    # A single line of text cannot be a two-column layout.
    if _count_lines(band) < 3:
        return [(0, width)]

    column_ink = (band > 0).sum(axis=0)
    empty = column_ink <= max(1, band_height * 0.005)
    total_ink = float(column_ink.sum())

    spans: list[tuple[int, int]] = []
    start: int | None = None
    for x, is_empty in enumerate(empty):
        if not is_empty and start is None:
            start = x
        elif is_empty and start is not None:
            spans.append((start, x))
            start = None
    if start is not None:
        spans.append((start, width))
    if not spans:
        return [(0, width)]

    # Join spans separated by an ordinary word space. The threshold is derived from this
    # page's own gap distribution rather than a constant, because handwriting scale varies.
    gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
    median_gap = float(np.median(gaps)) if gaps else 0.0
    gutter = max(16, int(median_gap * 2.5))

    merged = [list(spans[0])]
    for begin, end in spans[1:]:
        if begin - merged[-1][1] < gutter:
            merged[-1][1] = end
        else:
            merged.append([begin, end])

    if len(merged) < 2:
        return [(0, width)]

    columns = [(int(a), int(b)) for a, b in merged
               if (column_ink[a:b].sum() / total_ink if total_ink else 0) >= 0.15]
    return columns if len(columns) > 1 else [(0, width)]
