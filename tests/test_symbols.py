"""Tests for punctuation geometry, the symbol model, and photo preprocessing.

These cover the Tier 1-2 additions: symbols the 62-class glyph model cannot emit, and the
photo repairs that only matter on real camera input.
"""

from __future__ import annotations

import numpy as np
import pytest

from bern.preprocess import (
    correct_perspective,
    detect_columns,
    order_corners,
    remove_ruled_lines,
)
from bern.symbols import SYMBOL_CLASSES, classify_punctuation


class TestClassifyPunctuation:
    """Geometry rules that decide whether a component is a mark or a letter."""

    def test_letter_sized_component_is_not_a_mark(self):
        # 20 px tall on a 30 px line at 0.50 reference: far too tall for punctuation.
        assert classify_punctuation(18, 20, 0, 30, 30, reference_height=30) is None

    def test_period_below_half_letter_height(self):
        assert classify_punctuation(8, 8, 22, 30, 30, reference_height=30) == "."

    def test_degenerate_geometry_returns_none(self):
        assert classify_punctuation(0, 0, 0, 0, 0, reference_height=0) is None

    def test_reference_height_decides_mark_versus_letter(self):
        # The same 12 px blob is a mark against 40 px letters but a letter-sized shape
        # against 20 px letters. This is the point of the reference height: the line box is
        # only the ink extent and can be far shorter than the letters it contains.
        assert classify_punctuation(10, 12, 26, 40, 40, reference_height=40) is not None
        assert classify_punctuation(10, 12, 6, 40, 40, reference_height=20) is None

    def test_wide_flat_component_is_a_dash(self):
        assert classify_punctuation(24, 4, 16, 30, 30, reference_height=30) == "-"

    def test_class_list_is_stable(self):
        assert len(SYMBOL_CLASSES) == len(set(SYMBOL_CLASSES))
        for char in ". , ? ! - : ; ( ) / + = % \u00f1 \u00d1".split():
            assert char in SYMBOL_CLASSES


class TestOrderCorners:
    def test_returns_top_left_first_clockwise(self):
        # Deliberately scrambled input order.
        points = np.array([[100, 100], [0, 0], [0, 100], [100, 0]], dtype=np.float32)
        ordered = order_corners(points)
        assert tuple(ordered[0]) == (0.0, 0.0)      # top-left
        assert tuple(ordered[2]) == (100.0, 100.0)  # bottom-right


class TestCorrectPerspective:
    def test_flat_image_without_page_outline_is_untouched(self):
        gray = np.full((200, 300), 255, np.uint8)
        gray[80:120, 100:200] = 0
        out, corrected = correct_perspective(gray)
        assert corrected is False
        assert out.shape == gray.shape
        assert np.array_equal(out, gray)

    def test_tilted_page_is_straightened(self):
        # White page on a dark background, drawn as a trapezoid.
        canvas = np.zeros((400, 400), np.uint8)
        page = np.array([[80, 60], [340, 30], [370, 350], [50, 330]], np.int32)
        import cv2

        cv2.fillPoly(canvas, [page], 255)
        out, corrected = correct_perspective(canvas)
        assert corrected is True
        # The warped page fills the frame, so it is overwhelmingly white again.
        assert (out > 127).mean() > 0.9

    def test_rejects_non_grayscale(self):
        with pytest.raises(ValueError):
            correct_perspective(np.zeros((10, 10, 3), np.uint8))


class TestRemoveRuledLines:
    def test_long_rule_is_removed_and_ink_kept(self):
        mask = np.zeros((120, 400), np.uint8)
        mask[60, :] = 255              # ruled line across the page
        mask[30:50, 100:140] = 255     # a block of "writing"
        cleaned, removed = remove_ruled_lines(mask)
        assert removed >= 1
        assert cleaned[60, :].sum() == 0
        assert cleaned[30:50, 100:140].sum() > 0

    def test_abandons_when_most_ink_is_long_runs(self):
        # Nearly everything is a long run: these are strokes, not rules.
        mask = np.zeros((20, 400), np.uint8)
        mask[:, :] = 255
        cleaned, removed = remove_ruled_lines(mask)
        assert removed == 0
        assert np.array_equal(cleaned, mask)

    def test_no_rules_returns_input(self):
        mask = np.zeros((60, 200), np.uint8)
        mask[20:40, 20:60] = 255
        cleaned, removed = remove_ruled_lines(mask)
        assert removed == 0
        assert np.array_equal(cleaned, mask)


class TestDetectColumns:
    def test_single_line_is_one_column(self):
        import cv2

        mask = np.zeros((80, 500), np.uint8)
        cv2.putText(mask, "BIGAS PISO", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
        assert detect_columns(mask) == [(0, 500)]

    def test_two_column_page_splits(self):
        import cv2

        mask = np.zeros((300, 600), np.uint8)
        for i in range(5):
            cv2.putText(mask, f"LEFT{i}", (20, 45 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
            cv2.putText(mask, f"RIGHT{i}", (400, 45 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
        columns = detect_columns(mask)
        assert len(columns) == 2
        assert columns[0][0] < columns[1][0]
        # The gutter separates them and both keep real width.
        assert columns[0][1] < columns[1][0]

    def test_empty_mask_is_safe(self):
        assert detect_columns(np.zeros((100, 100), np.uint8)) == [(0, 100)]

    def test_tiny_mask_is_safe(self):
        assert detect_columns(np.zeros((10, 20), np.uint8)) == [(0, 20)]
