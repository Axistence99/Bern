"""Tests for Bern's handwriting pipeline.

These cover the parts that do not need the 190 MB EMNIST download: feature shape,
layout analysis, the deskew alignment guarantee, and the public reader API. The
recognition accuracy itself is measured by ``python3 -m bern.handwriting --train``
and reported in ``models/handwriting_metrics.json``.
"""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from bern.handwriting import (
    GLYPH_SIZE,
    HandwritingReader,
    deskew,
    glyph_features,
    ink_mask,
    normalize_glyph,
    segment_glyphs,
    segment_lines,
    segment_words,
    to_gray,
)


def _page_with_lines(lines: list[str], height: int = 160, width: int = 420) -> np.ndarray:
    """Render dark text on a light page, the way a scan or phone photo looks."""
    page = np.full((height, width), 255, np.uint8)
    for index, text in enumerate(lines):
        cv2.putText(page, text, (20, 45 + index * 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3, cv2.LINE_AA)
    return page


def _word_mask(word: str) -> np.ndarray:
    """Binary mask of one printed word, isolating segment_glyphs from the full pipeline."""
    page = np.full((60, 300), 0, np.uint8)
    cv2.putText(page, word, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 3, cv2.LINE_AA)
    return page


# --------------------------------------------------------------------------------------
# Features and glyph normalisation
# --------------------------------------------------------------------------------------

def test_glyph_features_shape_and_dtype():
    features = glyph_features(np.zeros((GLYPH_SIZE, GLYPH_SIZE), np.uint8))
    assert features.shape == (341,), "341 = 324 HOG + 16 zoning + 1 ink ratio"
    assert features.dtype == np.float32


def test_glyph_features_accepts_float_input():
    """The reader hands over float crops in places, so the descriptor must cope."""
    rng = np.random.default_rng(0)
    assert glyph_features(rng.random((GLYPH_SIZE, GLYPH_SIZE)) * 255).shape == (341,)


def test_glyph_features_rescales_odd_sizes():
    assert glyph_features(np.zeros((37, 19), np.uint8)).shape == (341,)


def test_normalize_glyph_outputs_requested_size():
    glyph = np.zeros((40, 25), np.uint8)
    glyph[5:35, 8:18] = 255
    out = normalize_glyph(glyph)
    assert out.shape == (GLYPH_SIZE, GLYPH_SIZE)
    assert out.dtype == np.uint8


def test_normalize_glyph_preserves_aspect_ratio():
    """A tall glyph must stay tall: squashing it to fill the box would break recognition."""
    tall = np.zeros((56, 14), np.uint8)
    tall[2:54, 4:10] = 255
    out = normalize_glyph(tall)
    rows = np.nonzero((out > 8).any(axis=1))[0]
    cols = np.nonzero((out > 8).any(axis=0))[0]
    assert len(rows) > len(cols) * 1.8


def test_normalize_glyph_scales_to_emnist_extent():
    """EMNIST glyphs fill ~86% of the frame; that fraction was measured, not guessed."""
    square = np.zeros((40, 40), np.uint8)
    square[4:36, 4:36] = 255
    out = normalize_glyph(square)
    extent = int(np.count_nonzero((out > 8).any(axis=1)))
    assert 22 <= extent <= 25


def test_normalize_glyph_handles_blank_input():
    out = normalize_glyph(np.zeros((20, 20), np.uint8))
    assert out.shape == (GLYPH_SIZE, GLYPH_SIZE)
    assert out.max() == 0


# --------------------------------------------------------------------------------------
# Binarisation and layout
# --------------------------------------------------------------------------------------

def test_to_gray_accepts_bgr_and_gray():
    bgr = np.zeros((10, 10, 3), np.uint8)
    assert to_gray(bgr).shape == (10, 10)
    assert to_gray(np.zeros((10, 10), np.uint8)).shape == (10, 10)


def test_ink_mask_marks_dark_ink_as_foreground():
    page = _page_with_lines(["BERN"])
    mask = ink_mask(page)
    assert mask.max() == 255
    ink_fraction = (mask > 0).mean()
    assert 0.001 < ink_fraction < 0.5, "text on paper is a small minority of pixels"


def test_ink_mask_handles_inverted_page():
    """White text on a dark background must still come out as ink=255."""
    page = 255 - _page_with_lines(["BERN"])
    mask = ink_mask(page)
    assert (mask > 0).mean() < 0.5


def test_segment_lines_finds_both_lines():
    mask = ink_mask(_page_with_lines(["BERN", "BIGAS"]))
    assert len(segment_lines(mask)) == 2


def test_segment_lines_empty_page():
    assert segment_lines(np.zeros((100, 100), np.uint8)) == []


def test_segment_words_splits_on_spaces():
    mask = _word_mask("AB CD")
    assert len(segment_words(mask)) == 2


def test_segment_glyphs_finds_four_letters():
    mask = _word_mask("BERN")
    glyphs = segment_glyphs(mask)
    assert 3 <= len(glyphs) <= 5, "letter count is approximate because kerned letters can touch"


def test_segment_glyphs_boxes_lie_inside_the_word():
    mask = _word_mask("SALAMAT")
    height, width = mask.shape
    for x, y, w, h in segment_glyphs(mask):
        assert 0 <= x and 0 <= y
        assert x + w <= width and y + h <= height


# --------------------------------------------------------------------------------------
# Deskew must keep the mask and the image in lockstep
# --------------------------------------------------------------------------------------

def test_deskew_rotates_mask_and_parallel_together():
    """Regression test: the mask used to be rotated while the image was not, which
    shifted every crop box and cost ~15 points of accuracy."""
    mask = np.zeros((120, 240), np.uint8)
    cv2.putText(mask, "BERN", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 4, cv2.LINE_AA)
    matrix = cv2.getRotationMatrix2D((120, 60), 8.0, 1.0)
    mask = cv2.warpAffine(mask, matrix, (240, 120), flags=cv2.INTER_NEAREST, borderValue=0)

    parallel = mask.copy()
    rotated_mask, rotated_parallel = deskew(mask, parallel)

    assert rotated_mask.shape == rotated_parallel.shape == mask.shape

    # The two images are resampled with different interpolations (NEAREST keeps the mask
    # crisp, LINEAR keeps the photo smooth), so they will never be pixel-identical. What
    # must hold is that the ink sits in the same place in both.
    def centroid(image: np.ndarray) -> tuple[float, float]:
        ys, xs = np.nonzero(image > 0)
        return float(xs.mean()), float(ys.mean())

    mask_x, mask_y = centroid(rotated_mask)
    photo_x, photo_y = centroid(rotated_parallel)
    assert abs(mask_x - photo_x) < 2.0 and abs(mask_y - photo_y) < 2.0, "images drifted apart"

    # And the page must actually have been straightened.
    assert not np.array_equal(rotated_mask, mask)


def test_deskew_leaves_level_text_alone():
    mask = np.zeros((100, 300), np.uint8)
    cv2.putText(mask, "LEVEL", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 3, cv2.LINE_AA)
    before = mask.copy()
    assert np.array_equal(deskew(mask), before)


def test_deskew_returns_single_value_without_parallel():
    mask = np.zeros((100, 200), np.uint8)
    assert isinstance(deskew(mask), np.ndarray)


# --------------------------------------------------------------------------------------
# Public reader API
# --------------------------------------------------------------------------------------

def _encoded_page(text: str) -> bytes:
    page = _page_with_lines([text])
    ok, buffer = cv2.imencode(".png", page)
    assert ok
    return buffer.tobytes()


def test_reader_returns_expected_schema():
    from pathlib import Path

    default_model = Path(__file__).resolve().parents[1] / "models" / "handwriting_glyph_model.joblib"
    if not default_model.exists():
        pytest.skip("needs a trained model — run: python3 -m bern.handwriting")

    reader = HandwritingReader()
    result = reader.read_image(_encoded_page("BERN"))
    for key in ("text", "lines", "backend", "confidence", "glyph_count", "inference_ms"):
        assert key in result
    assert isinstance(result["text"], str)


def test_reader_rejects_garbage_bytes():
    from pathlib import Path

    default_model = Path(__file__).resolve().parents[1] / "models" / "handwriting_glyph_model.joblib"
    if not default_model.exists():
        pytest.skip("needs a trained model — run: python3 -m bern.handwriting")
    reader = HandwritingReader()
    with pytest.raises(ValueError):
        reader.read_image(b"this is not an image")


def test_reader_rejects_unknown_backend(tmp_path):
    from pathlib import Path

    default_model = Path(__file__).resolve().parents[1] / "models" / "handwriting_glyph_model.joblib"
    if not default_model.exists():
        pytest.skip("needs a trained model — run: python3 -m bern.handwriting")
    reader = HandwritingReader(backend="nonsense")
    with pytest.raises(ValueError):
        reader.read_image(_encoded_page("BERN"))


def test_reader_accepts_numpy_input():
    from pathlib import Path

    default_model = Path(__file__).resolve().parents[1] / "models" / "handwriting_glyph_model.joblib"
    if not default_model.exists():
        pytest.skip("needs a trained model — run: python3 -m bern.handwriting")
    result = HandwritingReader().read_image(_page_with_lines(["BERN"]))
    assert "text" in result
