"""A photographed page is almost never level, and a tilted page loses its line structure.

Found from a real photo of a notebook: eleven lines of writing came back as a single
character. The reader finds lines by projecting ink onto rows, and five degrees is enough for
every line to smear across dozens of rows, so no row goes blank between two lines and the whole
page collapses into one band.

There are two readers and they are not the same code:

* ``bern/handwriting.py`` (server and phone backend) measures the tilt with
  ``cv2.minAreaRect`` and levels the mask. Verified: a 12-degree page comes back as 12.01
  degrees and keeps all four lines.
* ``web/demo/app.jsx`` (the browser) had no levelling at all and returned one character for a
  full page. It now estimates the tilt by asking at which angle the row profile is sharpest,
  and corrects it.

These tests pin the Python behaviour, which is already correct, and the fixture that reproduces
the browser failure. The browser's own estimator is checked against the page benchmark, since
it is JavaScript.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from bern import printed_glyphs as pg

PIL = pytest.importorskip("PIL.Image", reason="Pillow is needed to render a test page")
cv2 = pytest.importorskip("cv2", reason="OpenCV is needed for the reader")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

LINES = ["the quick brown fox", "jumps over the lazy", "dog and keeps on", "running through it"]


def _page(angle_degrees: float) -> np.ndarray:
    """Four lines of text on a white sheet, tilted by ``angle_degrees``."""
    font = ImageFont.truetype(pg.available_fonts()[0], 34)
    sheet = Image.new("L", (1400, 1000), 255)
    draw = ImageDraw.Draw(sheet)
    for index, line in enumerate(LINES):
        draw.text((300, 340 + index * 90), line, font=font, fill=0)
    if not angle_degrees:
        image = sheet
    else:
        # Nearest-neighbour keeps strokes hard-edged, the way a masked photo arrives.
        image = sheet.rotate(-angle_degrees, resample=Image.NEAREST, expand=True, fillcolor=255)
    array = np.asarray(image)
    height, width = array.shape
    return array[height // 2 - 210: height // 2 + 210, width // 2 - 400: width // 2 + 400]


def _row_bands(array: np.ndarray) -> int:
    """Bands of ink found by projecting onto rows, which is how both readers find lines."""
    ink = array < 128
    counts = ink.sum(axis=1)
    has_ink = counts > max(1, ink.shape[1] * 0.004)
    bands = 0
    was = False
    for flag in has_ink:
        if flag and not was:
            bands += 1
        was = bool(flag)
    return bands


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8)).convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


class TestTiltedPage:
    def test_a_level_page_separates_into_lines(self):
        assert _row_bands(_page(0)) >= 4

    def test_a_five_degree_tilt_is_still_readable(self):
        """Pinned because the severity was overstated once already.

        The browser's segmenter keeps all four lines at 5 degrees and only begins to merge them
        at 12. So "a slightly tilted photo returns one character" was wrong for a gentle tilt;
        the photo that produced one character was heavily tilted. Measured in the browser:
        5 degrees -> 4 lines, 12 degrees -> 2, 20 degrees -> 1.
        """
        assert _row_bands(_page(5)) >= 4

    @pytest.mark.parametrize("angle", [12, 20])
    def test_a_heavy_tilt_destroys_the_line_structure(self, angle):
        """The failure the de-skew exists for, reproduced without any reader involved."""
        assert _row_bands(_page(angle)) <= 2

    @pytest.mark.parametrize("angle", [5, 12, 20])
    def test_the_python_reader_levels_the_page_and_keeps_its_lines(self, angle):
        from bern import handwriting

        result = handwriting.HandwritingReader().read_image(_png(_page(angle)))
        assert len(result["lines"]) >= 3, f"{angle} degrees collapsed to {len(result['lines'])} lines"
        assert result["glyph_count"] > 40

    def test_the_tilt_is_measured_not_guessed(self):
        """`minAreaRect` on the ink cloud should land within a degree of the true angle."""
        from bern.portable_features import ink_mask

        page = _page(12).astype(np.uint8)
        coords = cv2.findNonZero((ink_mask(page) > 0).astype(np.uint8))
        (_, _), (_, _), angle = cv2.minAreaRect(coords)
        if angle > 45:
            angle -= 90
        assert abs(abs(angle) - 12) < 1.5, f"measured {angle:.2f} for a 12-degree page"


class TestLegibilityGate:
    """The reader must refuse input it cannot read instead of inventing text for it.

    This exists because it did the opposite. A screenshot of 9-pixel UI text produced 141
    characters of noise and the page stamped "74.1% measured accuracy" on it — a constant in
    the source, measured once on clean 216-glyph printed samples, presented as if it described
    whatever image had just been read.

    Measured across 12-48px text: every sample with a median glyph height of 11px or less was
    wrong on 100% of characters; 13px and above ran 48-66%. The gate is therefore median glyph
    height >= 12, and it is a hard boundary in the data rather than a tuned knob.
    """

    def test_the_threshold_sits_where_the_measurement_says(self):
        # Values measured on rendered text, kept here so a future change has to argue with them.
        measured = [
            (10, 100), (11, 100), (11, 100), (13, 65), (19, 66), (18, 55), (20, 60), (23, 54),
        ]
        readable = [h for h, cer in measured if h >= 12]
        refused = [h for h, cer in measured if h < 12]
        assert min(readable) == 13
        assert max(refused) == 11
        # Everything refused was completely wrong; nothing readable was.
        assert all(cer >= 90 for h, cer in measured if h < 12)
        assert all(cer <= 70 for h, cer in measured if h >= 12)

    def test_the_guard_code_matches_the_measurement(self):
        """The browser reader is JavaScript, so what can be checked from Python is that the
        constant it gates on is the one the measurement supports."""
        from pathlib import Path

        source = (Path(__file__).resolve().parent.parent / "web" / "demo" / "app.jsx").read_text()
        assert "tooSmall: medianHeight < 12" in source, "the legibility gate threshold moved"
        # The fake accuracy badge must stay gone.
        assert "reading_accuracy = 0.741" not in source
        assert 'caption="measured accuracy"' not in source

    def test_a_refused_reading_says_why(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parent.parent / "web" / "demo" / "app.jsx").read_text()
        assert "too small for me to read" in source
        assert "refused: true" in source
