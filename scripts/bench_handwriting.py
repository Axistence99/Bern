#!/usr/bin/env python3
"""Measure end-to-end handwriting accuracy on composed pages, including punctuation.

Composes pages from real EMNIST test glyphs (cropped to ink) plus synthetic marks, runs the
full reader, and reports character accuracy and mark accuracy. The reason a dedicated
harness exists: word and mark segmentation can only be measured on a whole page, never on a
single glyph, and the classifier's held-out score says nothing about them.

Usage:
    PYTHONPATH=. python3 scripts/bench_handwriting.py --data-dir /home/user/emnist_dl
    PYTHONPATH=. python3 scripts/bench_handwriting.py --data-dir ... --cases 40 --no-jitter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bern.handwriting import (  # noqa: E402
    HandwritingReader,
    iter_emnist_images,
    iter_emnist_labels,
    load_class_names,
)

WORDS = [
    "BIGAS", "PISO", "SALAMAT", "BERN", "GATAS", "TUBIG", "MANILA", "KANIN",
    "ISDA", "BULAK", "SINO", "MAHAL", "KAIN", "TINAPAY", "MAGANDANG",
    "UMAGA", "PAALAM", "MABUHAY", "KUMAIN", "TAYO",
]
MARKS = [".", ",", "-", "?"]


def build_pool(data_dir: Path, limit: int = 9000) -> dict[str, list[np.ndarray]]:
    """Collect EMNIST test glyphs by character so pages can be composed from real ink."""
    pool: dict[str, list[np.ndarray]] = {}
    images = data_dir / "byclass" / "emnist-byclass-test-images-idx3-ubyte.gz"
    labels = data_dir / "byclass" / "emnist-byclass-test-labels-idx1-ubyte.gz"
    classes = load_class_names(str(data_dir / "byclass" / "emnist-byclass-mapping.txt"))
    total = 0
    for batch_images, batch_labels in zip(iter_emnist_images(str(images), 8192),
                                          iter_emnist_labels(str(labels), 8192)):
        for image, label in zip(batch_images, batch_labels):
            pool.setdefault(classes[label], []).append(image)
            total += 1
        if total > limit:
            break
    return pool


def tight(glyph: np.ndarray) -> np.ndarray:
    """Crop a 28x28 EMNIST glyph to its ink.

    The margins around an EMNIST glyph are not part of the letter, and keeping them inflates
    every gap between letters well past what a pen produces — which then splits words.
    """
    rows, cols = np.nonzero(glyph > 8)
    if not len(rows):
        return glyph
    return glyph[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


#: A font family the symbol model was never trained on. Training uses DejaVu; scoring marks
#: with DejaVu would measure recall of the training set rather than the ability to read an
#: unfamiliar hand, which is what the benchmark is for.
MARK_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
]


def draw_mark(mark: str) -> tuple[np.ndarray, int] | None:
    """Render a mark with Pillow and return ``(glyph, lift)``, or ``None`` if unavailable.

    ``lift`` is how far the top of the mark sits above the baseline. Marks are rendered
    rather than hand-drawn because the point is to exercise the reader's mark handling, not
    to test which synthetic shape looks most convincing.
    """
    from PIL import Image, ImageDraw, ImageFont

    for font_path in MARK_FONTS:
        if not Path(font_path).exists():
            continue
        size = 44
        font = ImageFont.truetype(font_path, size)
        image = Image.new("L", (size * 3, size * 3), 0)
        ImageDraw.Draw(image).text((size // 2, size // 2), mark, font=font, fill=255)
        array = np.array(image)
        rows, cols = np.nonzero(array > 8)
        if not len(rows):
            continue
        glyph = array[rows.min():rows.max() + 1, cols.min():cols.max() + 1].astype(np.float32)
        # Place the mark so its bottom edge lands on the baseline, except a comma, which
        # hangs below it the way a written comma does.
        height = glyph.shape[0]
        lift = height if mark != "," else int(height * 0.7)
        # Ink-bright, like an EMNIST glyph: compose() does the inverting when it paints the
        # mark onto the page. Returning an already-inverted glyph paints it as a white gap.
        return (glyph, lift)
    return None


def compose(tokens, pool, rng, jitter: bool) -> np.ndarray:
    """Lay tokens onto a page, returning a BGR image."""
    baseline = 85
    cursor = 10
    placed: list[tuple[np.ndarray, int, int]] = []

    for kind, value in tokens:
        if kind == "space":
            cursor += 22
            continue
        if kind == "mark":
            drawn = draw_mark(value)
            if drawn is None:
                continue
            canvas, lift = drawn
            placed.append((canvas, cursor, baseline - lift))
            cursor += canvas.shape[1] + 7
            continue
        for character in value:
            glyph = tight(rng.choice(pool[character]).astype(np.float32))
            scale = rng.uniform(0.9, 1.15) if jitter else 1.0
            size = (max(6, int(glyph.shape[1] * scale)), max(6, int(glyph.shape[0] * scale)))
            glyph = cv2.resize(glyph, size, interpolation=cv2.INTER_AREA)
            if jitter:
                matrix = cv2.getRotationMatrix2D((glyph.shape[1] / 2, glyph.shape[0] / 2),
                                                 rng.uniform(-8, 8), 1)
                glyph = cv2.warpAffine(glyph, matrix, (glyph.shape[1], glyph.shape[0]))
            drift = int(rng.integers(-3, 4)) if jitter else 0
            placed.append((glyph, cursor, baseline - glyph.shape[0] + drift))
            cursor += glyph.shape[1] + (int(rng.integers(3, 6)) if jitter else 5)

    page = np.full((130, cursor + 15), 255, np.float32)
    for glyph, x, y in placed:
        region = page[y:y + glyph.shape[0], x:x + glyph.shape[1]]
        region[:] = np.minimum(region, 255.0 - glyph)

    page = np.clip(page, 0, 255).astype(np.uint8)
    if jitter:
        page = cv2.GaussianBlur(page, (3, 3), 0)
    return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


def edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Bern end-to-end on composed pages.")
    parser.add_argument("--data-dir", default="/home/user/emnist_dl")
    parser.add_argument("--cases", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-jitter", action="store_true")
    parser.add_argument("--no-punctuation", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"EMNIST not found at {data_dir}; see HANDOFF.md for the download step.")
        return 2

    pool = build_pool(data_dir)
    missing = {c for word in WORDS for c in word if c not in pool}
    if missing:
        print(f"missing glyph classes: {sorted(missing)}")
        return 2

    rng = np.random.default_rng(args.seed)
    reader = HandwritingReader()

    variants = [("clean", False)] if args.no_jitter else [("clean", False), ("jittered", True)]
    for label, jitter in variants:
        correct_chars = 0
        total_chars = 0
        exact_words = 0
        total_words = 0
        marks_expected = 0
        marks_found = 0

        for _ in range(args.cases):
            tokens: list[tuple[str, str | None]] = []
            expected_words: list[str] = []
            for word_index in range(int(rng.integers(1, 3))):
                word = WORDS[int(rng.integers(0, len(WORDS)))]
                if word_index:
                    tokens.append(("space", None))
                tokens.append(("word", word))
                mark = ""
                if not args.no_punctuation:
                    mark = MARKS[int(rng.integers(0, len(MARKS)))]
                    tokens.append(("mark", mark))
                expected_words.append(word + mark)

            expected = " ".join(expected_words)
            got = reader.read_image(compose(tokens, pool, rng, jitter))["text"]

            errors = edit_distance(expected, got)
            correct_chars += max(0, len(expected) - errors)
            total_chars += len(expected)

            got_words = got.split()
            for index, word in enumerate(expected_words):
                total_words += 1
                if index < len(got_words) and got_words[index] == word:
                    exact_words += 1
            for character in MARKS:
                marks_expected += expected.count(character)
                marks_found += got.count(character)

        char_accuracy = 100.0 * correct_chars / total_chars if total_chars else 0.0
        print(f"  {label:9s} chars={char_accuracy:5.1f}%  "
              f"exact words={exact_words}/{total_words}  "
              f"marks found={marks_found}/{marks_expected}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
