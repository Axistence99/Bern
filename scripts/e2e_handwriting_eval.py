"""End-to-end handwriting accuracy harness.

Character accuracy on EMNIST measures the classifier alone. This measures the *whole*
pipeline — segmentation included — by composing real EMNIST glyphs into jittered lines,
photographing them the way a phone would, and scoring what comes back out.

That distinction matters: the classifier scores 82.55% on isolated glyphs, but the full
pipeline scores ~75% clean and ~68% jittered, because segmentation and normalisation
introduce their own errors.

Usage:
    python3 scripts/e2e_handwriting_eval.py --data-dir /path/to/emnist
    python3 scripts/e2e_handwriting_eval.py --data-dir /path/to/emnist --words HELLO BERN

The expected layout under --data-dir is ``byclass/emnist-byclass-*-idx*-ubyte.gz``,
which is what ``bern.handwriting.ensure_emnist()`` produces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bern.handwriting import (  # noqa: E402
    HandwritingReader,
    iter_emnist_images,
    iter_emnist_labels,
    load_class_names,
)

DEFAULT_WORDS = [
    "HELLO", "BERN", "SALAMAT", "PISO", "KANIN", "BIGAS", "ISDA", "PRICE", "DATA",
    "MANILA", "ADOBE", "GATAS", "TUBIG", "SAGING", "MANGGA", "TAO", "BARYA", "INIT",
    "ULAM", "GULAY",
]


def build_glyph_pool(data_dir: Path, limit: int = 8000) -> dict[str, list[np.ndarray]]:
    """Collect real test-split glyphs, keyed by character, for composing test lines."""
    split = data_dir / "byclass"
    classes = load_class_names(split / "emnist-byclass-mapping.txt")
    pool: dict[str, list[np.ndarray]] = {}
    images = iter_emnist_images(split / "emnist-byclass-test-images-idx3-ubyte.gz", 8192)
    labels = iter_emnist_labels(split / "emnist-byclass-test-labels-idx1-ubyte.gz", 8192)
    for batch_images, batch_labels in zip(images, labels):
        for image, label in zip(batch_images, batch_labels):
            pool.setdefault(classes[label], []).append(image)
        if sum(len(v) for v in pool.values()) > limit:
            break
    return pool


def compose_line(text: str, pool: dict[str, list[np.ndarray]], rng: np.random.Generator,
                 jitter: bool) -> np.ndarray:
    """Render one line of handwriting by pasting real glyphs onto a light page.

    Glyphs are drawn as ``page = min(page, 255 - glyph)`` so the glyph's own antialiasing
    is preserved — the same soft edges the classifier was trained on.
    """
    placed: list[tuple[np.ndarray, int, int]] = []
    cursor, baseline = 8, 60
    for char in text:
        if char == " ":
            cursor += 18
            continue
        glyph = rng.choice(pool[char]).astype(np.float32)
        size = int(rng.integers(26, 32)) if jitter else 28
        glyph = cv2.resize(glyph, (size, size), interpolation=cv2.INTER_AREA)
        if jitter:
            matrix = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(-9, 9), 1)
            glyph = cv2.warpAffine(glyph, matrix, (size, size))
        y = baseline - size + (int(rng.integers(-4, 5)) if jitter else 0)
        placed.append((glyph, cursor, y))
        cursor += size + (int(rng.integers(2, 5)) if jitter else 4)

    page = np.full((110, cursor + 16), 255, np.float32)
    for glyph, x, y in placed:
        region = page[y:y + glyph.shape[0], x:x + glyph.shape[1]]
        region[:] = np.minimum(region, 255.0 - glyph)
    if jitter:
        page = cv2.GaussianBlur(page, (3, 3), 0)
    return cv2.cvtColor(np.clip(page, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="EMNIST root containing the byclass/ directory")
    parser.add_argument("--words", nargs="*", default=DEFAULT_WORDS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show-misses", action="store_true", help="print every mismatch")
    args = parser.parse_args()

    pool = build_glyph_pool(args.data_dir)
    reader = HandwritingReader()

    for jitter in (False, True):
        rng = np.random.default_rng(args.seed)
        exact = correct_chars = total_chars = 0
        misses: list[str] = []
        for word in args.words:
            image = compose_line(word, pool, rng, jitter)
            got = reader.read_image(image)["text"].replace(" ", "")
            exact += got == word
            correct_chars += sum(a == b for a, b in zip(word, got))
            total_chars += len(word)
            if got != word:
                misses.append(f"{word}->{got}")

        print(f"jitter={str(jitter):5s}  char={correct_chars / total_chars * 100:5.1f}%  "
              f"exact={exact}/{len(args.words)}")
        if args.show_misses:
            print("   " + ", ".join(misses))


if __name__ == "__main__":
    main()
