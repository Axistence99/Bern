"""End-to-end handwriting accuracy harness for English, Filipino and Taglish.

Character accuracy on EMNIST measures the classifier alone. This measures the *whole*
pipeline — segmentation, normalisation and dictionary correction included — by composing
real EMNIST glyphs into jittered lines, photographing them the way a phone would, and
scoring what comes back out.

The corpus is split by language on purpose. Filipino and Taglish are Latin-script
languages that EMNIST has never heard of as *words*, so the only thing that makes them
recognisable is the lexicon. Reporting one blended number would hide whether that works.

Usage:
    python3 scripts/e2e_handwriting_eval.py --data-dir ~/emnist_dl
    python3 scripts/e2e_handwriting_eval.py --data-dir ~/emnist_dl --show-misses
    python3 scripts/e2e_handwriting_eval.py --data-dir ~/emnist_dl --no-lexicon
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

# Word-level corpora. Multi-word entries test the line/word segmentation as well.
CORPORA: dict[str, list[str]] = {
    "english": [
        "HELLO", "PRICE", "DATA", "MANILA", "WATER", "STUDY", "NOTES", "QUIZ",
        "EXAM", "LECTURE", "CHAPTER", "REVIEW", "SCIENCE", "FORMULA", "PROTEIN",
    ],
    "filipino": [
        "PISO", "BIGAS", "GATAS", "TUBIG", "SAGING", "MANGGA", "SALAMAT", "MAGKANO",
        "PRESYO", "ADOBO", "BAWANG", "SIBUYAS", "KANIN", "ISDA", "GULAY", "MALUNGGAY",
    ],
    "taglish": [
        "MAGKANO ANG BIGAS", "SALAMAT PO", "PAANO MAGLUTO NG ADOBO",
        "ILANG CALORIES NG KANIN", "PRESYO NG GATAS", "READ MY NOTES",
        "QUIZ SA SCIENCE", "CLASS SA UMAGA", "BUDGET KO NGAYON", "TUBIG AT KAPE",
    ],
}


def build_glyph_pool(data_dir: Path, limit: int = 9000) -> dict[str, list[np.ndarray]]:
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
            cursor += 20
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


def score_corpus(words: list[str], pool: dict[str, list[np.ndarray]], reader: HandwritingReader,
                 rng: np.random.Generator, jitter: bool) -> dict[str, object]:
    """Run one corpus through the reader and return accuracy plus every mistake."""
    exact = 0
    correct_chars = total_chars = 0
    misses: list[str] = []
    for word in words:
        image = compose_line(word, pool, rng, jitter)
        got = reader.read_image(image)["text"].strip()
        got_flat = got.replace(" ", "")
        target_flat = word.replace(" ", "")
        exact += got_flat == target_flat
        correct_chars += sum(a == b for a, b in zip(target_flat, got_flat))
        total_chars += len(target_flat)
        if got_flat != target_flat:
            misses.append(f"{word!r}->{got!r}")
    return {
        "exact": exact,
        "total": len(words),
        "char_accuracy": round(correct_chars / total_chars * 100, 1) if total_chars else 0.0,
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="EMNIST root containing the byclass/ directory")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show-misses", action="store_true", help="print every mismatch")
    parser.add_argument("--no-lexicon", action="store_true",
                        help="disable dictionary correction, for a before/after comparison")
    args = parser.parse_args()

    pool = build_glyph_pool(args.data_dir)
    reader = HandwritingReader(use_lexicon=not args.no_lexicon)

    label = "raw model (no dictionary)" if args.no_lexicon else "with dictionary correction"
    print(f"== {label} ==")
    for jitter in (False, True):
        variant = "jittered" if jitter else "clean   "
        for language, words in CORPORA.items():
            rng = np.random.default_rng(args.seed)
            result = score_corpus(words, pool, reader, rng, jitter)
            print(f"  {variant}  {language:9s} char={result['char_accuracy']:5.1f}%  "
                  f"exact={result['exact']}/{result['total']}")
            if args.show_misses and result["misses"]:
                for miss in result["misses"]:
                    print(f"              {miss}")


if __name__ == "__main__":
    main()
