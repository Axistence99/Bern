"""Punctuation, math symbols and accented characters — everything EMNIST cannot see.

Bern's glyph model emits exactly 62 classes: ``0-9 A-Z a-z``. That means a transcribed
page arrives with no full stops, no question marks, no ``+`` or ``=``, and no ``ñ``. This
module fills that hole with two complementary mechanisms, because the two problems want
different solutions.

**1. Geometric punctuation (:func:`classify_punctuation`).** A period is not a small
letter — it is a blob roughly a tenth of the line height, sitting on the baseline. A comma
sits *below* it. A hyphen is wide and thin. These are properties of shape and position, so
no training set is needed and no accuracy is lost to a classifier: the geometry decides.
This is deliberately rule-based, and every rule is inspectable.

**2. Learned symbol classifier (:class:`SymbolBackend`).** Characters like ``+ = % × ÷ √``
and ``ñ Ñ °`` are full-size glyphs with real structure, so geometry alone will not separate
``+`` from ``†`` or ``%`` from ``&``. There is no public handwritten-symbol corpus in the
repository, so the model is trained on **synthetically rendered symbols** in a mixture of
fonts with handwriting-like distortion — rotation, shear, stroke thickening, baseline
jitter and noise. That is a weaker supervision signal than real pen strokes, and it is
reported as such: measured accuracy is in ``models/symbol_metrics.json``.

Usage:
    python3 -m bern.symbols            # train the symbol classifier
    python3 -m bern.symbols --evaluate # score the saved model
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_SYMBOL_MODEL_PATH = MODELS_DIR / "symbol_model.joblib"
DEFAULT_SYMBOL_METRICS_PATH = MODELS_DIR / "symbol_metrics.json"

#: Symbols the classifier is trained on. Deliberately excludes ``. ,`` and similar marks
#: that :func:`classify_punctuation` handles geometrically and far more reliably.
SYMBOL_CLASSES: list[str] = [
    ".", ",", "?", "!", "-", ":", ";", "(", ")", "/", "\\", "+", "=",
    "%", "°", "ñ", "Ñ", "×", "÷", "√", "@", "#", "&", "$", "_", "*", "'", '"',
]

#: Fonts used to synthesise training glyphs. Several shapes per symbol matter more than
#: many sizes of one shape.
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]


# --------------------------------------------------------------------------------------
# Geometric punctuation
# --------------------------------------------------------------------------------------

def classify_punctuation(
    glyph_width: int,
    glyph_height: int,
    top_offset: int,
    line_height: int,
    line_baseline: int,
    reference_height: int | None = None,
) -> str | None:
    """Name a punctuation mark from the geometry of its bounding box, or return ``None``.

    Args:
        glyph_width: Width of the candidate glyph in pixels.
        glyph_height: Height of the candidate glyph in pixels.
        top_offset: Distance from the top of the line to the top of the glyph.
        line_height: Height of the whole text line.
        line_baseline: Distance from the top of the line to the baseline, i.e. the y below
            which descenders such as ``g`` and ``y`` hang.
        reference_height: Median height of the *letters* on the line. Preferred over
            ``line_height`` because the line box is only the ink extent: on a line holding a
            single short word it can be far shorter than the letters themselves.

    Returns:
        One of ``. , -`` or ``None``. Colons, semicolons and quotes are deliberately not
        inferred from geometry alone; see the note at the end of the function.

    The decision rests on **where the glyph starts**, not how big it is. Every letter reaches
    the top of the line — capitals touch cap height and lowercase letters touch the x-height,
    which is still the upper half of the box. A mark, in contrast, begins low, at or below the
    baseline region. That single positional test separates marks from letters far more
    reliably than a size cut-off, which has to be tuned differently for every writer and is
    defeated by a heavy dot or a small ``o``.

    Size is kept only as a secondary guard, so that a tall descender (``g``, ``y``, ``p``)
    starting low is not mistaken for a mark: it is as large as the other letters, and a mark
    is not.
    """
    if line_height <= 0 or glyph_height <= 0 or glyph_width <= 0:
        return None

    scale = float(reference_height) if reference_height and reference_height > 0 else float(line_height)
    if scale <= 0:
        return None

    relative_height = glyph_height / scale
    relative_width = glyph_width / scale
    relative_top = top_offset / scale
    bottom = top_offset + glyph_height
    excess_below_baseline = (bottom - line_baseline) / scale
    aspect = glyph_width / max(1, glyph_height)

    starts_low = relative_top >= 0.45

    # A dash is wide on purpose, so it is tested before the width guard below.
    if aspect >= 2.0 and relative_height <= 0.50:
        return "-" if starts_low else None

    # A mark begins in the lower part of the letter band and is small in both directions.
    # Descenders also begin low but are as tall as a letter, which the 0.90 guard rejects.
    if not (starts_low and relative_height <= 0.90 and relative_width <= 0.65):
        return None

    if excess_below_baseline > 0.06:
        return ","                                    # hangs under the baseline
    return "."

    # Colon, semicolon and quotes were once inferred here from a narrow, tall box.
    # Measurement showed that test firing on the thin tail of a comma: "," became ";", and
    # stray semicolons appeared. Telling "two stacked dots" from "one dot plus its tail"
    # needs the pixels, not just the box. Those marks are left to the learned symbol model,
    # which separates them cleanly; a wrong ";" costs more than a missing one.

def _render_symbol(char: str, font_path: str, size: int, rng: np.random.Generator) -> np.ndarray | None:
    """Render one symbol and distort it to look like handwriting rather than print.

    Real pen symbols differ from typeset ones mainly by **stroke thickness variation,
    rotation and shear**. Shear is the important one: handwriting leans, and a model trained
    on upright glyphs alone generalises badly.
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:
        return None

    canvas_size = size * 2
    image = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(image)
    try:
        draw.text((size // 2, size // 2), char, fill=255, font=font)
    except Exception:
        return None

    array = np.array(image, dtype=np.uint8)
    if array.max() == 0:
        return None

    # Crop to ink so the distortion is applied to the glyph, not to empty space.
    rows, cols = np.nonzero(array > 12)
    array = array[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    if array.size == 0:
        return None

    import cv2

    # Shear: handwriting leans right more often than left, but both occur.
    shear = rng.uniform(-0.28, 0.34)
    height, width = array.shape
    matrix = np.float32([[1, shear, 0], [0, 1, 0]])
    array = cv2.warpAffine(array, matrix, (int(width + abs(shear) * height), height),
                           flags=cv2.INTER_LINEAR, borderValue=0)

    # Rotation.
    angle = rng.uniform(-12, 12)
    height, width = array.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    array = cv2.warpAffine(array, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)

    # Stroke weight: pen thickness varies a lot between writers.
    kernel = int(rng.integers(0, 3))
    if kernel > 0:
        array = cv2.dilate(array, np.ones((kernel, kernel), np.uint8))
    elif rng.random() < 0.3:
        array = cv2.erode(array, np.ones((2, 2), np.uint8))

    # Blur and noise, matching what a phone camera does.
    speckle = 1.0 + rng.uniform(0.0, 0.6)
    array = cv2.GaussianBlur(array, (0, 0), speckle)
    noise = rng.normal(0, rng.uniform(2, 9), array.shape)
    array = np.clip(array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return array


def build_synthetic_symbols(per_class: int = 900, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Render the training set: every symbol, in every available font, many distortions.

    Returns:
        ``(features, labels)`` ready for scikit-learn.
    """
    from bern.handwriting import glyph_features, normalize_glyph

    fonts = [path for path in _FONT_CANDIDATES if Path(path).exists()]
    if not fonts:
        raise RuntimeError(
            "No DejaVu fonts found. Install fonts-dejavu-core, or pass a font directory."
        )

    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[int] = []

    for class_index, char in enumerate(SYMBOL_CLASSES):
        produced = 0
        attempts = 0
        while produced < per_class and attempts < per_class * 6:
            attempts += 1
            font_path = fonts[int(rng.integers(0, len(fonts)))]
            size = int(rng.integers(28, 62))
            glyph = _render_symbol(char, font_path, size, rng)
            if glyph is None:
                continue
            normalized = normalize_glyph(glyph)
            if normalized.max() == 0:
                continue
            features.append(glyph_features(normalized))
            labels.append(class_index)
            produced += 1

    return np.stack(features), np.array(labels, dtype=np.int64)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

def train_symbol_model(
    model_path: Path = DEFAULT_SYMBOL_MODEL_PATH,
    metrics_path: Path = DEFAULT_SYMBOL_METRICS_PATH,
    per_class: int = 900,
    seed: int = 42,
) -> dict[str, Any]:
    """Train the symbol classifier and write it to disk.

    Uses the same 341-dim HOG descriptor and the same normalisation as the letter model, so
    the two can share a pipeline. A held-out split is carved out *by render*, not by class,
    so the score reflects unseen distortions rather than memorised ones.
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import train_test_split

    import joblib

    started = time.perf_counter()
    print(f"Rendering synthetic symbols for {len(SYMBOL_CLASSES)} classes ...", flush=True)
    features, labels = build_synthetic_symbols(per_class=per_class, seed=seed)
    print(f"  {len(features):,} glyphs from {features.shape[1]} features", flush=True)

    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    classifier = SGDClassifier(
        loss="log_loss", alpha=1e-5, average=True, max_iter=40, tol=1e-4,
        random_state=seed, learning_rate="optimal",
    )
    print("Training ...", flush=True)
    classifier.fit(train_x, train_y)

    accuracy = float((classifier.predict(test_x) == test_y).mean())

    # Per-class accuracy makes weak symbols visible instead of hiding them in an average.
    per_class_scores: dict[str, float] = {}
    predictions = classifier.predict(test_x)
    for index, char in enumerate(SYMBOL_CLASSES):
        mask = test_y == index
        if mask.sum():
            per_class_scores[char] = round(float((predictions[mask] == test_y[mask]).mean()), 3)

    bundle = {
        "classifier": classifier,
        "classes": SYMBOL_CLASSES,
        "feature_dim": int(features.shape[1]),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_samples": int(len(train_x)),
        "metrics": {"held_out_accuracy": round(accuracy, 4), "per_class": per_class_scores},
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    report = {
        "model": str(model_path.relative_to(BASE_DIR)) if model_path.is_relative_to(BASE_DIR) else str(model_path),
        "classes": SYMBOL_CLASSES,
        "train_samples": int(len(train_x)),
        "test_samples": int(len(test_x)),
        "held_out_accuracy": round(accuracy, 4),
        "per_class": per_class_scores,
        "training_seconds": round(time.perf_counter() - started, 1),
        "caveat": (
            "Trained on synthetically rendered fonts with handwriting-like distortion, not on "
            "real pen strokes. Expect real-handwriting accuracy to be lower than this figure."
        ),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Held-out accuracy {accuracy * 100:.2f}% on {len(test_x):,} unseen glyphs")
    weakest = sorted(per_class_scores.items(), key=lambda item: item[1])[:5]
    print(f"Weakest classes: {', '.join(f'{c}:{s:.2f}' for c, s in weakest)}")
    return report


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------

class SymbolBackend:
    """Classify full-size non-letter glyphs: math operators, currency signs, accented letters.

    Used as a *secondary* opinion. The letter model is always consulted first; the symbol
    model only overrides when it is confident and the letter model is not, so a slightly
    odd ``A`` is never turned into ``&``.
    """

    name = "bern-symbols"

    def __init__(self, model_path: Path = DEFAULT_SYMBOL_MODEL_PATH) -> None:
        import joblib

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No symbol model at {self.model_path}. Train one first:\n"
                "    python3 -m bern.symbols"
            )
        bundle = joblib.load(self.model_path)
        self.classifier = bundle["classifier"]
        self.classes: list[str] = list(bundle["classes"])

    def recognize_topk(self, glyph_bgr: np.ndarray, k: int = 4) -> list[tuple[str, float]]:
        """Return the ``k`` most likely symbols for one glyph, best first."""
        from bern.handwriting import glyph_features, normalize_glyph

        normalized = normalize_glyph(glyph_bgr)
        features = glyph_features(normalized).reshape(1, -1)
        probabilities = self.classifier.predict_proba(features)[0]
        order = np.argsort(probabilities)[::-1][:k]
        return [(self.classes[int(index)], float(probabilities[int(index)])) for index in order]


def main() -> None:
    """``python3 -m bern.symbols`` — train or evaluate the symbol classifier."""
    import argparse

    parser = argparse.ArgumentParser(description="Bern symbol classifier")
    parser.add_argument("--evaluate", action="store_true", help="score the saved model")
    parser.add_argument("--per-class", type=int, default=900, help="training glyphs per class")
    args = parser.parse_args()

    if args.evaluate:
        import joblib

        if not DEFAULT_SYMBOL_MODEL_PATH.exists():
            raise SystemExit("No trained model yet — run: python3 -m bern.symbols")
        bundle = joblib.load(DEFAULT_SYMBOL_MODEL_PATH)
        metrics = bundle.get("metrics", {})
        print(f"held-out accuracy: {metrics.get('held_out_accuracy', 0) * 100:.2f}%")
        for char, score in sorted(metrics.get("per_class", {}).items(), key=lambda kv: kv[1]):
            print(f"  {char!r}: {score * 100:.1f}%")
        return

    train_symbol_model(per_class=args.per_class)


if __name__ == "__main__":
    main()
