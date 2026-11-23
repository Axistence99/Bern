"""Render printed characters so the reader is not blind to the text that actually appears in photos.

The on-device glyph model is trained on EMNIST, which is **handwriting**. The page reader, though,
is pointed at photographs of printed text: labels, receipts, textbooks, signboards. Measured on
288 rendered characters in four fonts, the model read 32.6% of them - and its errors were shapes
that barely occur in handwriting at all: R read as B, A as 8, X and Y as 8, every one of them
repeated across fonts and sizes.

Handwriting and print are different distributions, and no amount of tuning a handwriting
classifier fixes that. What does fix it is showing the model both, which is what this module
produces: the same characters rendered across fonts, weights, sizes and small geometric
distortions, in the exact 28x28 ink-bright form the model already consumes.

Fonts used for training must not be used for evaluation, or the measured gain is memorisation of
a typeface rather than reading.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GLYPH = 28

#: The 62 classes the model emits. Print adds no new ones.
CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

#: Faces that are structurally different from one another: a sturdy serif, a grotesque, a
#: monospace, a humanist sans. Training on variety is the point; more of the same font is not.
TRAIN_FONT_HINTS = (
    "DejaVuSerif", "DejaVuSans", "DejaVuSansMono", "LiberationSerif",
    "LiberationSans", "FreeSerif", "FreeSans", "NotoSans", "NotoSerif",
)


def available_fonts() -> list[str]:
    """Every font file the system has, in a stable order."""
    try:
        out = subprocess.run(["fc-list", "--format", "%{file}\n"], capture_output=True,
                             text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    files = sorted({line.strip() for line in out.splitlines()
                    if line.strip().endswith((".ttf", ".otf"))})
    return files


def split_fonts(holdout: int = 1) -> tuple[list[str], list[str]]:
    """Divide available fonts into a training set and a *disjoint* evaluation set."""
    fonts = available_fonts()
    if not fonts:
        return [], []
    preferred = [f for f in fonts if any(h.lower() in Path(f).stem.lower() for h in TRAIN_FONT_HINTS)]
    train = preferred or fonts
    rest = [f for f in fonts if f not in train]
    held = rest[:holdout] if rest else [train[-1]]
    if not rest:
        train = train[:-1]
    return train, held


@lru_cache(maxsize=64)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def render(ch: str, font_path: str, size: int, shift: tuple[int, int] = (0, 0),
           rotation: float = 0.0) -> np.ndarray | None:
    """Render one character to a 28x28 ink-bright array, cropped tight like the photo path does."""
    canvas = Image.new("L", (size * 3, size * 3), 255)
    draw = ImageDraw.Draw(canvas)
    try:
        draw.text((size + shift[0], size + shift[1]), ch, font=_font(font_path, size), fill=0)
    except (OSError, ValueError):
        return None
    if rotation:
        canvas = canvas.rotate(rotation, resample=Image.BILINEAR, fillcolor=255)
    arr = np.asarray(canvas)
    ys, xs = np.where(arr < 128)
    if not len(ys):
        return None
    crop = Image.fromarray(arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    # Preserve aspect ratio inside a 20x20 box, centred, as the model's normalisation expects.
    scale = 20 / max(crop.width, crop.height)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                       Image.BILINEAR)
    tile = Image.new("L", (GLYPH, GLYPH), 255)
    tile.paste(crop, ((GLYPH - crop.width) // 2, (GLYPH - crop.height) // 2))
    out = np.asarray(tile)
    # The model consumes ink-bright glyphs: bright stroke on dark ground.
    return (255 - out).astype(np.uint8)


def sample_dataset(fonts: list[str], per_class: int = 40, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Render ``per_class`` variants of every class across the given fonts.

    Returns ``(images, class_indices)`` where the index is into :data:`CLASSES`.
    """
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []
    for index, ch in enumerate(CLASSES):
        made = 0
        attempts = 0
        while made < per_class and attempts < per_class * 6:
            attempts += 1
            font_path = fonts[int(rng.integers(0, len(fonts)))] if fonts else None
            if font_path is None:
                break
            size = int(rng.integers(34, 72))
            dx, dy = int(rng.integers(-3, 4)), int(rng.integers(-3, 4))
            rot = float(rng.normal(0, 4))
            glyph = render(ch, font_path, size, (dx, dy), rot)
            if glyph is None or glyph.max() == 0:
                continue
            images.append(glyph)
            labels.append(index)
            made += 1
    if not images:
        return np.empty((0, GLYPH, GLYPH), np.uint8), np.empty(0, np.int64)
    return np.stack(images), np.asarray(labels, dtype=np.int64)
