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


def family_of(path: str) -> str:
    """The typeface family a font file belongs to, without weight/style suffixes."""
    stem = Path(path).stem
    return stem.split("-")[0].split("_")[0]


def split_fonts(holdout_families: int = 1) -> tuple[list[str], list[str]]:
    """Divide fonts into training and evaluation sets that share no family.

    Holding out one *file* is not enough: the first split held out DejaVuSerif while training
    on every other serif the system had, so the evaluation measured a typeface the model had
    already seen in another weight, and serif handling stayed broken in the shipped model.
    Families are the unit now.
    """
    fonts = available_fonts()
    if not fonts:
        return [], []
    by_family: dict[str, list[str]] = {}
    for font in fonts:
        by_family.setdefault(family_of(font), []).append(font)

    # Hold out the families that are least like a plain grotesque, because those are the ones
    # that fail: a display serif and a heavy text serif.
    ordered = sorted(by_family, key=lambda name: (
        "serif" not in name.lower(), "playfair" not in name.lower(), name.lower()))
    held_families = ordered[:holdout_families]
    held = [f for family in held_families for f in by_family[family]]
    train = [f for family, files in by_family.items() if family not in held_families for f in files]
    if not train:
        train, held = fonts[:-1], fonts[-1:]
    return train, held


@lru_cache(maxsize=64)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def render(ch: str, font_path: str, size: int, shift: tuple[int, int] = (0, 0),
           rotation: float = 0.0, binarize: bool = False) -> np.ndarray | None:
    """Render one character to a 28x28 ink-bright array, cropped tight like the photo path does.

    ``binarize`` reproduces what the page reader actually feeds the model. Photo intake masks
    the page against a local threshold and hands over a 0/255 image, so every glyph that arrives
    from a camera has hard edges and no anti-aliasing. Training only on smooth renders cost 11
    points on held-out fonts (90.1% -> 79.4%) for no reason other than that mismatch.
    """
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
    # 20 of 28, not the 86% that ``normaliseGlyph`` uses. Measured both ways: filling to 24px
    # gained 4 points when glyphs were fed to `recognize()` directly, but the browser's page
    # reader re-normalises every crop to 86% itself - so a glyph pre-scaled to 24 arrived
    # oversized and the page read worse ("HELLo WoRLD", "T°ML"). Training convention has to
    # match what the pipeline actually produces, not what looks right in isolation.
    scale = 20 / max(crop.width, crop.height)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                       Image.BILINEAR)
    tile = Image.new("L", (GLYPH, GLYPH), 255)
    tile.paste(crop, ((GLYPH - crop.width) // 2, (GLYPH - crop.height) // 2))
    out = np.asarray(tile)
    # The model consumes ink-bright glyphs: bright stroke on dark ground.
    ink = (255 - out).astype(np.uint8)
    if binarize:
        # Same cut the page's ink mask makes, applied after resampling - which is the order the
        # page uses, so the stroke edges the model sees here are the ones it will see in a photo.
        ink = np.where(ink > 110, 255, 0).astype(np.uint8)
    return ink


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
            # Half the samples hard-edged, to cover a masked photo; half smooth, to cover a
            # clean scan. A model trained on only one of the two loses 11 points on the other.
            glyph = render(ch, font_path, size, (dx, dy), rot,
                           binarize=bool(rng.random() < 0.5))
            if glyph is None or glyph.max() == 0:
                continue
            images.append(glyph)
            labels.append(index)
            made += 1
    if not images:
        return np.empty((0, GLYPH, GLYPH), np.uint8), np.empty(0, np.int64)
    return np.stack(images), np.asarray(labels, dtype=np.int64)
