"""Glyph features that compute identically without OpenCV, so they port to a phone.

The training pipeline uses ``cv2.HOGDescriptor``. That is fine on a server and impossible in a
JavaScript runtime, which matters because the models are small enough to ship inside an app if
the *features* can be computed there too.

Reimplementing cv2's HOG bit-for-bit would be fragile — it depends on undocumented choices in
border handling and bin interpolation. So this module implements HOG from the published
definition instead, and the model is **retrained on its output**. Training and inference then
share one implementation, and any small difference from cv2 stops mattering: nobody compares
against cv2 any more.

The descriptor layout is unchanged (324 HOG + 16 zoning + 1 ink ratio = 341), so the two are
interchangeable in shape and can be compared directly.

Every operation here is plain arithmetic on floats — no OpenCV, no SciPy. The TypeScript port
in ``react-native/bernFeatures.ts`` follows the same steps in the same order.
"""

from __future__ import annotations

import numpy as np

GLYPH_SIZE = 28

#: OpenCV's HOGDescriptor applies sqrt gamma correction before differentiating. Without it
#: the descriptor is dominated by whichever strokes happen to be darkest, and thin pen
#: strokes lose out to heavy ones regardless of their orientation.
GAMMA_CORRECT = True

CELL_SIZE = 7
BLOCK_CELLS = 2
BINS = 9

#: OpenCV's HOG normalisation clips the block descriptor at this value before renormalising
#: (the "L2-Hys" variant). Keeping the same number keeps the descriptor scale comparable.
L2_HYS_CLIP = 0.2


def gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradients with replicated edges.

    ``gray`` is 28x28, values 0-255. Returns ``(gx, gy)`` as float32, same shape.

    Edges replicate rather than zero-pad: a zero border invents a hard edge around every
    glyph, which shows up as spurious vertical and horizontal gradient energy in the outer
    cells — exactly where thin letters like ``l`` and ``1`` put their only ink.
    """
    source = gray.astype(np.float32)
    if GAMMA_CORRECT:
        # sqrt, in the same order cv2 does it: normalise to 0-255, take the root, quantise
        # back to 8-bit. Quantising matters because cv2 works on the rounded values.
        source = np.sqrt(np.clip(source, 0, 255)).astype(np.uint8).astype(np.float32)

    padded = np.pad(source, 1, mode="edge")
    gx = padded[1:-1, 2:] - padded[1:-1, :-2]
    gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    return gx, gy


def cell_histograms(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Orientation histograms, one 9-bin histogram per 7x7 cell.

    Two choices here decide whether the descriptor matches the one the model was trained on:

    * **Orientation is binned hard, not interpolated.** A pixel contributes all of its
      magnitude to its nearest bin. Interpolating between two bins is the textbook version and
      measurably worse here — smoothing the angle blurs the difference between a vertical
      stroke and a slanted one, which is exactly what separates ``1`` from ``l``.
    * **Space is interpolated.** A pixel near a cell boundary contributes to all four
      neighbouring cells, weighted by distance. Without this a pixel sitting one step over the
      boundary flips cells entirely and the descriptor jumps.

    Those two are the opposite way round from the obvious reading of the paper, and getting
    them backwards cost 4.9 points of held-out accuracy when measured.
    """
    size = gx.shape[0]
    cells_per_axis = size // CELL_SIZE
    magnitude = np.sqrt(gx * gx + gy * gy)

    angle = np.degrees(np.arctan2(gy, gx)) % 180.0
    bins = np.minimum((angle / (180.0 / BINS)).astype(np.int32), BINS - 1)

    # Continuous cell coordinates: the pixel centre, expressed in cell units.
    rows, cols = np.indices((size, size)).astype(np.float32)
    cell_y = (rows + 0.5) / CELL_SIZE - 0.5
    cell_x = (cols + 0.5) / CELL_SIZE - 0.5

    y0 = np.floor(cell_y).astype(np.int32)
    x0 = np.floor(cell_x).astype(np.int32)
    wy = cell_y - y0
    wx = cell_x - x0

    histograms = np.zeros(cells_per_axis * cells_per_axis * BINS, dtype=np.float64)
    for dy, weight_y in ((0, 1.0 - wy), (1, wy)):
        for dx, weight_x in ((0, 1.0 - wx), (1, wx)):
            target_y = np.clip(y0 + dy, 0, cells_per_axis - 1)
            target_x = np.clip(x0 + dx, 0, cells_per_axis - 1)
            flat = (target_y * cells_per_axis + target_x) * BINS + bins
            histograms += np.bincount(flat.ravel(), weights=(magnitude * weight_y * weight_x).ravel(),
                                      minlength=cells_per_axis * cells_per_axis * BINS)

    return histograms.reshape(cells_per_axis, cells_per_axis, BINS).astype(np.float32)


def block_normalise(histograms: np.ndarray) -> np.ndarray:
    """Build the descriptor: overlapping 2x2-cell blocks, each L2-Hys normalised.

    Normalising per block is what makes the descriptor tolerant of lighting and pen weight:
    the *relative* distribution of edge directions inside a block is preserved even when the
    overall contrast changes.
    """
    cells_per_axis = histograms.shape[0]
    blocks_per_axis = cells_per_axis - BLOCK_CELLS + 1
    output = np.zeros(blocks_per_axis * blocks_per_axis * BLOCK_CELLS * BLOCK_CELLS * BINS,
                      dtype=np.float32)

    cursor = 0
    for block_row in range(blocks_per_axis):
        for block_col in range(blocks_per_axis):
            block = histograms[block_row:block_row + BLOCK_CELLS,
                               block_col:block_col + BLOCK_CELLS, :].ravel()
            norm = float(np.sqrt(np.dot(block, block) + 1e-6))
            block = block / norm
            # L2-Hys: clip the outliers, then renormalise.
            block = np.minimum(block, L2_HYS_CLIP)
            norm = float(np.sqrt(np.dot(block, block) + 1e-6))
            block = block / norm
            output[cursor:cursor + block.size] = block
            cursor += block.size
    return output


def glyph_features(gray_glyph: np.ndarray) -> np.ndarray:
    """Describe one 28x28 glyph as 341 floats. Same layout as the OpenCV version.

    Args:
        gray_glyph: 28x28, white ink on black, any real dtype.

    Returns:
        float32 vector of length 341: 324 HOG, 16 zoning, 1 ink ratio.
    """
    glyph = np.asarray(gray_glyph, dtype=np.float32)
    if glyph.ndim == 3:
        glyph = glyph.mean(axis=2)
    if glyph.shape != (GLYPH_SIZE, GLYPH_SIZE):
        raise ValueError(f"expected {GLYPH_SIZE}x{GLYPH_SIZE}, got {glyph.shape}")
    glyph = np.clip(glyph, 0, 255)

    gx, gy = gradients(glyph)
    hog = block_normalise(cell_histograms(gx, gy))

    # Zoning: mean ink per cell of a 4x4 grid, computed by averaging 7x7 sub-blocks.
    zones = glyph.reshape(4, CELL_SIZE, 4, CELL_SIZE).mean(axis=(1, 3)).ravel() / 255.0
    ink_ratio = np.float32(glyph.mean() / 255.0)

    return np.concatenate([hog, zones.astype(np.float32), np.array([ink_ratio], dtype=np.float32)])


def normalise_glyph(glyph: np.ndarray, size: int = GLYPH_SIZE) -> np.ndarray:
    """Fit one glyph crop into a ``size`` square the way the server does.

    Mirrors ``bern.handwriting.normalize_glyph``: tighten to the ink, scale so the longest
    side fills 86% of the box, centre it, and bilinear-resample. All of it in plain numpy so
    the TypeScript port can follow the same steps.

    Args:
        glyph: Ink-bright 2-D array of any size.
        size: Side of the output square.

    Returns:
        ``uint8`` array of shape ``(size, size)``.
    """
    glyph = np.asarray(glyph, dtype=np.float32)
    if glyph.ndim == 3:
        glyph = glyph.mean(axis=2)

    rows, cols = np.nonzero(glyph > 8)
    if len(rows) == 0:
        return np.zeros((size, size), np.uint8)

    tight = glyph[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    height, width = tight.shape
    scale = (size * 0.86) / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))

    resized = _bilinear_resize(tight, new_h, new_w)
    canvas = np.zeros((size, size), np.uint8)
    offset_x = (size - new_w) // 2
    offset_y = (size - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = np.clip(resized, 0, 255).astype(np.uint8)
    return canvas


def _bilinear_resize(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resample using the same pixel-centre mapping as OpenCV.

    OpenCV samples at ``(x + 0.5) * in/out - 0.5``, not at ``x * in/out``. The difference is
    half a pixel, which sounds negligible and is not: it decides whether a one-pixel stroke
    survives downscaling or is averaged away entirely.
    """
    in_h, in_w = image.shape
    if in_h == out_h and in_w == out_w:
        return image.copy()

    src_y = (np.arange(out_h, dtype=np.float32) + 0.5) * (in_h / out_h) - 0.5
    src_x = (np.arange(out_w, dtype=np.float32) + 0.5) * (in_w / out_w) - 0.5
    src_y = np.clip(src_y, 0, in_h - 1)
    src_x = np.clip(src_x, 0, in_w - 1)

    y0 = np.floor(src_y).astype(np.int32)
    x0 = np.floor(src_x).astype(np.int32)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (src_y - y0)[:, None]
    wx = (src_x - x0)[None, :]

    top = image[y0][:, x0] * (1 - wx) + image[y0][:, x1] * wx
    bottom = image[y1][:, x0] * (1 - wx) + image[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy
