"""Bern's handwriting-to-text module.

Turns a photo or scan of handwritten text into plain text. The module is deliberately
split into two halves that can be used separately:

1. **Document layout analysis** (:func:`segment_page`) — pure OpenCV. Finds the ink,
   straightens the page, then cuts it into lines, words and individual character
   images. This half is backend-independent and is the part that decides whether
   recognition succeeds on a given photo.

2. **Recognition backends** (:class:`GlyphBackend`, :class:`TrocrBackend`,
   :class:`TesseractBackend`) — each one maps a cropped image to characters.

The default backend is Bern's own ``GlyphBackend``: a scikit-learn model trained on
the EMNIST *byclass* corpus (697,932 real handwritten characters, 62 classes). It
ships with the project, needs no GPU, no network and no extra dependency beyond
scikit-learn/OpenCV, and is what ``python3 -m bern.handwriting`` builds.

``GlyphBackend`` expects **hand-printed** writing — block capitals, separated
letters, reasonably clean paper. It segments a page into single glyphs and classifies
them one by one, so connected cursive defeats it: segmentation cannot split a word
that is written without lifting the pen. For cursive, use :class:`TrocrBackend`
(``pip install transformers torch pillow``, ~1.3 GB download, CPU inference is
possible but slow). The backend is chosen automatically by :class:`HandwritingReader`
unless you name one explicitly.

Public API
----------
    >>> from bern.handwriting import HandwritingReader
    >>> reader = HandwritingReader()
    >>> reader.read_image(open("note.jpg", "rb").read())["text"]
    'HELLO WORLD'

Command line
------------
    python3 -m bern.handwriting              # train the glyph model (downloads EMNIST)
    python3 -m bern.handwriting --evaluate   # score an existing model on the held-out split
"""

from __future__ import annotations

import argparse
import gzip
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import joblib
import numpy as np

from bern.ph_lexicon import decode_word, get_lexicon
from bern.preprocess import correct_perspective, detect_columns, remove_ruled_lines
from bern.symbols import classify_punctuation

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_HANDWRITING_MODEL_PATH = MODELS_DIR / "handwriting_glyph_model.joblib"
DEFAULT_METRICS_PATH = MODELS_DIR / "handwriting_metrics.json"
EMNIST_DIR = BASE_DIR / "data" / "external" / "emnist"

#: Official NIST mirror of the EMNIST distribution (contains all six splits).
EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

#: Side length, in pixels, that every segmented glyph is resized to before feature extraction.
GLYPH_SIZE = 28

#: Character classes that ``GlyphBackend`` can never output because EMNIST has no samples for them.
_PUNCTUATION_ONLY = frozenset(['.', ',', ':', ';', '-', chr(39), chr(34), '?', '!'])
EMNIST_UNSUPPORTED = {".": "period", ",": "comma", "?": "question mark", "!": "exclamation mark",
                      "-": "hyphen", "(": "open paren", ")": "close paren", "/": "slash", ";": "semicolon",
                      ":": "colon", "@": "at sign", "&": "ampersand", "\"": "double quote", "\u2019": "apostrophe"}


# --------------------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------------------

# A single reusable HOG descriptor. Computed on a 28x28 grayscale glyph this yields
# 324 values: 3x3 blocks of 2x2 cells, 9 orientation bins each.
_HOG = cv2.HOGDescriptor(
    _winSize=(GLYPH_SIZE, GLYPH_SIZE),
    _blockSize=(14, 14),
    _blockStride=(7, 7),
    _cellSize=(7, 7),
    _nbins=9,
)


def glyph_features(gray_glyph: np.ndarray) -> np.ndarray:
    """Describe one 28x28 grayscale glyph as a fixed-length float32 vector.

    The descriptor is the concatenation of:

    * **HOG** (324 values) — captures stroke direction, which is what separates
      ``o`` from ``0`` or ``n`` from ``h`` far better than raw pixels do.
    * **4x4 zoning** (16 values) — mean ink per quadrant-grid cell, a cheap
      "is there ink in the top-left?" signal that survives small crops.
    * **Ink ratio** (1 value) — total ink / total area, which distinguishes dense
      glyphs (``8``, ``B``) from sparse ones (``1``, ``l``).

    Args:
        gray_glyph: 28x28 single-channel image, white ink on black background.

    Returns:
        float32 array of length 341.
    """
    if gray_glyph.ndim == 3:
        gray_glyph = cv2.cvtColor(gray_glyph, cv2.COLOR_BGR2GRAY)
    if gray_glyph.shape != (GLYPH_SIZE, GLYPH_SIZE):
        gray_glyph = cv2.resize(gray_glyph, (GLYPH_SIZE, GLYPH_SIZE), interpolation=cv2.INTER_AREA)

    # cv2's HOG descriptor refuses anything but 8-bit input, and normalize_glyph hands
    # back float32 (it needs the precision while centroid-shifting).
    if gray_glyph.dtype != np.uint8:
        gray_glyph = np.clip(gray_glyph, 0, 255).astype(np.uint8)

    hog = _HOG.compute(gray_glyph).ravel().astype(np.float32)

    # Zoning: average ink inside each cell of a 4x4 grid.
    cells = cv2.resize(gray_glyph, (4, 4), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0

    ink_ratio = np.float32(gray_glyph.mean() / 255.0)
    return np.concatenate([hog, cells, np.array([ink_ratio], dtype=np.float32)])


# --------------------------------------------------------------------------------------
# EMNIST (IDX) reading
# --------------------------------------------------------------------------------------

def _read_idx_header(stream: Any) -> tuple[int, ...]:
    """Parse the 4-byte magic number of an IDX file and return the remaining dimensions.

    IDX is the file format NIST uses for MNIST/EMNIST: a big-endian magic number whose
    low two bytes are the data type and the number of dimensions, followed by one
    big-endian int32 per dimension.
    """
    raw = stream.read(4)
    if len(raw) < 4:
        raise ValueError("Truncated IDX file: could not read the magic number")
    magic = int.from_bytes(raw, "big")
    ndim = magic & 0xFF
    return tuple(int.from_bytes(stream.read(4), "big") for _ in range(ndim))


def iter_emnist_images(path: Path, chunk: int = 8192) -> Iterator[np.ndarray]:
    """Stream an EMNIST image file as upright ``(n, 28, 28)`` uint8 batches.

    The iterator is *out-of-core*: it never holds more than ``chunk`` images in memory,
    which is what lets the trainer fit on a machine with 1 GB of RAM even though the
    full training split is 697,932 images.

    EMNIST stores every image transposed relative to how it is displayed, so each batch
    is transposed back here — forgetting this step yields sideways characters that the
    model can never learn.
    """
    with gzip.open(path, "rb") as fh:
        n, rows, cols = _read_idx_header(fh)
        for start in range(0, n, chunk):
            take = min(chunk, n - start)
            buf = fh.read(take * rows * cols)
            if len(buf) < take * rows * cols:
                raise ValueError(f"Truncated EMNIST image file: {path}")
            batch = np.frombuffer(buf, dtype=np.uint8).reshape(take, rows, cols)
            yield np.transpose(batch, (0, 2, 1))


def iter_emnist_labels(path: Path, chunk: int = 8192) -> Iterator[np.ndarray]:
    """Stream an EMNIST label file as uint8 class-index batches (0-61 for *byclass*)."""
    with gzip.open(path, "rb") as fh:
        (n,) = _read_idx_header(fh)
        for start in range(0, n, chunk):
            take = min(chunk, n - start)
            buf = fh.read(take)
            if len(buf) < take:
                raise ValueError(f"Truncated EMNIST label file: {path}")
            yield np.frombuffer(buf, dtype=np.uint8)


def load_class_names(mapping_path: Path) -> list[str]:
    """Read ``emnist-byclass-mapping.txt`` into a class-index -> character list.

    Each line is "<class index> <ASCII code>". Painting apps and OCR engines think in
    characters, scikit-learn thinks in integers, and this file is the bridge.
    """
    names: dict[int, str] = {}
    with open(mapping_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                names[int(parts[0])] = chr(int(parts[1]))
    return [names[i] for i in range(len(names))]


def ensure_emnist(dataset_dir: Path = EMNIST_DIR) -> Path:
    """Download and extract the EMNIST *byclass* split if it is not already on disk.

    Only the four byclass files are extracted; the rest of the 536 MB archive is
    discarded immediately so the checkout does not grow by half a gigabyte.

    Returns:
        Path to the directory holding the extracted ``emnist-byclass-*`` files.
    """
    target = dataset_dir / "byclass"
    required = [
        target / "emnist-byclass-train-images-idx3-ubyte.gz",
        target / "emnist-byclass-train-labels-idx1-ubyte.gz",
        target / "emnist-byclass-test-images-idx3-ubyte.gz",
        target / "emnist-byclass-test-labels-idx1-ubyte.gz",
        target / "emnist-byclass-mapping.txt",
    ]
    if all(p.exists() for p in required):
        return target

    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive = dataset_dir / "gzip.zip"
    if not archive.exists():
        print(f"Downloading EMNIST (~536 MB) from {EMNIST_URL} ...", flush=True)
        # The NIST host answers 403 to the default urllib User-Agent, so identify
        # ourselves explicitly and stream to disk instead of buffering half a gigabyte.
        request = urllib.request.Request(EMNIST_URL, headers={"User-Agent": "bern-handwriting/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, open(archive, "wb") as out:  # noqa: S310
            while chunk := response.read(1 << 20):
                out.write(chunk)

    print("Extracting the byclass split ...", flush=True)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if "emnist-byclass" not in member:
                continue
            name = Path(member).name
            with zf.open(member) as src, open(target / name, "wb") as dst:
                dst.write(src.read())
    archive.unlink(missing_ok=True)
    return target


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

def _iter_feature_batches(images_path: Path, labels_path: Path, chunk: int = 4096) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(features, labels)`` batches so training never materialises the full matrix."""
    for images, labels in zip(iter_emnist_images(images_path, chunk), iter_emnist_labels(labels_path, chunk)):
        feats = np.stack([glyph_features(img) for img in images])
        yield feats, labels.astype(np.int64)


def evaluate_glyph_model(model: dict[str, Any], images_path: Path, labels_path: Path,
                         max_samples: int | None = None, chunk: int = 4096) -> dict[str, Any]:
    """Score a trained bundle on a labelled EMNIST split and return accuracy statistics."""
    clf = model["classifier"]
    correct = total = 0
    confusion: dict[str, int] = {}
    t0 = time.perf_counter()
    for feats, labels in _iter_feature_batches(images_path, labels_path, chunk):
        if max_samples is not None and total + len(labels) > max_samples:
            feats, labels = feats[: max_samples - total], labels[: max_samples - total]
        preds = clf.predict(feats)
        correct += int((preds == labels).sum())
        total += len(labels)
        for truth, pred in zip(labels, preds):
            if truth != pred:
                key = f"{model['classes'][truth]}->{model['classes'][pred]}"
                confusion[key] = confusion.get(key, 0) + 1
        if max_samples is not None and total >= max_samples:
            break
    top_confusions = sorted(confusion.items(), key=lambda kv: kv[1], reverse=True)[:15]
    return {
        "samples": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "top_confusions": [{"pair": k, "count": v} for k, v in top_confusions],
    }


def train_glyph_model(
    dataset_dir: Path = EMNIST_DIR,
    model_path: Path = DEFAULT_HANDWRITING_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    epochs: int = 5,
    max_train: int | None = None,
    max_test: int | None = 20_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Train Bern's handwritten-character classifier and persist it to ``model_path``.

    Uses :class:`sklearn.linear_model.SGDClassifier` with the log-loss objective, fed
    batch by batch through ``partial_fit``. That is the only reason this fits on modest
    hardware: a one-shot ``LogisticRegression.fit`` on 697k x 341 floats would need
    roughly 950 MB for the design matrix alone.

    Args:
        dataset_dir: Where the EMNIST archive should live / already lives.
        model_path: Destination ``.joblib`` bundle.
        metrics_path: Destination ``.json`` metrics report.
        epochs: Passes over the training split. Five measured best in tuning; beyond
            that the linear boundary starts to chase noise.
        max_train: Optional cap on training rows (useful for a quick smoke run).
        max_test: Cap on rows used for the held-out evaluation (``None`` = all 116,323).
        seed: Shuffling seed, so runs are reproducible.

    Returns:
        The metrics dictionary that was written to ``metrics_path``.
    """
    from sklearn.linear_model import SGDClassifier

    split_dir = ensure_emnist(dataset_dir)
    train_images = split_dir / "emnist-byclass-train-images-idx3-ubyte.gz"
    train_labels = split_dir / "emnist-byclass-train-labels-idx1-ubyte.gz"
    test_images = split_dir / "emnist-byclass-test-images-idx3-ubyte.gz"
    test_labels = split_dir / "emnist-byclass-test-labels-idx1-ubyte.gz"
    classes = load_class_names(split_dir / "emnist-byclass-mapping.txt")

    # implicit: log-loss + softmax gives calibrated probabilities, so the confidence we
    # report to the caller is meaningful rather than a raw margin.
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        average=True,        # averaged SGD: the weights are the running mean, which is
                             # markedly steadier than the final noisy iterate on this task
        max_iter=1,          # one pass per partial_fit call; epochs are driven below
        tol=None,
        random_state=seed,
        learning_rate="optimal",
    )

    rng = np.random.default_rng(seed)
    all_classes = np.arange(len(classes))
    started = time.perf_counter()
    seen = 0

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        for feats, labels in _iter_feature_batches(train_images, train_labels):
            if max_train is not None:
                remaining = max_train - seen
                if remaining <= 0:
                    break
                feats, labels = feats[:remaining], labels[:remaining]
            order = rng.permutation(len(labels))
            feats, labels = feats[order], labels[order]
            # `classes` must be supplied on the very first call, otherwise partial_fit
            # believes it is seeing a single-class problem and locks that in forever.
            clf.partial_fit(feats, labels, classes=all_classes if seen == 0 else None)
            seen += len(labels)
        print(f"epoch {epoch + 1}/{epochs}: {seen:,} glyphs in {time.perf_counter() - epoch_start:.1f}s", flush=True)
        if max_train is not None and seen >= max_train:
            break

    model: dict[str, Any] = {
        "classifier": clf,
        "classes": classes,
        "glyph_size": GLYPH_SIZE,
        "feature_dim": int(glyph_features(np.zeros((GLYPH_SIZE, GLYPH_SIZE), np.uint8)).shape[0]),
        "dataset": "EMNIST byclass",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_samples": int(seen),
        "epochs": epochs,
    }

    print("Evaluating on the held-out EMNIST test split ...", flush=True)
    evaluation = evaluate_glyph_model(model, test_images, test_labels, max_samples=max_test)
    model["metrics"] = evaluation

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    report = {
        "model": str(model_path.relative_to(BASE_DIR)) if model_path.is_relative_to(BASE_DIR) else str(model_path),
        "trained_at": model["trained_at"],
        "train_samples": model["train_samples"],
        "epochs": epochs,
        "classes": len(classes),
        "feature_dim": model["feature_dim"],
        "training_seconds": round(time.perf_counter() - started, 1),
        "held_out": evaluation,
        "notes": (
            "Character-level accuracy on EMNIST byclass. Includes the ambiguous case pairs "
            "(c/C, o/O, s/S, v/V, w/W, x/X, z/Z, and 1/l/I) that EMNIST scores as separate classes; "
            "flagged ones would raise the number."
        ),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {model_path} — held-out accuracy {evaluation['accuracy'] * 100:.2f}% "
          f"on {evaluation['samples']:,} unseen characters", flush=True)
    return report


# --------------------------------------------------------------------------------------
# Document layout analysis (backend independent)
# --------------------------------------------------------------------------------------

def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a BGR/BGRA/grayscale array to single-channel grayscale."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Threshold a grayscale page into a binary mask where handwriting is white (255).

    Otsu picks a global cut, which is right for a scanned page or a phone photo with
    even lighting. It fails on photos with a strong shadow, where a fixed cut wipes out
    one corner — so the result is sanity-checked: if Otsu marks more than half the page
    as ink the polarity must be reversed (white paper, dark ink), and if it marks less
    than 0.2% as ink the page is probably washed out and an adaptive threshold is used
    instead.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coverage = mask.mean() / 255.0
    if coverage > 0.5:                       # page was inverted (white text on dark background)
        mask = cv2.bitwise_not(mask)
    elif coverage < 0.002:                   # faint pencil / bad exposure: fall back to local threshold
        mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)

    # Deliberately no morphological opening here. A 2x2 open looks like harmless
    # denoising but it shaves a pixel off every stroke, and at 28px a thin "I" stem is
    # only two pixels wide — cleaning up noise cost ~6 points of character accuracy.
    # Specks are dropped later, by the minimum-line-height filter in segment_lines.
    return mask


def deskew(mask: np.ndarray, parallel: np.ndarray | None = None
           ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Rotate a binary ink mask so the text baselines are horizontal.

    ``parallel`` lets a second image (usually the grayscale ink image) be rotated by the
    exact same matrix. That matters: segmentation boxes are computed on the mask, and if
    the image they are used to crop from is still at the original angle, every crop ends
    up offset from the ink it is supposed to contain.

    A phone photo is almost never level, and a 5-degree tilt is enough to merge
    neighbouring lines in the projection profile below. The angle comes from
    :func:`cv2.minAreaRect` fitted to the ink pixels, which is robust and needs no
    line detection. Angles beyond 45 degrees are rejected: that is a sideways photo,
    not a tilt, and "correcting" it would make things worse.
    """
    def _bundle(rotated_mask: np.ndarray, rotated_parallel: np.ndarray | None) -> Any:
        """Return just the mask, or the (mask, parallel) pair when one was supplied."""
        if parallel is None:
            return rotated_mask
        return rotated_mask, rotated_parallel

    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 50:
        return _bundle(mask, parallel)
    (_, _), (_, _), angle = cv2.minAreaRect(coords)
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.7:
        return _bundle(mask, parallel)
    h, w = mask.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated_mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    if parallel is None:
        return rotated_mask
    # Ink images are bright-on-dark, so the new border must be black (no ink).
    rotated_parallel = cv2.warpAffine(parallel, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return rotated_mask, rotated_parallel


def _runs(profile: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Return ``[start, end)`` index pairs where ``profile`` exceeds ``threshold``."""
    above = profile > threshold
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(above):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(above)))
    return spans


def _merge_spans(spans: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Merge spans that are separated by fewer than ``gap`` pixels.

    Dots on an ``i``, the crossbar of a ``t`` and stray specks all produce their own
    horizontal band. Without merging, one written line would be split into three.
    """
    if not spans:
        return []
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start - merged[-1][1] < gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(int(s), int(e)) for s, e in merged]


def segment_lines(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find the vertical ``(top, bottom)`` extent of every text line in a page mask.

    Works from the horizontal projection profile — the count of ink pixels per row.
    Rows that contain writing have a tall bar, the gap between lines is close to zero.
    """
    if mask.size == 0:
        return []
    profile = (mask > 0).sum(axis=1).astype(np.float32)
    if profile.max() == 0:
        return []
    spans = _runs(profile, max(1.0, profile.max() * 0.04))
    if not spans:
        return []
    heights = [e - s for s, e in spans]
    spans = _merge_spans(spans, gap=max(2, int(np.median(heights) * 0.55)))
    # Anything shorter than a third of the median line is a speck, not a line.
    heights = [e - s for s, e in spans]
    floor = max(3, int(np.median(heights) * 0.33)) if heights else 3
    return [(s, e) for s, e in spans if e - s >= floor]


def segment_words(line_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Split one line mask into word boxes ``(x, y, w, h)``.

    Vertical projection profile again, this time on columns. The space between words is
    wider than the gap between letters, so the threshold is derived from the median
    letter width instead of being a magic constant — that keeps it working for both
    large and small handwriting.
    """
    cols = (line_mask > 0).sum(axis=0)
    if cols.max() == 0:
        return []
    spans = _runs(cols, 0)
    if not spans:
        return []

    widths = [e - s for s, e in spans]
    median_width = float(np.median(widths)) or 1.0

    # Splitting words on a fixed fraction of the median letter width is fragile: narrow
    # letters ("I", "l", "1") leave wide gaps inside a word, and a single threshold then
    # tears "BIGAS" into "B" + "IGAS". Instead the gap distribution is split into two
    # clusters with Otsu, which adapts to the writer's spacing on that particular line.
    gaps = np.array([b[0] - a[1] for a, b in zip(spans, spans[1:])], dtype=np.uint8).reshape(-1, 1)
    if gaps.size >= 2:
        median_gap = float(np.median(gaps))
        # Evenly spaced letters mean one word: without this guard a line of uniformly
        # spaced capitals ("MANILA") gets split on nothing but noise.
        if median_gap <= 0 or float(gaps.max()) < median_gap * 1.6:
            gap_threshold = int(gaps.max()) + 1
        elif float(gaps.max()) > median_width * 0.35:
            threshold, _ = cv2.threshold(gaps, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            gap_threshold = int(max(threshold, median_gap * 1.6, median_width * 0.35))
        else:
            gap_threshold = max(3, int(median_width * 0.75))
    else:
        gap_threshold = max(3, int(median_width * 0.75))

    words = _merge_spans(spans, gap=gap_threshold)

    boxes: list[tuple[int, int, int, int]] = []
    for x0, x1 in words:
        sub = line_mask[:, x0:x1]
        rows = (sub > 0).sum(axis=1)
        row_span = _runs(rows, 0)
        if not row_span:
            continue
        y0, y1 = row_span[0][0], row_span[-1][1]
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


def segment_glyphs(word_mask: np.ndarray, max_ratio: float = 1.75) -> list[tuple[int, int, int, int]]:
    """Split one word mask into individual character boxes ``(x, y, w, h)``.

    Letters that touch — ``tt``, ``oo``, a sloppy ``rn`` — arrive as one wide blob.
    Any blob wider than ``max_ratio`` times the median letter width is therefore cut
    into equal slices. That is a heuristic, not a solution: it is exactly the point
    where block-letter recognition stops coping with cursive, which is why the module
    documents cursive as out of scope for this backend.
    """
    cols = (word_mask > 0).sum(axis=0)
    if cols.max() == 0:
        return []
    spans = _runs(cols, 0)
    if not spans:
        return []

    widths = [e - s for s, e in spans]
    median_w = float(np.median(widths)) or 1.0

    glyphs: list[tuple[int, int, int, int]] = []
    for x0, x1 in spans:
        sub = word_mask[:, x0:x1]
        rows = (sub > 0).sum(axis=1)
        row_span = _runs(rows, 0)
        if not row_span:
            continue
        y0, y1 = row_span[0][0], row_span[-1][1]
        width = x1 - x0
        pieces = max(1, int(round(width / median_w))) if width > median_w * max_ratio else 1
        step = width / pieces
        for piece in range(pieces):
            px0 = int(round(x0 + piece * step))
            px1 = int(round(x0 + (piece + 1) * step))
            glyphs.append((px0, y0, max(1, px1 - px0), max(1, y1 - y0)))
    return glyphs


def normalize_glyph(glyph: np.ndarray, size: int = GLYPH_SIZE) -> np.ndarray:
    """Fit one character crop into a ``size x size`` box the way EMNIST is laid out.

    Two details here were measured, not guessed, because getting them wrong costs more
    accuracy than any model choice:

    * **Scale.** EMNIST glyphs fill about 86% of their 28-pixel frame. Scaling to any
      other fraction (0.8, 0.92, ...) drops held-out accuracy by 1-5 points.
    * **No centroid shift.** Re-centring by centre of mass looks tidy but measurably
      hurts (-3.5 points): EMNIST already centred its glyphs, so shifting again moves
      asymmetric characters such as ``j`` or ``Q`` away from where the model learned them.

    The crop is also tightened to the ink bounding box first, so a glyph photographed
    with generous margins lands in the same place as one that fills the crop.
    """
    if glyph.ndim == 3:
        glyph = to_gray(glyph)

    ink_rows, ink_cols = np.nonzero(glyph > 8)
    if len(ink_rows) == 0:
        return np.zeros((size, size), np.uint8)

    tight = glyph[ink_rows.min():ink_rows.max() + 1, ink_cols.min():ink_cols.max() + 1]
    height, width = tight.shape[:2]
    scale = (size * 0.86) / max(height, width)
    new_w, new_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    resized = cv2.resize(tight, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((size, size), np.uint8)
    offset_x = (size - new_w) // 2
    offset_y = (size - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
    return canvas


# --------------------------------------------------------------------------------------
# Recognition backends
# --------------------------------------------------------------------------------------

class GlyphBackend:
    """Bern's own classifier: one scikit-learn model over EMNIST handwritten characters.

    Recognises 62 classes — ``0-9``, ``A-Z`` and ``a-z``. Punctuation is not in EMNIST,
    so it is reported as unsupported rather than silently guessed (see
    :data:`EMNIST_UNSUPPORTED`).
    """

    name = "bern-glyph"

    def __init__(self, model_path: Path = DEFAULT_HANDWRITING_MODEL_PATH, case_mode: str = "auto") -> None:
        """
        Args:
            model_path: Trained ``.joblib`` bundle.
            case_mode: ``auto`` keeps whatever case the model prefers, ``upper`` or
                ``lower`` forces the output case. Under ``auto`` the case twins of each
                letter are still folded (see :meth:`recognize`).
        """
        self.model_path = Path(model_path)
        self.case_mode = case_mode
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No handwriting model at {self.model_path}. Train one first:\n"
                "    python3 -m bern.handwriting"
            )
        bundle = joblib.load(self.model_path)
        self.classifier = bundle["classifier"]
        self.classes: list[str] = list(bundle["classes"])
        self.metrics: dict[str, Any] = dict(bundle.get("metrics", {}))

        # Class indices that describe the same letter in a different case, e.g. O/o and S/s.
        twins: dict[str, list[int]] = {}
        for index, char in enumerate(self.classes):
            if char.isalpha():
                twins.setdefault(char.lower(), []).append(index)
        self.case_twins: list[list[int]] = [group for group in twins.values() if len(group) > 1]

    def _class_probabilities(self, glyph_bgr: np.ndarray) -> np.ndarray:
        """Per-class probabilities for one glyph, after case-twin folding.

        The probabilities of the two case forms of a letter are pooled onto whichever
        form the model prefers. EMNIST scores ``O`` and ``o`` (and ``0``) as separate
        classes, so without this step the model keeps splitting its confidence three
        ways and picks the wrong one; pooling bought about 5 points of end-to-end word
        accuracy in testing. Digits are never pooled, so ``0`` still has to win on its
        own merit.
        """
        normalized = normalize_glyph(glyph_bgr)
        features = glyph_features(normalized).reshape(1, -1)
        probas = self.classifier.predict_proba(features)[0].copy()

        for group in self.case_twins:
            mass = float(probas[group].sum())
            winner = int(group[int(np.argmax(probas[group]))])
            probas[group] = 0.0
            probas[winner] = mass
        return probas

    def _apply_case_mode(self, char: str) -> str:
        """Force upper/lower case when the caller asked for it."""
        if self.case_mode == "upper":
            return char.upper()
        if self.case_mode == "lower":
            return char.lower()
        return char

    def recognize(self, glyph_bgr: np.ndarray) -> tuple[str, float]:
        """Classify a single normalised character.

        Returns:
            ``(character, probability)``. Probability is the softmax output of the
            log-loss classifier, so it is a real confidence and not a raw margin.
        """
        probas = self._class_probabilities(glyph_bgr)
        best = int(np.argmax(probas))
        return self._apply_case_mode(self.classes[best]), float(probas[best])

    def recognize_topk(self, glyph_bgr: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Return the ``k`` most likely characters for one glyph, best first.

        This is the input to :func:`bern.ph_lexicon.decode_word`, which needs the
        runner-up letters — a ``g`` read as ``9`` with 0.41 confidence usually has ``g``
        sitting right behind it at 0.38, and that second option is what lets the dictionary
        rescue the word.
        """
        probas = self._class_probabilities(glyph_bgr)
        best_indices = np.argsort(probas)[::-1][:k]
        return [(self._apply_case_mode(self.classes[int(i)]), float(probas[int(i)])) for i in best_indices]


class TrocrBackend:
    """Microsoft TrOCR — a transformer trained specifically on handwritten text lines.

    This is the backend to use for cursive: it reads a whole line as one sequence and
    therefore never needs to segment individual letters. It is optional because it
    pulls in PyTorch and a ~1.3 GB model, which is a much heavier dependency than the
    rest of Bern.

    Install with ``pip install transformers torch pillow``.
    """

    name = "trocr"
    model_id = "microsoft/trocr-base-handwritten"

    def __init__(self, model_id: str | None = None) -> None:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:  # pragma: no cover - depends on optional packages
            raise ImportError(
                "The TrOCR backend needs the optional dependencies:\n"
                "    pip install transformers torch pillow\n"
                "Bern's built-in glyph backend works without them (python3 -m bern.handwriting)."
            ) from exc

        from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: F811

        self.model_id = model_id or self.model_id
        self.processor = TrOCRProcessor.from_pretrained(self.model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id)
        self.model.eval()

    def read_line(self, line_bgr: np.ndarray) -> tuple[str, float]:
        """Transcribe one line image. Returns ``(text, confidence)``.

        The confidence is the geometric mean of the per-token probabilities, which is
        comparable across lines of different lengths; TrOCR itself does not expose a
        calibrated score.
        """
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(line_bgr, cv2.COLOR_GRAY2RGB) if line_bgr.ndim == 2 else cv2.cvtColor(line_bgr, cv2.COLOR_BGR2RGB)
        pixel_values = self.processor(images=Image.fromarray(rgb), return_tensors="pt").pixel_values
        with torch.no_grad():
            generated = self.model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
        text = self.processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]
        scores = generated.sequences_scores if hasattr(generated, "sequences_scores") else None
        confidence = float(np.exp(scores[0] / max(1, len(text)))) if scores is not None else float("nan")
        return text.strip(), min(1.0, confidence)


class TesseractBackend:
    """Fallback wrapper around the Tesseract binary.

    Good on printed text, mediocre on neat block handwriting, poor on cursive. Kept
    because it is a two-line install on most Linux distributions and makes a useful
    cross-check when Bern's own model disagrees with itself.
    """

    name = "tesseract"

    def __init__(self) -> None:
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover - depends on optional packages
            raise ImportError("Tesseract backend needs `pip install pytesseract` and the tesseract binary.") from exc
        self._pt = pytesseract

    def read_line(self, line_bgr: np.ndarray) -> tuple[str, float]:
        text = self._pt.image_to_string(line_bgr).strip()
        return text, float("nan")


# --------------------------------------------------------------------------------------
# High level reader
# --------------------------------------------------------------------------------------

class HandwritingReader:
    """Turns a photo of handwriting into text.

    The pipeline is: decode -> grayscale -> binarise -> deskew -> segment into lines ->
    segment each line into words -> hand each word to a backend -> reassemble.

    Example:
        >>> reader = HandwritingReader()
        >>> result = reader.read_image(Path("note.jpg").read_bytes())
        >>> result["text"]
        'HELLO WORLD'
    """

    def __init__(self, backend: str = "auto", model_path: Path = DEFAULT_HANDWRITING_MODEL_PATH,
                 case_mode: str = "auto", use_lexicon: bool = True,
                 use_symbols: bool = True, photo_mode: bool = True) -> None:
        """
        Args:
            backend: ``auto``, ``glyph``, ``trocr`` or ``tesseract``.
            model_path: Trained glyph model.
            case_mode: ``auto``, ``upper`` or ``lower``.
            use_lexicon: Correct each word against the Filipino/English dictionary. Costs
                nothing measurable and repairs the digit-for-letter confusions that the
                character model cannot resolve on its own; disable only to inspect raw
                model output.
            use_symbols: Recover punctuation, math operators and accented letters, which the
                62-class glyph model cannot emit at all.
            photo_mode: Apply the photo repairs (perspective correction, ruled-line removal,
                column-aware reading order). Harmless on a clean scan and essential on a
                phone photo; switch it off only to compare against the raw pipeline.
        """
        self.model_path = Path(model_path)
        self.requested_backend = backend
        self.case_mode = case_mode
        self.use_lexicon = use_lexicon
        self.use_symbols = use_symbols
        self.photo_mode = photo_mode
        self._symbol_backend: Any = None
        self._backend: Any = None
        self._backend_name = ""

    def _resolve_backend(self) -> Any:
        """Pick a backend, honouring an explicit request and falling back gracefully.

        ``auto`` prefers TrOCR when it is installed because it reads cursive far better,
        and otherwise uses Bern's glyph model. The choice is reported back to the caller
        in ``result["backend"]`` so nobody has to guess which engine produced the text.
        """
        if self._backend is not None:
            return self._backend

        choice = self.requested_backend
        if choice == "auto":
            try:
                self._backend = TrocrBackend()
            except Exception:
                self._backend = GlyphBackend(self.model_path, self.case_mode)
        elif choice == "trocr":
            self._backend = TrocrBackend()
        elif choice == "tesseract":
            self._backend = TesseractBackend()
        elif choice == "glyph":
            self._backend = GlyphBackend(self.model_path, self.case_mode)
        else:
            raise ValueError(f"Unknown backend {choice!r}. Choose auto, glyph, trocr or tesseract.")

        self._backend_name = self._backend.name
        return self._backend

    def read_image(self, source: bytes | str | Path | np.ndarray) -> dict[str, Any]:
        """Recognise handwriting from raw bytes, a file path or an already-decoded image.

        Args:
            source: JPEG/PNG/WebP bytes, a filesystem path, or a BGR ``numpy`` array.

        Returns:
            A dictionary with ``text`` (the full transcription, lines joined by
            newlines), ``lines`` (per-line detail), ``backend``, ``confidence``,
            ``glyph_count`` and ``inference_ms``.

        Raises:
            ValueError: if the image cannot be decoded.
        """
        started = time.perf_counter()
        image = self._decode(source)
        backend = self._resolve_backend()
        if self.use_symbols:
            # Load the symbol model now rather than on first use. It is consulted per word,
            # and a lazily-loaded backend that is never touched stays silently disabled.
            self._get_symbol_backend()

        gray = to_gray(image)
        # Cap the working size: beyond ~1600 px there is no accuracy to gain, only time.
        scale = min(1.0, 1600.0 / max(gray.shape[:2]))
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        repairs: dict[str, Any] = {"perspective": False, "rules_removed": 0, "columns": 1}
        if self.photo_mode:
            # A phone photo is a trapezoid, not a rectangle. Straighten it before anything
            # measures the page, otherwise glyphs in one corner are systematically larger
            # than those in the other and one normalisation cannot serve both.
            gray, repaired = correct_perspective(gray)
            repairs["perspective"] = repaired

        # Invert first (ink bright, paper black) and deskew both images with one matrix
        # so the crop boxes below line up with the pixels they refer to.
        ink = cv2.bitwise_not(gray)
        mask = ink_mask(gray)

        if self.photo_mode:
            # Notebook rules survive thresholding as long horizontal runs and weld whole
            # lines of text into single smeared components.
            mask, removed = remove_ruled_lines(mask)
            repairs["rules_removed"] = removed

        mask, ink = deskew(mask, ink)

        # Dictionary decoding only makes sense for backends that expose ranked characters.
        top_k_backend = hasattr(backend, "recognize_topk") and self.use_lexicon
        # 8 candidates, not 5: on a hard glyph the correct letter is sometimes ranked
        # 6th-8th, and if it is absent from the list no dictionary prior can recover it.
        # Swept 5/8/12 — no measurable cost, so the wider list is kept.
        top_k = 8

        lines_out: list[dict[str, Any]] = []
        # Read each column top-to-bottom before moving right. Reading across a gutter
        # stitches the left column's line to the right column's and produces nonsense.
        if self.photo_mode:
            columns = detect_columns(mask)
        else:
            columns = [(0, mask.shape[1])]
        repairs["columns"] = len(columns)

        line_boxes: list[tuple[int, int, int, int]] = []   # (top, bottom, x0, x1)
        for x0, x1 in columns:
            for top, bottom in segment_lines(mask[:, x0:x1]):
                line_boxes.append((top, bottom, x0, x1))

        if not line_boxes:
            return self._result("", [], backend, 0, started, repairs)

        for top, bottom, column_x0, column_x1 in line_boxes:
            line_mask = mask[top:bottom, column_x0:column_x1]
            line_ink = ink[top:bottom, column_x0:column_x1]

            if isinstance(backend, TrocrBackend):
                # Line-level engine: skip segmentation entirely.
                text, confidence = backend.read_line(cv2.bitwise_and(line_ink, line_ink, mask=line_mask))
                if text:
                    lines_out.append({"text": text, "confidence": round(confidence, 3) if confidence == confidence else None,
                                      "glyphs": len(text.replace(" ", ""))})
                continue

            word_boxes = segment_words(line_mask)

            # Baseline: most letters sit on it, descenders hang below. The median of every
            # glyph's bottom edge is a robust estimate even when a line contains several
            # descenders, which a mean would drag downward.
            glyph_boxes: list[list[tuple[int, int, int, int]]] = []
            bottoms: list[int] = []
            heights: list[int] = []
            for x, y, w, h in word_boxes:
                boxes = [(x + gx, y + gy, gw, gh) for gx, gy, gw, gh in segment_glyphs(line_mask[y:y + h, x:x + w])]
                glyph_boxes.append(boxes)
                bottoms.extend(box_top + box_h for _bx, box_top, _bw, box_h in boxes)
                heights.extend(box_h for _bx, _by, _bw, box_h in boxes)

            # Median *letter* height: the yardstick for "is this a mark or a letter".
            reference_height = int(np.median(heights)) if heights else int(line_mask.shape[0])

            # Baseline: most letters sit on it, descenders hang below. Only letter-sized
            # glyphs vote. A comma is a descender by nature, and letting its own tail into the
            # median pulls the baseline down until the comma no longer looks like it hangs
            # below anything — which is exactly how a "," became a ".".
            letter_bottoms = [
                box_top + box_h
                for boxes in glyph_boxes
                for _bx, box_top, _bw, box_h in boxes
                if box_h >= 0.6 * reference_height
            ]
            baseline = float(np.median(letter_bottoms)) if letter_bottoms else float(line_mask.shape[0])
            height_of_line = int(line_mask.shape[0])

            words: list[str] = []
            confidences: list[float] = []
            glyph_total = 0

            for word_index, (x, y, w, h) in enumerate(word_boxes):
                word_ink = line_ink[y:y + h, x:x + w]
                letter_options: list[list[tuple[str, float]]] = []
                fallback_chars: list[str] = []
                word_text = ""
                # A comma is a dot with a tail, and connected-component labelling splits the
                # two, so one comma arrives as two mark boxes and would be emitted twice.
                # The most recent mark is held here so a vertically stacked partner can be
                # folded back into a single mark before it reaches the text.
                pending_mark: tuple[int, int, int, int] | None = None

                for bx, by, bw, bh in glyph_boxes[word_index]:
                    # Coordinates inside this word's crop.
                    gx, gy = bx - x, by - y
                    glyph = word_ink[gy:gy + bh, gx:gx + bw]
                    if not glyph.size:
                        continue
                    glyph_total += 1

                    # Punctuation first: geometry settles it more reliably than any
                    # classifier, and it must never enter the beam search — a stray period
                    # inside a word would break the dictionary match.
                    mark = classify_punctuation(bw, bh, by, height_of_line, baseline, reference_height)
                    if mark is not None:
                        if pending_mark is not None:
                            union = self._union_box(pending_mark, (bx, by, bw, bh))
                            if union is not None and word_text:
                                merged = classify_punctuation(
                                    union[2], union[3], union[1], height_of_line, baseline, reference_height
                                )
                                word_text = word_text[:-1] + (merged or mark)
                                pending_mark = union
                                continue
                        if letter_options:
                            decoded, decoded_conf = decode_word(letter_options, self._lexicon)
                            word_text += self._apply_word_case(decoded)
                            confidences.append(decoded_conf)
                            letter_options = []
                        elif fallback_chars:
                            word_text += "".join(fallback_chars)
                            fallback_chars = []
                        word_text += mark
                        pending_mark = (bx, by, bw, bh)
                        continue

                    pending_mark = None

                    if top_k_backend:
                        options = backend.recognize_topk(glyph, k=top_k)
                        letter_options.append(options)
                    else:
                        char, prob = backend.recognize(glyph)
                        fallback_chars.append(char)
                        confidences.append(prob)

                if len(letter_options) == 1:
                    # A single glyph that is not a word is usually a standalone symbol:
                    # "+", "=", "%", "₱", "@", or a full-height "?" / "!". Multi-glyph words
                    # never reach the symbol model — measured attempts to do so turned
                    # ordinary letters into "#" and "@" and cost ~20 points of accuracy.
                    only_box = glyph_boxes[word_index][0] if glyph_boxes[word_index] else None
                    if only_box is not None:
                        _bx, by, bw, bh = only_box
                        glyph = line_ink[by:by + bh, _bx:_bx + bw]
                        if glyph.size:
                            letter_options[0] = self._merge_symbol_options(glyph, letter_options[0])
                elif self._symbol_backend is not None and self.use_symbols:
                    # "?" and "!" reach full height, so the geometry test cannot see them and
                    # the mark stays inside the word. Only the trailing glyph is considered,
                    # and only when the symbol model is nearly certain about it. Running the
                    # symbol model across a whole word was measured to turn ordinary letters
                    # into "#" and "@" and cost about twenty points of accuracy.
                    trailing_box = glyph_boxes[word_index][-1] if glyph_boxes[word_index] else None
                    if trailing_box is not None and len(letter_options) == len(glyph_boxes[word_index]):
                        _bx, by, bw, bh = trailing_box
                        glyph = line_ink[by:by + bh, _bx:_bx + bw]
                        if glyph.size and letter_options[-1]:
                            symbol, symbol_prob = self._symbol_backend.recognize_topk(glyph, k=1)[0]
                            # The letter model is often confidently wrong here: a written
                            # "?" comes back as "7" at 0.85. So the gate is the symbol
                            # model's own certainty, not the letter model's doubt, and the
                            # accepted set stays narrow.
                            if symbol_prob >= 0.95 and symbol in ".?!":
                                decoded, decoded_conf = decode_word(letter_options[:-1], self._lexicon)
                                word_text += self._apply_word_case(decoded)
                                confidences.append(decoded_conf)
                                word_text += symbol
                                letter_options = []

                if letter_options:
                    # Dictionary-constrained beam search over the whole word. This is what
                    # turns "p1s0" into "piso" and "6A7AS" into "gatas".
                    tail, tail_conf = decode_word(letter_options, self._lexicon)
                    word_text += self._apply_word_case(tail)
                    confidences.append(tail_conf)
                elif fallback_chars:
                    word_text += "".join(fallback_chars)

                if not word_text:
                    continue
                # A mark isolated as its own "word" belongs to the word before it:
                # "...PISO" + "." must join, not become "PISO .".
                word_text = self._tidy_marks(word_text)
                if word_text in _PUNCTUATION_ONLY and words:
                    words[-1] += word_text
                else:
                    words.append(word_text)

            if words:
                line_conf = float(np.mean(confidences)) if confidences else float("nan")
                lines_out.append({"text": " ".join(words),
                                  "confidence": round(line_conf, 3) if line_conf == line_conf else None,
                                  "glyphs": glyph_total})

        text = "\n".join(item["text"] for item in lines_out)
        return self._result(text, lines_out, backend, sum(i["glyphs"] for i in lines_out), started, repairs)

    @property
    def _lexicon(self) -> Any:
        """The shared word list, or ``None`` when correction is switched off."""
        return get_lexicon() if self.use_lexicon else None

    def _apply_word_case(self, word: str) -> str:
        """Force word case when requested, otherwise keep the decoded spelling."""
        if self.case_mode == "upper":
            return word.upper()
        if self.case_mode == "lower":
            return word.lower()
        return word

    @staticmethod
    def _tidy_marks(text: str) -> str:
        """Collapse marks that one glyph was split into.

        Connected-component labelling sometimes breaks a single mark into pieces — a comma
        into head and tail — and each piece is then classified on its own, so the same mark
        arrives twice. A run of exactly two identical marks is folded into one; a run of
        three or more is kept, because that is an ellipsis and is meant to be there.
        A semicolon immediately followed by a comma is likewise one split comma.
        """
        if len(text) < 2:
            return text

        out: list[str] = []
        index = 0
        while index < len(text):
            char = text[index]
            run = 1
            while index + run < len(text) and text[index + run] == char:
                run += 1
            if char in _PUNCTUATION_ONLY and run == 2:
                out.append(char)
            else:
                out.extend([char] * run)
            index += run

        tidied = "".join(out)
        return tidied.replace(";,", ",").replace(";.", ".")

    @staticmethod
    def _union_box(
        first: tuple[int, int, int, int], second: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int] | None:
        """Combine two glyph boxes if they overlap horizontally enough to be one mark.

        Returns ``None`` when the boxes sit well apart, which means they are two separate
        marks rather than the pieces of one.
        """
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        # A mark split down the middle yields two halves that merely touch, so a one-pixel
        # gap still counts as one mark. Two genuinely separate marks are further apart.
        overlap = min(ax + aw, bx + bw) - max(ax, bx)
        if overlap < -1:
            return None
        left, top = min(ax, bx), min(ay, by)
        right, bottom = max(ax + aw, bx + bw), max(ay + ah, by + bh)
        return (left, top, right - left, bottom - top)

    def _merge_symbol_options(self, glyph: np.ndarray,
                              letter_options: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Let the symbol model override a low-confidence letter guess.

        The rule is conservative on purpose. The symbol model is trained on rendered fonts,
        not real pen strokes, so it is only trusted when it is *sure* and the letter model
        is *not* — otherwise a slightly odd ``A`` would come back as ``&``.

        The merged list keeps the letter candidates first, so when the two models agree on a
        plausible word the dictionary still wins.
        """
        if not self.use_symbols:
            return letter_options
        try:
            symbol_backend = self._get_symbol_backend()
        except (FileNotFoundError, ImportError):
            self.use_symbols = False          # trained model absent: run without it, quietly
            return letter_options

        symbol_options = symbol_backend.recognize_topk(glyph, k=2)
        if not symbol_options:
            return letter_options

        best_symbol, symbol_probability = symbol_options[0]
        best_letter, letter_probability = letter_options[0] if letter_options else ("", 0.0)
        # Only symbols that cannot plausibly be a letter are allowed to win, and only by a
        # wide margin. The symbol model is trained on rendered fonts, so it is trusted for
        # shape-distinctive marks and nothing else.
        _ = best_letter
        if symbol_probability >= 0.85 and letter_probability <= 0.45:
            return [(best_symbol, symbol_probability)] + letter_options
        return letter_options

    def _get_symbol_backend(self) -> Any:
        """Lazily build the symbol classifier, so the model file is optional at runtime."""
        if self._symbol_backend is None:
            from bern.symbols import SymbolBackend

            self._symbol_backend = SymbolBackend()
        return self._symbol_backend

    def _decode(self, source: bytes | str | Path | np.ndarray) -> np.ndarray:
        """Normalise the four accepted input types into one BGR array."""
        if isinstance(source, np.ndarray):
            return source
        if isinstance(source, (str, Path)):
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read an image from {source}")
            return image
        if isinstance(source, (bytes, bytearray)):
            buffer = np.frombuffer(bytes(source), dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode image. Please upload a valid PNG, JPEG, or WebP photo.")
            return image
        raise TypeError(f"Unsupported input type {type(source)!r}")

    def _result(self, text: str, lines: list[dict[str, Any]], backend: Any,
                glyph_count: int, started: float,
                repairs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Assemble the public result dictionary shared by every backend."""
        confidences = [item["confidence"] for item in lines if item.get("confidence") is not None]
        return {
            "text": text,
            "lines": lines,
            "backend": getattr(backend, "name", self._backend_name),
            "confidence": round(float(np.mean(confidences)), 3) if confidences else None,
            "glyph_count": glyph_count,
            "inference_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "unsupported_symbols": sorted(set(EMNIST_UNSUPPORTED) & set(text)),
            "repairs": repairs or {},
        }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main() -> None:
    """``python3 -m bern.handwriting`` — train, evaluate, or transcribe from the shell."""
    parser = argparse.ArgumentParser(description="Bern handwriting recognition")
    parser.add_argument("--train", action="store_true", help="train the glyph model (default action)")
    parser.add_argument("--evaluate", action="store_true", help="score the saved model on EMNIST test")
    parser.add_argument("--image", type=Path, help="transcribe a single image file")
    parser.add_argument("--epochs", type=int, default=5, help="training passes (default 5)")
    parser.add_argument("--max-train", type=int, default=None, help="cap training rows (smoke test)")
    parser.add_argument("--max-test", type=int, default=20_000, help="cap evaluation rows")
    parser.add_argument("--data-dir", type=Path, default=EMNIST_DIR,
                        help="where EMNIST lives / should be downloaded (default: data/external/emnist)")
    args = parser.parse_args()

    if args.image:
        result = HandwritingReader().read_image(args.image)
        print(f"[{result['backend']}] confidence={result['confidence']} {result['inference_ms']} ms")
        print(result["text"])
        return

    if args.evaluate:
        if not DEFAULT_HANDWRITING_MODEL_PATH.exists():
            raise SystemExit("No trained model yet — run: python3 -m bern.handwriting --train")
        split_dir = ensure_emnist(args.data_dir)
        model = joblib.load(DEFAULT_HANDWRITING_MODEL_PATH)
        report = evaluate_glyph_model(
            model,
            split_dir / "emnist-byclass-test-images-idx3-ubyte.gz",
            split_dir / "emnist-byclass-test-labels-idx1-ubyte.gz",
            max_samples=args.max_test,
        )
        print(f"accuracy {report['accuracy'] * 100:.2f}% on {report['samples']:,} characters")
        for row in report["top_confusions"]:
            print(f"  {row['pair']}: {row['count']}")
        return

    train_glyph_model(dataset_dir=args.data_dir, epochs=args.epochs,
                      max_train=args.max_train, max_test=args.max_test)


if __name__ == "__main__":
    main()
