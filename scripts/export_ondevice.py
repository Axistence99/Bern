#!/usr/bin/env python3
"""Export Bern's models for on-device inference, with no Python at runtime.

The models are tiny — 62 x 341 weights for handwriting, 24 x 18,898 for intents — so the only
reason Bern needs a server is the runtime around them. This script writes plain data files
that a JavaScript runtime can load and evaluate with arithmetic.

Two decisions carry the whole design:

* **Everything that needs retraining is retrained on the portable feature code.** Training
  and inference must share one feature extractor. Reimplementing OpenCV's HOG bit-for-bit is
  not achievable in reasonable time, so instead both sides use
  :mod:`bern.portable_features`, and whatever it does differently from OpenCV stops mattering.
* **Weights are base64 float32, not JSON numbers.** 453,552 decimal literals would be
  megabytes of text for what 1.8 MB of bytes holds exactly.

Usage:
    PYTHONPATH=. python3 scripts/export_ondevice.py                  # intents and symbols
    PYTHONPATH=. python3 scripts/export_ondevice.py --train-glyphs   # also retrain handwriting
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
from sklearn.linear_model import SGDClassifier  # noqa: E402

from bern import handwriting, symbols  # noqa: E402
from bern.portable_features import BINS, BLOCK_CELLS, CELL_SIZE, GAMMA_CORRECT, GLYPH_SIZE  # noqa: E402
from bern.portable_text import VectorizerSpec  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "models" / "ondevice"
DEFAULT_MODEL = BASE_DIR / "models" / "bern_model.joblib"
DEFAULT_GLYPH = BASE_DIR / "models" / "handwriting_glyph_model.joblib"
DEFAULT_SYMBOL = BASE_DIR / "models" / "symbol_model.joblib"

#: Intents whose stored reply *announces* an answer the server then produces — a price, a
#: running total, the clock, a transcription. The phone recognises the question perfectly well
#: but has none of the data, so printing the announcement on its own would promise a result
#: and then show nothing. Read the first stored reply for each of these and the pattern is
#: visible: "Let me check my 2026 Philippine Price Knowledge Base!", "Let me run the numbers
#: for you!". Every other intent's stored reply is a complete answer.
SERVER_BACKED_INTENTS = (
    "calculator",
    "file_scan",
    "handwriting_ocr",
    "ph_budget_calc",
    "ph_money_counter",
    "ph_nutrition_lookup",
    "ph_price_catalog",
    "ph_price_lookup",
    "study_materials",
    "time_date",
    "train_help",
)

#: Constants the JavaScript has to agree with, recorded in the payload rather than assumed.
DESCRIPTOR = {
    "kind": "hog-zoning-ink",
    "glyph_size": GLYPH_SIZE,
    "cell": CELL_SIZE,
    "block": BLOCK_CELLS,
    "bins": BINS,
    "gamma_correct": GAMMA_CORRECT,
    "clip": 0.2,
    "zoning": 4,
    "ink_threshold": 8,
    "fill": 0.86,
}


def pack_floats(array: np.ndarray) -> str:
    """Encode a float32 array as base64. Exact, and smaller than decimal text."""
    return base64.b64encode(np.asarray(array, dtype=np.float32).tobytes()).decode("ascii")


def pack_strings(items: list[str]) -> str:
    """Join strings with a newline.

    Vocabulary terms come from tokenising text and never contain a newline — word tokens are
    ``\\w`` runs and character n-grams come from ``str.split()`` words — so a newline is an
    unambiguous separator that keeps the payload readable in a diff.
    """
    return "\n".join(items)


def export_intent_model(source: Path, destination: Path) -> dict:
    """Export the intent classifier as vocabulary, IDF vectors and the weight matrix."""
    bundle = joblib.load(source)
    pipeline = bundle["pipeline"]

    features = dict(pipeline.steps)["features"]
    classifier = dict(pipeline.steps)["clf"]

    layers = []
    offset = 0
    for name, vectorizer in features.transformer_list:
        terms = [""] * len(vectorizer.vocabulary_)
        for term, index in vectorizer.vocabulary_.items():
            terms[index] = term
        layers.append(
            {
                "name": name,
                "weight": float((features.transformer_weights or {}).get(name, 1.0)),
                "analyzer": vectorizer.analyzer,
                "ngram_range": list(vectorizer.ngram_range),
                "sublinear_tf": bool(vectorizer.sublinear_tf),
                "size": len(terms),
                "offset": offset,
                "terms": pack_strings(terms),
                "idf": pack_floats(vectorizer.idf_),
            }
        )
        offset += len(terms)

    payload = {
        "format": "bern-intent-2",
        "classes": list(classifier.classes_),
        "intercept": pack_floats(classifier.intercept_),
        "coefficients": pack_floats(classifier.coef_),
        "coef_shape": list(classifier.coef_.shape),
        "feature_count": offset,
        "layers": layers,
        "intent_responses": bundle.get("intent_responses", {}),
        "server_backed": list(SERVER_BACKED_INTENTS),
        "gating": bundle.get("gating", {}),
        "ai_name": bundle.get("ai_name", "Bern"),
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"classes": len(payload["classes"]), "features": offset, "bytes": destination.stat().st_size}


def check_intent_parity(
    source: Path, layers: list[tuple[str, "VectorizerSpec"]], sample: list[str]
) -> float:
    """Largest element-wise difference between scikit-learn and the portable vectoriser."""
    from bern.portable_text import union_row

    features = dict(joblib.load(source)["pipeline"].steps)["features"]
    weights = features.transformer_weights or {}
    reference = features.transform(sample).toarray()
    portable = np.stack([union_row(layers, weights, text) for text in sample])
    return float(np.abs(reference - portable).max())


def _named_layers(source: Path) -> list[tuple[str, "VectorizerSpec"]]:
    """Rebuild the vectorisers from a bundle in the order the model was fitted with."""
    layers = []
    for name, vectorizer in dict(joblib.load(source)["pipeline"].steps)["features"].transformer_list:
        terms = [""] * len(vectorizer.vocabulary_)
        for term, index in vectorizer.vocabulary_.items():
            terms[index] = term
        layers.append(
            (
                name,
                VectorizerSpec(
                    terms=terms,
                    idf=vectorizer.idf_,
                    analyzer=vectorizer.analyzer,
                    ngram_range=tuple(vectorizer.ngram_range),
                    sublinear_tf=vectorizer.sublinear_tf,
                ),
            )
        )
    return layers


FEATURE_DIM = 341
#: Rows handed to one ``partial_fit`` call. The full training matrix is 300,000 x 341 float32,
#: which is 400 MB; slicing a permutation of it would briefly double that and the machine this
#: is exported on has 1.3 GB free.
TRAIN_CHUNK = 25_000


def extract_glyph_features(images_path: Path, labels_path: Path, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract portable-descriptor features for the first ``limit`` glyphs of a split.

    Preallocated rather than accumulated in a list: 300,000 separate 341-float arrays cost
    roughly twice their contents in object headers alone.
    """
    from bern.portable_features import glyph_features

    features = np.empty((limit, FEATURE_DIM), dtype=np.float32)
    labels = np.empty(limit, dtype=np.int64)
    filled = 0
    for images, batch_labels in zip(
        handwriting.iter_emnist_images(str(images_path), 4096),
        handwriting.iter_emnist_labels(str(labels_path), 4096),
    ):
        take = min(len(images), limit - filled)
        for offset in range(take):
            features[filled + offset] = glyph_features(images[offset])
        labels[filled:filled + take] = batch_labels[:take]
        filled += take
        if filled >= limit:
            break
    return features, labels


def train_portable_glyph_model(data_dir: Path, samples: int, epochs: int, seed: int = 42,
                              printed_per_class: int = 220) -> dict:
    """Retrain the handwriting classifier on the portable descriptor.

    Mirrors ``bern.handwriting.train_glyph_model``: averaged SGD with the log-loss objective,
    driven through ``partial_fit`` so the design matrix never exists twice.

    **Epochs are the setting that matters and the intuitive answer is wrong.** With
    ``average=True`` and ``learning_rate="optimal"`` the step size has collapsed by the end of
    the first pass, so further passes wander around a poor iterate and drag the running
    *average* down with them. Measured on 100,000 glyphs: one pass 79.9%, three passes 75.7%,
    five passes 79.3%. Two passes on 300,000 glyphs scored 80.9%, against 80.6% for one.
    """
    split = data_dir / "byclass"
    mapping = split / "emnist-byclass-mapping.txt"
    if not mapping.exists():
        raise FileNotFoundError(
            f"Missing EMNIST class mapping at {mapping}. "
            "Run `python3 -m bern.handwriting` once, or point --data-dir at a byclass split."
        )
    classes = handwriting.load_class_names(str(mapping))

    started = time.perf_counter()
    print(f"  extracting {samples:,} training glyphs ...", flush=True)
    X, y = extract_glyph_features(
        split / "emnist-byclass-train-images-idx3-ubyte.gz",
        split / "emnist-byclass-train-labels-idx1-ubyte.gz",
        samples,
    )
    print(f"  extracting held-out glyphs ...", flush=True)
    X_test, y_test = extract_glyph_features(
        split / "emnist-byclass-test-images-idx3-ubyte.gz",
        split / "emnist-byclass-test-labels-idx1-ubyte.gz",
        20_000,
    )

    # EMNIST is handwriting. The page reader is pointed at photographs of printed text, and a
    # handwriting classifier reads 32.6% of rendered characters - R as B, A as 8, X and Y as 8,
    # repeatedly, across fonts and sizes. Adding rendered print to the training set took that to
    # 89.7% on a font held out of training while handwriting held at 78.1%, so both
    # distributions are trained. `printed_glyphs.py` owns the rendering and the font split.
    from bern import printed_glyphs

    train_fonts, holdout_fonts = printed_glyphs.split_fonts(holdout=1)
    printed_accuracy = None
    if train_fonts:
        printed_images, printed_labels = printed_glyphs.sample_dataset(
            train_fonts, per_class=printed_per_class, seed=seed
        )
        if len(printed_images):
            from bern.portable_features import glyph_features as _glyph_features

            printed_features = np.empty((len(printed_images), FEATURE_DIM), dtype=np.float32)
            for index, image in enumerate(printed_images):
                printed_features[index] = _glyph_features(image)
            printed_global = np.asarray(
                [classes.index(printed_glyphs.CLASSES[i]) for i in printed_labels], dtype=np.int64
            )
            X = np.vstack([X, printed_features])
            y = np.concatenate([y, printed_global])
            print(f"  added {len(printed_features):,} rendered glyphs from {len(train_fonts)} fonts",
                  flush=True)

    classifier = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=1e-5, average=True,
        max_iter=1, tol=None, random_state=seed, learning_rate="optimal",
    )
    rng = np.random.default_rng(seed)
    class_indices = np.arange(len(classes))
    for epoch in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(X), TRAIN_CHUNK):
            block = order[start:start + TRAIN_CHUNK]
            first = epoch == 0 and start == 0
            classifier.partial_fit(X[block], y[block], classes=class_indices if first else None)

    accuracy = float((classifier.predict(X_test) == y_test).mean())
    return {
        "classifier": classifier,
        "classes": classes,
        "metrics": {
            "samples": int(len(X)),
            "held_out": int(len(X_test)),
            "accuracy": round(accuracy, 4),
            "printed_fonts": len(train_fonts),
            "printed_holdout_fonts": [Path(f).stem for f in holdout_fonts],
            "epochs": epochs,
            "feature_extractor": "portable",
            "elapsed_s": round(time.perf_counter() - started, 1),
        },
    }


def train_portable_symbol_model(per_class: int, seed: int = 42) -> dict:
    """Retrain the symbol classifier on the portable descriptor.

    ``bern.symbols`` renders its training corpus from fonts and distorts it, so the corpus can
    be rebuilt on demand. ``build_synthetic_symbols`` looks the two feature names up on the
    ``handwriting`` module at call time, so patching them there swaps the extractor for the
    duration of the render without touching any other caller.

    The hyperparameters and stratified 80/20 split mirror ``symbols.train_symbol_model``, so
    the score means the same thing. That function is deliberately not reused: it writes to the
    on-disk model, and overwriting the server's OpenCV-feature model with a portable one would
    quietly break the server.
    """
    from sklearn.model_selection import train_test_split

    from bern.portable_features import glyph_features, normalise_glyph

    original = (handwriting.glyph_features, handwriting.normalize_glyph)
    handwriting.glyph_features = glyph_features
    handwriting.normalize_glyph = normalise_glyph
    try:
        features, labels = symbols.build_synthetic_symbols(per_class=per_class, seed=seed)
    finally:
        handwriting.glyph_features, handwriting.normalize_glyph = original

    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    classifier = SGDClassifier(
        loss="log_loss", alpha=1e-5, average=True, max_iter=40, tol=1e-4,
        random_state=seed, learning_rate="optimal",
    )
    classifier.fit(train_x, train_y)
    accuracy = float((classifier.predict(test_x) == test_y).mean())
    return {
        "classifier": classifier,
        "classes": list(symbols.SYMBOL_CLASSES),
        "metrics": {
            "held_out_accuracy": round(accuracy, 4),
            "train_samples": int(len(train_x)),
            "feature_extractor": "portable",
        },
    }


def reuse_existing(destination: Path, kind: str, flag: str) -> dict | None:
    """Keep an already-exported portable model, or explain how to make one.

    Exporting is idempotent: the bundles are committed, so a plain run only refreshes the intent
    model and leaves the two models that need the EMNIST download and the font renderer alone.
    """
    if not destination.exists():
        print(f"  no {kind} bundle yet — pass {flag} to build one")
        return None
    shipped = json.loads(destination.read_text(encoding="utf-8"))
    if shipped.get("metrics", {}).get("feature_extractor") != "portable":
        print(f"  ERROR: {destination.name} holds a model trained on OpenCV features", file=sys.stderr)
        return None
    print(f"  kept the existing portable model, {len(shipped['classes'])} classes")
    return {"classes": len(shipped["classes"]), "bytes": destination.stat().st_size}


def export_glyph_model(classifier, classes: list[str], metrics: dict, destination: Path) -> dict:
    """Write the handwriting classifier as classes, intercept and a weight matrix."""
    payload = {
        "format": "bern-glyph-2",
        "classes": list(classes),
        "intercept": pack_floats(classifier.intercept_),
        "coefficients": pack_floats(classifier.coef_),
        "coef_shape": list(classifier.coef_.shape),
        "descriptor": DESCRIPTOR,
        "metrics": metrics,
    }
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return {"classes": len(classes), "bytes": destination.stat().st_size}


def export_symbol_model(classifier, classes: list[str], metrics: dict, destination: Path) -> dict:
    """Write the symbol classifier in the same shape as the glyph model."""
    payload = {
        "format": "bern-symbol-2",
        "classes": list(classes),
        "intercept": pack_floats(classifier.intercept_),
        "coefficients": pack_floats(classifier.coef_),
        "coef_shape": list(classifier.coef_.shape),
        "descriptor": DESCRIPTOR,
        "metrics": metrics,
    }
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return {"classes": len(classes), "bytes": destination.stat().st_size}


SAMPLE_UTTERANCES = [
    "bakit mahal ang bigas",
    "magkano ang bigas ngayon?",
    "Paano mag-tanim ng palay?",
    "LAGAY MO SA BULSA MO",
    "hello po",
    "2+2",
    "what is the price of rice in manila today",
    "salamat!!",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Bern's models as portable data files.")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="destination directory")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="intent model bundle")
    parser.add_argument("--glyph-model", type=Path, default=DEFAULT_GLYPH, help="handwriting bundle")
    parser.add_argument("--symbol-model", type=Path, default=DEFAULT_SYMBOL, help="symbol bundle")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data" / "external" / "emnist",
                        help="EMNIST directory holding the byclass split")
    parser.add_argument("--train-glyphs", action="store_true", help="retrain handwriting on portable features")
    parser.add_argument("--train-symbols", action="store_true", help="retrain symbols on portable features")
    parser.add_argument("--glyph-samples", type=int, default=120_000, help="glyphs to train on")
    parser.add_argument("--symbol-samples", type=int, default=900, help="renders per symbol class")
    parser.add_argument("--epochs", type=int, default=2, help="training passes over the glyph set")
    parser.add_argument("--skip-parity", action="store_true", help="skip the scikit-learn comparison")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0

    print("intent model ...", flush=True)
    intent = export_intent_model(args.model, args.out / "intent.json")
    print(f"  {intent['classes']} classes, {intent['features']:,} features, {intent['bytes'] / 1e6:.2f} MB")
    total += intent["bytes"]

    if not args.skip_parity:
        worst = check_intent_parity(args.model, _named_layers(args.model), SAMPLE_UTTERANCES)
        verdict = "exact" if worst < 1e-9 else f"DIFFERS by {worst:.2e}"
        print(f"  vectoriser vs scikit-learn on {len(SAMPLE_UTTERANCES)} utterances: {verdict}")
        if worst >= 1e-9:
            print("  ERROR: the portable vectoriser does not reproduce the model's features", file=sys.stderr)
            return 1

    print("symbol model ...", flush=True)
    symbol_path = args.out / "symbol.json"
    if args.train_symbols:
        created = train_portable_symbol_model(args.symbol_samples)
        symbol = export_symbol_model(created["classifier"], created["classes"], created["metrics"], symbol_path)
        print(f"  retrained on portable features: {created['metrics']['held_out_accuracy']:.2%} held out")
        print(f"  {symbol['classes']} classes, {symbol['bytes'] / 1e6:.2f} MB")
        total += symbol["bytes"]
    else:
        existing = reuse_existing(symbol_path, "symbol", "--train-symbols")
        if existing:
            total += existing["bytes"]

    print("glyph model ...", flush=True)
    if args.train_glyphs:
        trained = train_portable_glyph_model(args.data_dir, args.glyph_samples, args.epochs)
        glyph = export_glyph_model(trained["classifier"], trained["classes"], trained["metrics"],
                                   args.out / "glyph.json")
        metrics = trained["metrics"]
        print(f"  {glyph['classes']} classes, {glyph['bytes'] / 1e6:.2f} MB")
        print(f"  held out: {metrics['accuracy']:.2%} on {metrics['held_out']:,} glyphs "
              f"(trained on {metrics['samples']:,}, {metrics['epochs']} epochs, {metrics['elapsed_s']}s)")
    else:
        existing = reuse_existing(args.out / "glyph.json", "glyph", "--train-glyphs")
        if existing:
            total += existing["bytes"]

    print(f"\ntotal shipped: {total / 1e6:.2f} MB in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
