"""Keep the on-device bundles honest.

The phone loads `models/ondevice/*.json` and evaluates them with its own arithmetic, so no
Python test can prove the app works. What can be proved here is that the files are internally
consistent — every weight vector is exactly as long as the number of classes, every vocabulary
is as long as its IDF vector, every term count equals the declared size — and that the
portable feature code the TypeScript was ported from still reproduces scikit-learn.

A length mismatch would not raise on the phone. It would read past the end of a typed array,
score zeros, and return a confident answer built from nothing.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from bern.portable_features import ink_mask

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLES = REPO_ROOT / "models" / "ondevice"


def _load(name: str) -> dict:
    path = BUNDLES / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} has not been exported; run scripts/export_ondevice.py")
    return json.loads(path.read_text(encoding="utf-8"))


def _floats(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.float32)


@pytest.mark.parametrize("name", ["intent", "glyph", "symbol"])
def test_bundle_is_present_and_declares_its_format(name: str) -> None:
    bundle = _load(name)
    assert bundle["format"] == f"bern-{name}-2"
    assert bundle["classes"], "a bundle with no classes cannot answer anything"
    assert len(bundle["coef_shape"]) == 2


@pytest.mark.parametrize("name", ["glyph", "symbol"])
def test_weight_matrix_matches_classes_and_feature_count(name: str) -> None:
    bundle = _load(name)
    rows, columns = bundle["coef_shape"]
    assert rows == len(bundle["classes"])
    assert columns == 341, "the descriptor produces 341 values; anything else is a stale export"
    assert _floats(bundle["coefficients"]).size == rows * columns
    assert _floats(bundle["intercept"]).size == rows


def test_intent_weights_match_layers_and_classes() -> None:
    bundle = _load("intent")
    total = sum(layer["size"] for layer in bundle["layers"])
    assert total == bundle["feature_count"]

    rows, columns = bundle["coef_shape"]
    assert rows == len(bundle["classes"])
    assert columns == total
    assert _floats(bundle["coefficients"]).size == rows * columns
    assert _floats(bundle["intercept"]).size == rows


def test_every_layer_vocabulary_and_idf_line_up() -> None:
    """A vocabulary shorter than its IDF vector shifts every weight after the gap."""
    bundle = _load("intent")
    offset = 0
    for layer in bundle["layers"]:
        terms = layer["terms"].split("\n")
        assert len(terms) == layer["size"], f"{layer['name']} declares {layer['size']} terms, holds {len(terms)}"
        assert _floats(layer["idf"]).size == layer["size"]
        assert layer["offset"] == offset, "layers must sit end to end in the weight matrix"
        assert len(set(terms)) == len(terms), "a duplicated term makes one of them unreachable"
        assert "" not in terms
        offset += layer["size"]


def test_every_intent_has_a_stored_answer() -> None:
    """`reply()` on the phone returns nothing for an intent with no stored text."""
    bundle = _load("intent")
    missing = [label for label in bundle["classes"] if not bundle["intent_responses"].get(label)]
    assert not missing, f"intents with no answer: {missing}"


def test_descriptor_settings_are_recorded_with_the_weights() -> None:
    """The app trusts the constants it finds here rather than hardcoding its own."""
    descriptor = _load("glyph")["descriptor"]
    assert descriptor["glyph_size"] == 28
    assert descriptor["cell"] == 7
    assert descriptor["bins"] == 9
    assert descriptor["block"] == 2
    assert descriptor["zoning"] == 4


@pytest.mark.parametrize("name,floor", [("glyph", 0.75), ("symbol", 0.95)])
def test_exported_model_clears_its_accuracy_floor(name: str, floor: float) -> None:
    """Guards against exporting a model that trained into a degenerate corner.

    The floors are well below the measured scores (80.9% and 99.3%) on purpose: this is a
    tripwire for a broken export, not a benchmark.
    """
    metrics = _load(name)["metrics"]
    accuracy = metrics.get("accuracy", metrics.get("held_out_accuracy"))
    assert accuracy is not None, "an export without a measured score cannot be judged"
    assert accuracy >= floor


def test_portable_vectoriser_still_reproduces_scikit_learn() -> None:
    """The behaviour the TypeScript was ported from, checked against the real library."""
    model_path = REPO_ROOT / "models" / "bern_model.joblib"
    if not model_path.exists():
        pytest.skip("intent model has not been trained")

    import joblib

    from bern.portable_text import VectorizerSpec, union_row

    bundle = joblib.load(model_path)
    features = dict(bundle["pipeline"].steps)["features"]

    layers = []
    for name, vectorizer in features.transformer_list:
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

    messages = [
        "bakit mahal ang bigas",
        "magkano ang bigas ngayon?",
        "LAGAY MO SA BULSA MO",
        "2+2",
        "salamat!!",
        "   ",
    ]
    reference = features.transform(messages).toarray()
    weights = features.transformer_weights or {}
    portable = np.stack([union_row(layers, weights, text) for text in messages])
    assert float(np.abs(reference - portable).max()) < 1e-9


def test_normalised_glyph_tracks_opencv() -> None:
    """`normalise_glyph` must put the ink where OpenCV puts it, to within one level.

    OpenCV's bilinear path works in fixed point and rounds its last step; the portable code
    truncates the way `ndarray.astype` does. Over 200 random crops that shows up as 29 exactly
    identical outputs and a maximum difference of one intensity level, never a shifted
    stroke — the geometry agrees, only the final rounding differs.

    Exactness is not required: the phone's model was trained on `normalise_glyph`, so it sees
    what this test pins down. Chasing OpenCV's fixed-point rounding would be work spent on a
    difference no model can see.
    """
    cv2 = pytest.importorskip("cv2")

    from bern.handwriting import normalize_glyph
    from bern.portable_features import normalise_glyph

    rng = np.random.default_rng(3)
    for _ in range(20):
        crop = np.zeros((rng.integers(20, 90), rng.integers(20, 90)), np.uint8)
        top, left = rng.integers(0, 8, 2)
        crop[top:top + 18, left:left + 12] = 255
        crop[rng.integers(0, crop.shape[0]), rng.integers(0, crop.shape[1])] = 255

        reference = normalize_glyph(crop)
        portable = normalise_glyph(crop)
        assert portable.shape == (28, 28)
        difference = np.abs(reference.astype(int) - portable.astype(int))
        assert int(difference.max()) <= 1
        # Ink position is what the descriptor reads, so that part must be identical.
        assert np.array_equal(reference > 8, portable > 8) or int((reference > 8).sum()) == 0


def test_ink_mask_marks_the_stroke_not_the_paper() -> None:
    """The bug this guards: `prepare()` without a mask reads a photographed "T" as "8" at 1.00.

    `normaliseGlyph` scales whatever exceeds 8, so on a camera photo it fits the paper to the
    box and the character becomes a hole. The mask has to come first, and it has to work in
    both polarities — a whiteboard photo is the same problem upside down.
    """
    from bern.portable_features import normalise_glyph

    width, height = 160, 120
    for paper, ink in ((245, 20), (15, 235)):
        page = np.full((height, width), paper, np.uint8)
        page[30:90, 60:80] = ink
        page[20:34, 40:100] = ink

        mask = ink_mask(page)
        coverage = float((mask > 0).mean())
        # The stroke is roughly 1,600 of 19,200 pixels; the page is more than ten times that.
        assert 0.04 < coverage < 0.25, f"masked {coverage:.1%} of the page as ink"

        glyph = normalise_glyph(mask)
        # Enough ink survived the resize to be a character at all; the exact count depends on
        # the stroke width chosen above and is not the property under test.
        assert int((glyph > 8).sum()) > 150
        # The ink must land in the middle of the frame, not against an edge. Scaling to 86% of
        # the box is what leaves the margin, so a glyph touching a border means the mask found
        # something the size of the page.
        rows, cols = np.nonzero(glyph > 8)
        assert rows.min() >= 1 and rows.max() <= 26
        assert cols.min() >= 1 and cols.max() <= 26


def test_arithmetic_port_agrees_with_the_server() -> None:
    """`extract_math_expression` in TypeScript must read what Python reads.

    The intent model can only say *that* a message is arithmetic; producing the number is this
    port's job, and a divergence would mean the phone answering a question differently from the
    server that trained it. Cases are chosen to cover every branch of the extractor, including
    the Tagalog forms the dataset contains.
    """
    from bern.engine import extract_math_expression, safe_eval_math_ast

    cases = {
        "10 + 10": 20.0,
        "what is 24 * 15": 360.0,
        "15% of 400": 60.0,
        "divide 144 by 12": 12.0,
        "multiply 33 and 3": 99.0,
        "add 500 and 250": 750.0,
        "subtract 80 from 200": 120.0,
        "sum of 45 and 67": 112.0,
        "product of 7 and 8": 56.0,
        "difference between 50 and 20": 30.0,
        "what is 6 x 9": 54.0,
        "100 divided by 5": 20.0,
        "50 plus 25": 75.0,
        "75 minus 30": 45.0,
        "5 squared": 25.0,
        "2 ** 10": 1024.0,
        "(10 + 5) * 3": 45.0,
        "what is 3 + 4 * 5": 23.0,
        "pakisolve 18 * 4": 72.0,
        "ilan ang 120 / 4": 30.0,
        "compute 7 + 8 + 9": 24.0,
    }
    for message, expected in cases.items():
        expression = extract_math_expression(message)
        assert expression is not None, f"no expression found in {message!r}"
        value = safe_eval_math_ast(expression)
        assert value is not None, f"{message!r} produced an unparsable {expression!r}"
        assert abs(value - expected) < 1e-9, f"{message!r} gave {value}, expected {expected}"


def test_non_arithmetic_does_not_look_like_math() -> None:
    """A price question and a date must not be mistaken for sums."""
    from bern.engine import extract_math_expression

    for message in ["magkano ang bigas ngayon?", "kumusta ka naman?", "what time is it", "salamat po"]:
        assert extract_math_expression(message) is None, message
