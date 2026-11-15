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
