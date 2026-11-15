#!/usr/bin/env python3
"""Check that the TypeScript runtime reproduces Python's numbers exactly.

The phone and the server compute the same 341-float descriptor and the same TF-IDF rows from
two separate implementations. If those drift, nothing crashes — the app just returns
confidently wrong characters. This script is the guard: it feeds identical inputs to both
sides and fails if they disagree beyond float32 rounding.

It needs `node` and the TypeScript compiler. Neither is a runtime dependency of Bern; both are
only needed to run this check.

    PYTHONPATH=. python3 scripts/check_ondevice_parity.py
    TS_COMPILER=/path/to/tsc PYTHONPATH=. python3 scripts/check_ondevice_parity.py
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
REPO = BASE_DIR
MODELS = REPO / "models" / "ondevice"
RN = REPO / "react-native"
GLYPH_COUNT = 52
#: float32 has about seven significant digits. The two implementations sum the same values in
#: a different order and one keeps its accumulators in float64, so agreement is "far below the
#: smallest change that can alter a prediction", not "bit-identical". Measured differences are
#: around 1e-7 for features and up to 1e-5 for probabilities with the largest weight matrices.
FEATURE_TOLERANCE = 1e-5
PROBABILITY_TOLERANCE = 1e-4


def find_typescript() -> str | None:
    """Locate `tsc`, preferring an explicit override over anything on PATH."""
    override = os.environ.get("TS_COMPILER")
    if override and Path(override).exists():
        return override
    found = shutil.which("tsc")
    if found:
        return found
    for candidate in ("/tmp/node_modules/.bin/tsc", str(REPO / "node_modules" / ".bin" / "tsc")):
        if Path(candidate).exists():
            return candidate
    return None


def make_inputs() -> dict:
    """Deterministic inputs: random noise plus structured strokes and blocks.

    Noise alone is a weak test — every glyph looks equally unlike every character, so both
    implementations can agree on nonsense. Structured glyphs put ink where the descriptor has
    to make a decision.
    """
    rng = np.random.default_rng(7)
    glyphs = [(rng.random((28, 28)) * 255).astype(np.uint8).ravel().tolist() for _ in range(40)]
    for offset in range(6):
        stroke = np.zeros((28, 28), np.uint8)
        stroke[:, 8 + offset : 10 + offset] = 200
        stroke[6:8, :] = 150
        glyphs.append(stroke.ravel().tolist())

        block = np.zeros((28, 28), np.uint8)
        block[5:23, 5:23] = 230
        glyphs.append(block.ravel().tolist())

    texts = [
        "bakit mahal ang bigas",
        "magkano ang bigas ngayon?",
        "hello po",
        "2+2",
        "Paano mag-tanim ng palay?",
        "salamat!!",
        "x",
        "LAGAY MO SA BULSA MO",
        "what is the price of rice in manila today",
        "₱50 + ₱20",
        "   ",
        "Bigas po ba ito?  MANILA   price  ",
        "ate, magkano ang isang kilong kamatis?",
        "Ano ang recipe ng adobong manok?",
        "buksan mo nga ang ilaw",
        "HELLO!!!",
        "ñandú",
        "a" * 50,
    ]
    assert len(glyphs) == GLYPH_COUNT
    return {"glyphs": glyphs, "texts": texts}


HARNESS = """
const fs = require('fs');
const F = require(__OUT_DIR__ + '/bernFeatures.js');
const O = require(__OUT_DIR__ + '/bernOnDevice.js');
const input = JSON.parse(fs.readFileSync(__INPUT_PATH__, 'utf8'));
const read = (n) => JSON.parse(fs.readFileSync(__MODELS_DIR__ + '/' + n + '.json', 'utf8'));

(async () => {
  const bern = await O.BernOnDevice.load(read('intent'), read('glyph'), read('symbol'));
  const out = { features: [], intents: [], glyphs: [], symbols: [] };
  for (const g of input.glyphs) out.features.push(Array.from(F.glyphFeatures(Uint8Array.from(g))));
  for (const t of input.texts) {
    const r = bern.classify(t);
    out.intents.push({ intent: r.intent, confidence: r.confidence });
  }
  for (const g of input.glyphs) {
    const r = bern.readGlyph(Uint8Array.from(g));
    out.glyphs.push({ char: r.char, confidence: r.confidence });
  }
  for (const g of input.glyphs) {
    const r = bern.readSymbol(Uint8Array.from(g));
    out.symbols.push({ char: r.char, confidence: r.confidence });
  }
  fs.writeFileSync(__OUTPUT_PATH__, JSON.stringify(out));
  console.log('ran ' + out.features.length + ' glyphs and ' + out.intents.length + ' messages');
})();
"""


def decode(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.float32)


def probabilities(feature_rows: np.ndarray, bundle: dict) -> np.ndarray:
    """Softmax per row, the same arithmetic the app performs."""
    coefficients = decode(bundle["coefficients"]).reshape(bundle["coef_shape"])
    intercept = decode(bundle["intercept"])
    scores = feature_rows @ coefficients.T + intercept
    scores = scores - scores.max(axis=1, keepdims=True)
    exponentiated = np.exp(scores)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def fold_case_twins(scores: np.ndarray, classes: list[str]) -> np.ndarray:
    """Pool each letter's two cases onto whichever form wins, as the app does.

    Applied per row: a single 2-D fancy-index would select rows instead of classes and quietly
    fold whole glyphs into each other.
    """
    folded = scores.copy()
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(classes):
        if len(label) == 1 and label.isalpha():
            groups.setdefault(label.lower(), []).append(index)
    twin_groups = [group for group in groups.values() if len(group) >= 2]

    for row in folded:
        for group in twin_groups:
            mass = float(row[group].sum())
            winner = group[int(np.argmax(row[group]))]
            row[group] = 0.0
            row[winner] = mass
    return folded


def main() -> int:
    if not MODELS.exists() or not (MODELS / "glyph.json").exists():
        print(f"No exported bundles in {MODELS}. Run scripts/export_ondevice.py first.", file=sys.stderr)
        return 2

    compiler = find_typescript()
    if compiler is None:
        print("No TypeScript compiler found. Install it with `npm install typescript`, or set "
              "TS_COMPILER to its path. This check needs node and tsc; the app needs neither.",
              file=sys.stderr)
        return 2
    if shutil.which("node") is None:
        print("No `node` on PATH; cannot run the TypeScript runtime.", file=sys.stderr)
        return 2

    workdir = Path(tempfile.mkdtemp(prefix="bern-parity-"))
    input_path = workdir / "inputs.json"
    output_path = workdir / "results.json"
    input_path.write_text(json.dumps(make_inputs()), encoding="utf-8")

    compile_result = subprocess.run(
        [compiler, "--outDir", str(workdir), "--module", "commonjs", "--target", "ES2020",
         "--strict", "--lib", "ES2020", "--skipLibCheck",
         str(RN / "bernFeatures.ts"), str(RN / "bernOnDevice.ts")],
        capture_output=True, text=True,
    )
    if compile_result.returncode != 0:
        print("TypeScript did not compile:\n" + compile_result.stdout + compile_result.stderr, file=sys.stderr)
        return 1

    harness = workdir / "harness.js"
    harness.write_text(
        HARNESS.replace("__OUT_DIR__", repr(str(workdir)))
        .replace("__INPUT_PATH__", repr(str(input_path)))
        .replace("__OUTPUT_PATH__", repr(str(output_path)))
        .replace("__MODELS_DIR__", repr(str(MODELS))),
        encoding="utf-8",
    )
    run = subprocess.run(["node", str(harness)], capture_output=True, text=True)
    if run.returncode != 0:
        print("The TypeScript runtime failed:\n" + run.stdout + run.stderr, file=sys.stderr)
        return 1
    print("  node: " + run.stdout.strip())

    from bern.portable_features import glyph_features
    from bern.portable_text import VectorizerSpec, union_row

    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    results = json.loads(output_path.read_text(encoding="utf-8"))
    failures = 0

    features = np.stack([glyph_features(np.array(glyph, np.uint8).reshape(28, 28)) for glyph in inputs["glyphs"]])
    difference = float(np.abs(features - np.array(results["features"], np.float32)).max())
    status = "ok" if difference < FEATURE_TOLERANCE else "FAIL"
    print(f"  glyph features      max diff {difference:.2e}   {status}")
    failures += difference >= FEATURE_TOLERANCE

    intent_bundle = json.loads((MODELS / "intent.json").read_text(encoding="utf-8"))
    layers = [
        (layer["name"], VectorizerSpec(
            terms=layer["terms"].split("\n"), idf=decode(layer["idf"]), analyzer=layer["analyzer"],
            ngram_range=tuple(layer["ngram_range"]), sublinear_tf=layer["sublinear_tf"]))
        for layer in intent_bundle["layers"]
    ]
    weights = {layer["name"]: layer["weight"] for layer in intent_bundle["layers"]}
    rows = np.stack([union_row(layers, weights, text) for text in inputs["texts"]])
    network = probabilities(rows, intent_bundle)
    predicted = [intent_bundle["classes"][int(index)] for index in network.argmax(axis=1)]
    delivered = [entry["intent"] for entry in results["intents"]]
    agreement = sum(a == b for a, b in zip(predicted, delivered))
    difference = float(np.abs(network.max(axis=1) - np.array([e["confidence"] for e in results["intents"]])).max())
    status = "ok" if agreement == len(predicted) and difference < PROBABILITY_TOLERANCE else "FAIL"
    print(f"  intents             {agreement}/{len(predicted)} agree, max diff {difference:.2e}   {status}")
    failures += status == "FAIL"
    for text, expected, got in zip(inputs["texts"], predicted, delivered):
        if expected != got:
            print(f"    mismatch {text!r}: python={expected} typescript={got}")

    for name, key, fold in (("glyph", "glyphs", True), ("symbol", "symbols", False)):
        bundle = json.loads((MODELS / f"{name}.json").read_text(encoding="utf-8"))
        scores = probabilities(features, bundle)
        if fold:
            scores = fold_case_twins(scores, bundle["classes"])
        labels = [bundle["classes"][int(index)] for index in scores.argmax(axis=1)]
        delivered = [entry["char"] for entry in results[key]]
        agreement = sum(a == b for a, b in zip(labels, delivered))
        difference = float(np.abs(scores.max(axis=1) - np.array([e["confidence"] for e in results[key]])).max())
        status = "ok" if agreement == len(labels) and difference < PROBABILITY_TOLERANCE else "FAIL"
        print(f"  {name + ' reading':<19} {agreement}/{len(labels)} agree, max diff {difference:.2e}   {status}")
        failures += status == "FAIL"

    shutil.rmtree(workdir, ignore_errors=True)
    if failures:
        print("\nFAILED: the TypeScript runtime does not match Python.", file=sys.stderr)
        return 1
    print("\nTypeScript and Python agree on every input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
