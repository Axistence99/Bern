"""TF-IDF text features without scikit-learn.

The intent model is a ``Pipeline`` of two ``TfidfVectorizer`` instances feeding a logistic
regression. A phone cannot load that pickle, so this module rebuilds the same numbers from
plain data: a vocabulary, an IDF vector and a weight matrix.

Everything here follows scikit-learn's semantics rather than a convenient simplification,
because those are the semantics the shipped weights were fitted against:

* ``sublinear_tf`` replaces each count ``c`` with ``1 + ln(c)``.
* ``idf`` is the *smooth* variant, ``ln((1 + n) / (1 + df)) + 1``, and is baked into the
  exported vector rather than recomputed.
* each vectorizer L2-normalises its own row, and only then does the ``FeatureUnion`` apply
  its per-transformer weights.

The character analyser is the fiddly one. ``char_wb`` builds n-grams *inside* word
boundaries, padding each word with a single space on both sides, and yields a short word
only once rather than once per n. Both quirks change the feature set, so both are reproduced.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

#: Unicode-aware word runs. Python's ``\w`` is Unicode by default; JavaScript's is not, so the
#: TypeScript port spells this out as ``[\p{L}\p{N}_]``.
WORD_PATTERN = re.compile(r"\w+|[+\-*/=()]", re.UNICODE)


@dataclass
class VectorizerSpec:
    """Everything needed to reproduce one ``TfidfVectorizer``'s output."""

    terms: list[str]
    idf: np.ndarray
    analyzer: str
    ngram_range: tuple[int, int]
    sublinear_tf: bool
    index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {term: position for position, term in enumerate(self.terms)}


def word_tokens(document: str) -> list[str]:
    """Split a document on Bern's word pattern, plus standalone operators."""
    return WORD_PATTERN.findall(document)


def char_wb_tokens(document: str, minimum: int, maximum: int) -> list[str]:
    """Character n-grams restricted to word interiors, spaces padding each word.

    scikit-learn emits a word shorter than ``n`` exactly once (guarding with ``if offset == 0:
    break``), which is why the inner loop counts rather than assuming the word is long
    enough.
    """
    tokens: list[str] = []
    append = tokens.append
    for word in document.split():
        padded = f" {word} "
        length = len(padded)
        for n in range(minimum, maximum + 1):
            offset = 0
            append(padded[0:n])
            while offset + n < length:
                offset += 1
                append(padded[offset : offset + n])
            if offset == 0:
                break
    return tokens


def tokens_for(spec: VectorizerSpec, document: str) -> list[str]:
    """Return the analyser's tokens, including the joined n-grams for word mode.

    Lowercasing happens here rather than inside either analyser, because scikit-learn runs
    the preprocessor over the whole document first: the character analyser sees lowercased
    text too. Lowercasing only in the word path leaves every upper-case message with zero
    recognisable characters and a nearly empty feature row.
    """
    document = document.lower()
    minimum, maximum = spec.ngram_range
    if spec.analyzer == "char_wb":
        return char_wb_tokens(document, minimum, maximum)

    tokens = word_tokens(document)
    if maximum == 1:
        return tokens

    # scikit-learn emits unigrams in order, then bigrams, then trigrams, and so on. The order
    # only matters for counting, which is commutative, but keeping it makes the two
    # implementations easy to compare side by side.
    original = tokens
    generated = list(original) if minimum == 1 else []
    count = len(original)
    for n in range(max(minimum, 2), min(maximum + 1, count + 1)):
        for start in range(count - n + 1):
            generated.append(" ".join(original[start : start + n]))
    return generated


def tfidf_row(spec: VectorizerSpec, document: str) -> np.ndarray:
    """One L2-normalised TF-IDF row, shaped like the vectorizer's output."""
    counts: dict[int, float] = {}
    for token in tokens_for(spec, document):
        position = spec.index.get(token)
        if position is not None:
            counts[position] = counts.get(position, 0.0) + 1.0

    row = np.zeros(len(spec.terms), dtype=np.float64)
    if not counts:
        return row

    indices = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    values = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
    if spec.sublinear_tf:
        values = 1.0 + np.log(values)

    weighted = values * spec.idf[indices]
    norm = math.sqrt(float(np.dot(weighted, weighted)))
    # A row of all-zero IDF weights (possible when every term is absent from the corpus)
    # leaves the norm at zero; scikit-learn leaves such a row untouched rather than dividing.
    if norm > 0.0:
        weighted = weighted / norm

    row[indices] = weighted
    return row


def union_row(
    layers: list[tuple[str, VectorizerSpec]], weights: dict[str, float], document: str
) -> np.ndarray:
    """The ``FeatureUnion`` output: each vectorizer's row scaled by its own weight.

    ``layers`` carries the order the vectorizers were listed in when the model was fitted,
    because that is the order their columns are concatenated in. Sorting the names instead
    silently shifts every column by one block — a mistake worth 0.8 in the first column and
    a wrong answer at the end.
    """
    parts = [tfidf_row(spec, document) * weights.get(name, 1.0) for name, spec in layers]
    return np.concatenate(parts) if parts else np.zeros(0)


def predict_proba(feature_row: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    """Softmax over ``coef @ x + intercept``, matching ``LogisticRegression.predict_proba``."""
    scores = coef @ feature_row + intercept
    scores = scores - scores.max()
    exponentiated = np.exp(scores)
    return exponentiated / exponentiated.sum()
