from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = BASE_DIR / "data" / "bern_dataset.json"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "bern_model.joblib"
DEFAULT_METRICS_PATH = BASE_DIR / "models" / "metrics.json"


def normalize_text(text: str) -> str:
    """Clean text while preserving math symbols and alphanumeric tokens."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


# Hyper-parameters chosen by cross-validated model selection (scripts/train_bern.py -> models/training_report.json)
DEFAULT_PARAMS: dict[str, Any] = {
    "word_ngram": [1, 2],
    "char_ngram": [3, 5],
    "word_weight": 1.4,
    "char_weight": 0.6,
    "C": 8.0,
    "oos_weight": 1.0,
}
OOS_LABEL = "out_of_scope"


class BernClassifier(LogisticRegression):
    """LogisticRegression with 'balanced' class weights and an adjustable weight for the out_of_scope class.

    ``out_of_scope`` is the biggest class (it holds every kind of off-topic message), so its weight can be
    turned down to stop it swallowing short in-domain phrases such as "yo", "thx" or "ty".
    """

    def __init__(self, C: float = 8.0, max_iter: int = 1500, random_state: int = 42, oos_weight: float = 1.0,
                 tol: float = 1e-4) -> None:
        super().__init__(C=C, max_iter=max_iter, random_state=random_state, tol=tol)
        self.oos_weight = oos_weight

    def fit(self, X, y, sample_weight=None):  # noqa: N803 - sklearn naming
        classes, counts = np.unique(y, return_counts=True)
        n, k = len(y), len(classes)
        weights = {c: n / (k * cnt) for c, cnt in zip(classes, counts)}
        if OOS_LABEL in weights:
            weights[OOS_LABEL] *= self.oos_weight
        self.class_weight = weights
        return super().fit(X, y, sample_weight)


def build_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    """Build Bern's hybrid word + character n-gram TF-IDF + LogisticRegression pipeline."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    features = FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(p["word_ngram"]),
                    lowercase=True,
                    sublinear_tf=True,
                    min_df=1,
                    token_pattern=r"(?u)\b\w+\b|[+\-*/=()]",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=tuple(p["char_ngram"]),
                    lowercase=True,
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
        ],
        transformer_weights={"word_tfidf": p["word_weight"], "char_tfidf": p["char_weight"]},
    )
    classifier = BernClassifier(C=p["C"], max_iter=1500, random_state=42, oos_weight=p["oos_weight"])
    return Pipeline([("features", features), ("clf", classifier)])


def load_dataset(dataset_path: Path = DEFAULT_DATASET_PATH) -> dict[str, Any]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(dataset: dict[str, Any], dataset_path: Path = DEFAULT_DATASET_PATH) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def make_bundle(pipeline: Pipeline, texts: list[str], labels: list[str]) -> dict[str, Any]:
    """Minimal model bundle (what BernAI.predict needs) for an already fitted pipeline - used for cross-validated calibration."""
    feature_union = pipeline.named_steps["features"]
    matrix = feature_union.transform(texts)
    norms = np.sqrt(np.maximum(np.asarray(matrix.power(2).sum(axis=1)).ravel(), 1e-12))
    return {"pipeline": pipeline, "classes": list(pipeline.named_steps["clf"].classes_), "train_texts": texts,
            "train_labels": labels, "train_matrix": matrix, "train_norms": norms}


DEFAULT_AUGMENT = True   # use template-generated phrases (data/augmented_examples.json) for training only


def read_examples(dataset: dict[str, Any], augment: bool) -> tuple[list[str], list[str], list[bool], dict[str, list[str]], dict[str, str]]:
    """Return (texts, labels, is_generated, intent_responses, intent_descriptions)."""
    texts: list[str] = []
    labels: list[str] = []
    generated: list[bool] = []
    intent_responses: dict[str, list[str]] = {}
    intent_descriptions: dict[str, str] = {}
    for intent in dataset.get("intents", []):
        tag = str(intent.get("tag", "")).strip()
        if not tag:
            continue
        patterns = [str(p).strip() for p in intent.get("patterns", []) if str(p).strip()]
        responses = [str(r).strip() for r in intent.get("responses", []) if str(r).strip()]
        if not patterns or not responses:
            continue
        intent_responses[tag] = responses
        intent_descriptions[tag] = str(intent.get("description", "")).strip()
        for p in patterns:
            texts.append(normalize_text(p))
            labels.append(tag)
            generated.append(False)
    if augment:
        from bern.augment import load_augmented

        have = set(texts)
        for ex in load_augmented():
            t = normalize_text(ex["text"])
            if ex["intent"] in intent_responses and t not in have:
                have.add(t)
                texts.append(t)
                labels.append(ex["intent"])
                generated.append(True)
    return texts, labels, generated, intent_responses, intent_descriptions


def train_bern_model(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    params: dict[str, Any] | None = None,
    augment: bool | None = None,
    fast: bool = False,
) -> dict[str, Any]:
    """Train Bern from dataset JSON, run real held-out evaluation, and persist artifacts.

    ``fast=True`` (used by live "teach" retraining) skips the cross-validation and only runs the 80/20 split.

    Evaluation (80/20 split + 5-fold CV) only ever scores the hand-written phrases; template-generated phrases
    (``augment``) are added to the training side of every split.
    """
    start = time.perf_counter()
    dataset = load_dataset(dataset_path)
    tuned = dataset.get("training", {})            # settings chosen by scripts/train_bern.py travel with the dataset
    use_params = {**DEFAULT_PARAMS, **tuned.get("params", {}), **(params or {})}
    use_aug = tuned.get("augment", DEFAULT_AUGMENT) if augment is None else augment

    all_texts, all_labels, is_gen, intent_responses, intent_descriptions = read_examples(dataset, use_aug)
    texts = [t for t, g in zip(all_texts, is_gen) if not g]
    labels = [l for l, g in zip(all_labels, is_gen) if not g]
    gen_texts = [t for t, g in zip(all_texts, is_gen) if g]
    gen_labels = [l for l, g in zip(all_labels, is_gen) if g]

    if len(set(labels)) < 2:
        raise ValueError("Need at least 2 distinct intents with examples to train Bern.")

    counts = Counter(labels)

    # Honest evaluation on classes with >= 3 examples so stratified 80/20 split is reliable
    # (Falls back to >= 2 if dataset is still small; final model always trains on 100% of intents)
    eval_indices = [i for i, lbl in enumerate(labels) if counts[lbl] >= 3]
    if len({labels[i] for i in eval_indices}) < 2:
        eval_indices = [i for i, lbl in enumerate(labels) if counts[lbl] >= 2]
    eval_texts = [texts[i] for i in eval_indices]
    eval_labels = [labels[i] for i in eval_indices]
    eval_counts = Counter(eval_labels)
    n_eval_classes = len(eval_counts)

    test_accuracy: float | None = None
    weighted_f1: float | None = None
    macro_f1: float | None = None
    cv_accuracy: float | None = None
    cv_macro_f1: float | None = None
    per_intent_metrics: dict[str, dict[str, Any]] = {}
    eval_train_count = len(texts)
    eval_test_count = 0

    if n_eval_classes >= 2 and len(eval_texts) >= n_eval_classes * 2:
        min_test_size = max(n_eval_classes, int(round(len(eval_texts) * 0.2)))
        # Ensure at least one sample per class remains in train
        if len(eval_texts) - min_test_size >= n_eval_classes:
            X_train, X_test, y_train, y_test = train_test_split(
                eval_texts,
                eval_labels,
                test_size=min_test_size,
                random_state=42,
                stratify=eval_labels,
            )
            eval_pipe = build_pipeline(use_params)
            eval_pipe.fit(X_train + gen_texts, y_train + gen_labels)
            y_pred = eval_pipe.predict(X_test)

            test_accuracy = float(accuracy_score(y_test, y_pred))
            weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            eval_train_count = len(X_train) + len(gen_texts)
            eval_test_count = len(X_test)

            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            for tag in sorted(intent_responses.keys()):
                if tag in report:
                    per_intent_metrics[tag] = {
                        "precision": round(float(report[tag]["precision"]), 4),
                        "recall": round(float(report[tag]["recall"]), 4),
                        "f1": round(float(report[tag]["f1-score"]), 4),
                        "test_support": int(report[tag]["support"]),
                        "total_examples": int(counts[tag]),
                    }
                else:
                    per_intent_metrics[tag] = {
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "test_support": 0,
                        "total_examples": int(counts[tag]),
                    }

        min_class_count = min(eval_counts.values())
        if min_class_count >= 2 and not fast:
            n_splits = min(5, min_class_count)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            ev_t, ev_l = np.array(eval_texts, dtype=object), np.array(eval_labels)

            def _fold(tr: np.ndarray, te: np.ndarray) -> tuple[float, float]:
                fold = build_pipeline(use_params)
                fold.fit(list(ev_t[tr]) + gen_texts, list(ev_l[tr]) + gen_labels)
                pred = fold.predict(list(ev_t[te]))
                return accuracy_score(ev_l[te], pred), f1_score(ev_l[te], pred, average="macro", zero_division=0)

            scores = Parallel(n_jobs=-1)(delayed(_fold)(tr, te) for tr, te in cv.split(ev_t, ev_l))
            cv_accuracy = float(np.mean([a for a, _ in scores]))
            cv_macro_f1 = float(np.mean([f for _, f in scores]))

    # Refit final deployed pipeline on 100% of current training examples (+ generated phrases)
    texts_all, labels_all = texts + gen_texts, labels + gen_labels
    production_pipeline = build_pipeline(use_params)
    production_pipeline.fit(texts_all, labels_all)

    # Precompute normalized feature vectors for fast cosine-similarity calibration
    feature_union = production_pipeline.named_steps["features"]
    train_matrix = feature_union.transform(texts_all)
    norms = np.asarray(train_matrix.power(2).sum(axis=1)).ravel()
    norms = np.sqrt(np.maximum(norms, 1e-12))

    trained_at = datetime.now(timezone.utc).isoformat()
    train_duration_ms = round((time.perf_counter() - start) * 1000.0, 2)

    version_num = dataset.get("version", "1.0.0")
    model_bundle = {
        "ai_name": dataset.get("ai_name", "Bern"),
        "version": version_num,
        "trained_at": trained_at,
        "pipeline": production_pipeline,
        "classes": list(production_pipeline.named_steps["clf"].classes_),
        "intent_responses": intent_responses,
        "intent_descriptions": intent_descriptions,
        "train_texts": texts_all,
        "train_labels": labels_all,
        "train_matrix": train_matrix,
        "train_norms": norms,
        "params": use_params,
        "gating": tuned.get("gating", {}),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)

    metrics = {
        "ai_name": model_bundle["ai_name"],
        "version": version_num,
        "trained_at": trained_at,
        "train_duration_ms": train_duration_ms,
        "num_intents": len(intent_responses),
        "num_examples": len(texts),
        "generated_examples": len(gen_texts),
        "eval_train_examples": eval_train_count,
        "eval_test_examples": eval_test_count,
        "test_accuracy": round(test_accuracy, 4) if test_accuracy is not None else None,
        "cv_accuracy": round(cv_accuracy, 4) if cv_accuracy is not None else None,
        "cv_macro_f1": round(cv_macro_f1, 4) if cv_macro_f1 is not None else None,
        "weighted_f1": round(weighted_f1, 4) if weighted_f1 is not None else None,
        "macro_f1": round(macro_f1, 4) if macro_f1 is not None else None,
        "algorithm": "FeatureUnion(Word TF-IDF + Char-WB TF-IDF) -> LogisticRegression (balanced, tunable out_of_scope weight)",
        "params": use_params,
        "per_intent": per_intent_metrics,
    }

    # keep the held-out benchmark results of the last full training run (scripts/train_bern.py) next to the metrics
    if metrics_path.exists():
        try:
            previous = json.loads(metrics_path.read_text("utf-8"))
            if "benchmark" in previous:
                metrics["benchmark"] = previous["benchmark"]
        except (OSError, json.JSONDecodeError):
            pass

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    """``python3 -m bern.trainer`` — retrain the intent model from ``data/``.

    ``--fast`` skips the model search and calibration and just refits, which is what
    ``POST /api/train`` does between phrases. Without it the full search runs, which is the
    path to take after changing the dataset rather than after adding one sentence.
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Train Bern's intent model from data/bern_dataset.json."
    )
    parser.add_argument("--dataset", default="data/bern_dataset.json",
                        help="training dataset (default: data/bern_dataset.json)")
    parser.add_argument("--model", default="models/bern_model.joblib",
                        help="output model bundle (default: models/bern_model.joblib)")
    parser.add_argument("--metrics", default="models/metrics.json",
                        help="output metrics report (default: models/metrics.json)")
    parser.add_argument("--fast", action="store_true",
                        help="skip model search and calibration; refit only (~10 s)")
    args = parser.parse_args()

    result = train_bern_model(
        Path(args.dataset), Path(args.model), Path(args.metrics), fast=args.fast
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    # Running as ``python3 -m bern.trainer`` makes this file ``__main__``, and any class
    # defined here would be pickled as ``__main__.BernClassifier`` — a name that does not
    # exist when ``bern.engine`` loads the bundle from another process, so the model file
    # would be unloadable by the very code that is meant to use it. Delegating to the
    # canonical module keeps the pickle path as ``bern.trainer.BernClassifier``.
    from bern.trainer import main as _train_main

    _train_main()
