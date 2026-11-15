from __future__ import annotations

import ast
import operator
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np

from bern.conversation import Memory, wants_explanation
from bern.ph_nutrition import get_shared_expert
from bern.ph_prices import PHPriceExpert
from bern.trainer import (
    DEFAULT_DATASET_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    load_dataset,
    normalize_text,
    save_dataset,
    train_bern_model,
)

_QUESTION_FILLERS = {
    "a", "an", "and", "are", "be", "can", "could", "did", "do", "does", "for",
    "from", "give", "have", "how", "i", "im", "i'm", "in", "is", "it", "me",
    "my", "of", "on", "or", "please", "show", "tell", "that", "the", "there",
    "this", "to", "u", "ur", "what", "whats", "what's", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your",
}

_SAFE_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_SAFE_UNARYOPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval_math_ast(expr: str) -> float | None:
    """Safely evaluate basic arithmetic expressions via AST without eval()."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent too large")
            return float(_SAFE_BINOPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
            return float(_SAFE_UNARYOPS[type(node.op)](_eval(node.operand)))
        raise ValueError("Unsupported math expression")

    try:
        result = _eval(tree)
        if not np.isfinite(result):
            return None
        return result
    except Exception:
        return None


def extract_math_expression(message: str) -> str | None:
    """Extract an arithmetic expression from natural language or raw math input."""
    text = message.strip().lower()

    # Handle "15% of 200" or "25 percent of 80"
    pct_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(-?\d+(?:\.\d+)?)", text)
    if pct_match:
        pct = float(pct_match.group(1))
        base = float(pct_match.group(2))
        return f"({pct} / 100) * {base}"

    # Handle natural phrases like "divide 144 by 12", "multiply 8 by 9", "add 5 and 3", "subtract 4 from 10"
    m = re.search(r"divide\s+(-?\d+(?:\.\d+)?)\s+by\s+(-?\d+(?:\.\d+)?)", text)
    if m:
        return f"{m.group(1)} / {m.group(2)}"
    m = re.search(r"multiply\s+(-?\d+(?:\.\d+)?)\s+(?:by|and)\s+(-?\d+(?:\.\d+)?)", text)
    if m:
        return f"{m.group(1)} * {m.group(2)}"
    m = re.search(r"add\s+(-?\d+(?:\.\d+)?)\s+(?:and|to)\s+(-?\d+(?:\.\d+)?)", text)
    if m:
        return f"{m.group(1)} + {m.group(2)}"
    m = re.search(r"subtract\s+(-?\d+(?:\.\d+)?)\s+from\s+(-?\d+(?:\.\d+)?)", text)
    if m:
        return f"{m.group(2)} - {m.group(1)}"

    # "sum of 45 and 67", "product of 7 and 8", "difference between 50 and 20"
    num = r"(-?\d+(?:\.\d+)?)"
    m = re.search(rf"(?:sum|total)\s+of\s+{num}\s+and\s+{num}", text)
    if m:
        return f"{m.group(1)} + {m.group(2)}"
    m = re.search(rf"product\s+of\s+{num}\s+and\s+{num}", text)
    if m:
        return f"{m.group(1)} * {m.group(2)}"
    m = re.search(rf"difference\s+(?:of|between)\s+{num}\s+and\s+{num}", text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return f"{max(a, b)} - {min(a, b)}"

    # Replace 'x' / spelled-out operators between numbers ("25 times 4", "81 divided by 9", "5 squared")
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s*squared", r"(\1**2)", text)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s*cubed", r"(\1**3)", cleaned)
    left, right = r"(?<=[\d)])\s*", r"\s*(?=[\d(])"
    cleaned = re.sub(left + r"x" + right, "*", cleaned)
    cleaned = re.sub(left + r"(?:times|multiplied by)" + right, "*", cleaned)
    cleaned = re.sub(left + r"(?:divided by|over)" + right, "/", cleaned)
    cleaned = re.sub(left + r"plus" + right, "+", cleaned)
    cleaned = re.sub(left + r"(?:minus|less)" + right, "-", cleaned)
    # Extract longest mathematical substring containing digits and operators
    candidates = re.findall(r"[\d\.\(\)\s+\-*/%^]+", cleaned)
    for cand in sorted(candidates, key=len, reverse=True):
        cand_s = cand.strip().replace("^", "**")
        if any(op in cand_s for op in ["+", "-", "*", "/", "%"]) and re.search(r"\d", cand_s):
            if safe_eval_math_ast(cand_s) is not None:
                return cand_s
    return None


_MATH_FILLER = {
    "how", "much", "is", "what", "whats", "what's", "calculate", "compute", "solve", "equals", "equal", "the", "answer", "of", "to",
    "plus", "minus", "times", "divided", "by", "please", "pls", "x", "ano", "ang", "ba", "result", "value", "sum", "total",
    "and", "multiplied", "over", "squared", "cubed", "product", "difference", "between", "add", "subtract", "multiply", "divide",
    "from", "pakicompute", "pakicalculate", "pakisolve", "pakisagot", "ilan", "kuwentahin", "less",
}


def is_pure_math(message: str) -> bool:
    """True for 'how much is 12 x 12' - arithmetic with only filler words around it."""
    words = re.findall(r"[a-z']+", message.lower())
    return bool(words) and all(w in _MATH_FILLER for w in words) and bool(re.search(r"\d", message))


_NAME_STRONG = [
    r"(?:my name is actually|my name is|my name's|name's|name is|you can call me|you may call me|just call me|please call me|call me|"
    r"they call me|everyone calls me|my friends call me|people call me|i'm known as|i am known as|i am called|i'm called|im called)\s+",
    r"(?:ako po si|ako si|pangalan ko po ay|pangalan ko ay|pangalan ko si|ang pangalan ko po ay|ang pangalan ko ay|ang pangalan ko si|"
    r"name ko si|name ko ay|tawagin mo akong|tawagin mo ako ng|ang tawag sa akin ay|tawag sa akin ay|ang tawag nila sa akin ay)\s+",
]
_NAME_WEAK = r"(?i:\bi am|\bi'm|\bim|\bthis is|\bit's)\s+"        # only accepted when the name is Capitalised
_NAME_TOKEN = r"([A-Za-z][A-Za-z\-']{1,24})"
_NAME_BLOCKED = {
    "good", "fine", "okay", "ok", "tired", "happy", "here", "back", "leaving", "sure", "ready", "not", "a", "an", "the", "called", "known",
    "actually", "bern", "from", "hungry", "sad", "new", "just", "also", "so", "very", "still", "going", "looking", "trying", "learning",
    "developer", "student", "filipino", "pinoy", "using", "building", "making", "testing", "asking", "wondering", "interested", "curious",
    "bored", "sick", "busy", "late", "early", "done", "glad", "sorry", "thinking", "wanting", "in", "at", "on", "with", "your", "my",
}


def extract_user_name(message: str, strong_only: bool = False) -> str | None:
    """Pull a person's name out of 'my name is Juan', 'ako si Ana', 'they call me Jay', 'i'm Pedro' ..."""
    text = message.strip()
    candidates: list[str] = []
    for prefix in _NAME_STRONG:
        for m in re.finditer(prefix + _NAME_TOKEN, text, re.IGNORECASE):
            candidates.append(m.group(1))
    if not strong_only:
        for m in re.finditer(_NAME_WEAK + r"([A-Z][a-zA-Z\-']{1,24})", text):
            candidates.append(m.group(1))
    for cand in candidates:
        if cand.lower() not in _NAME_BLOCKED:
            return cand[0].upper() + cand[1:]
    return None


# Rejection ("I don't know this yet") thresholds. The defaults are calibrated by cross-validation in scripts/train_bern.py
# and stored with the model (bundle["gating"]).
DEFAULT_GATING: dict[str, float] = {
    "low_conf": 0.35,       # reject when blended confidence is below this AND nearest-phrase similarity below low_conf_sim
    "low_conf_sim": 0.36,
    "partial_conf": 0.56,   # reject short messages with many unseen words unless confidence reaches this
    "partial_cov": 0.5,
    "oov_prob": 0.72,       # reject messages made only of unseen words unless the classifier is at least this sure
}


def compute_top_signals(bundle: dict[str, Any], text_norm: str, class_idx: int, limit: int = 5) -> list[str]:
    pipeline = bundle["pipeline"]
    features = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]

    word_vec = features.transformer_list[0][1]
    X_word = word_vec.transform([text_norm])
    word_names = word_vec.get_feature_names_out()
    coefs = clf.coef_[class_idx] if clf.coef_.shape[0] > 1 else clf.coef_[0]

    nonzero_cols = X_word.nonzero()[1]
    scored: list[tuple[float, str]] = []
    for col in nonzero_cols:
        weight = float(X_word[0, col] * coefs[col])
        if weight > 0:
            scored.append((weight, str(word_names[col])))

    scored.sort(reverse=True)
    return [tok for _, tok in scored[:limit]]


def predict_with_bundle(bundle: dict[str, Any], message: str, want_signals: bool = True) -> dict[str, Any]:
    """Pure scoring function: classifier probability blended with nearest-training-phrase similarity."""
    text_norm = normalize_text(message)
    pipeline = bundle["pipeline"]
    classes = bundle["classes"]
    features = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    train_matrix = bundle["train_matrix"]
    train_norms = bundle["train_norms"]
    train_labels = bundle["train_labels"]
    train_texts = bundle["train_texts"]

    X_feat = features.transform([text_norm])
    probas = clf.predict_proba(X_feat)[0]

    # Cosine similarity to all training patterns
    q_norm = float(np.sqrt(max(float(X_feat.power(2).sum()), 1e-12)))
    dots = np.asarray(train_matrix.dot(X_feat.T).toarray()).ravel()
    sims = dots / (train_norms * q_norm)

    best_sim_by_intent: dict[str, float] = {}
    best_pattern_by_intent: dict[str, str] = {}
    for idx, lbl in enumerate(train_labels):
        sim_val = float(sims[idx])
        if lbl not in best_sim_by_intent or sim_val > best_sim_by_intent[lbl]:
            best_sim_by_intent[lbl] = sim_val
            best_pattern_by_intent[lbl] = train_texts[idx]

    # Calibrated hybrid confidence: blends classifier probability & nearest semantic similarity
    ranked: list[dict[str, Any]] = []
    for idx, tag in enumerate(classes):
        prob = float(probas[idx])
        sim = best_sim_by_intent.get(tag, 0.0)
        blended = 0.68 * prob + 0.32 * sim
        ranked.append({"intent": tag, "confidence": round(blended, 4), "classifier_prob": round(prob, 4),
                       "similarity": round(sim, 4), "class_idx": idx})

    ranked.sort(key=lambda item: item["confidence"], reverse=True)
    best = ranked[0]
    top_signals = compute_top_signals(bundle, text_norm, best["class_idx"]) if want_signals else []

    # Content-token vocabulary coverage catches out-of-domain queries
    word_vec = features.transformer_list[0][1]
    vocab = word_vec.vocabulary_
    tokens = re.findall(r"\b\w+\b", text_norm)
    content_tokens = [t for t in tokens if t not in _QUESTION_FILLERS and not t.isdigit()]
    if content_tokens:
        vocab_coverage = round(sum(1 for t in content_tokens if t in vocab) / len(content_tokens), 4)
    else:
        vocab_coverage = 1.0

    return {
        "intent": best["intent"],
        "confidence": best["confidence"],
        "classifier_prob": best["classifier_prob"],
        "similarity": best["similarity"],
        "vocab_coverage": vocab_coverage,
        "content_token_count": len(content_tokens),
        "matched_pattern": best_pattern_by_intent.get(best["intent"], ""),
        "top_predictions": [{"intent": r["intent"], "confidence": r["confidence"], "classifier_prob": r["classifier_prob"],
                             "similarity": r["similarity"]} for r in ranked[:3]],
        "top_signals": top_signals,
    }


def gating_rejects(pred: dict[str, Any], gating: dict[str, float]) -> bool:
    """True when Bern should say 'I don't recognize this yet' (apart from the explicit out_of_scope class)."""
    cnt, cov = pred["content_token_count"], pred["vocab_coverage"]
    out_of_vocab = cnt >= 2 and cov == 0.0 and pred["classifier_prob"] < gating["oov_prob"]
    low_conf = pred["confidence"] < gating["low_conf"] and pred["similarity"] < gating["low_conf_sim"]
    partially_unseen = cnt >= 2 and cov <= gating["partial_cov"] and pred["confidence"] < gating["partial_conf"]
    return bool(out_of_vocab or low_conf or partially_unseen)


def _entity_name(entry: Any) -> str | None:
    """Pull a readable item name out of whatever shape the price expert returns.

    ``search_items`` yields item dicts directly, while the basket parser yields
    ``{"item": {...}, "qty": n}``. Reading ``entry["name"]`` on the second shape returns
    nothing and ``entry["item"]`` returns a dict, which is how a follow-up once resolved to
    ``magkano ang {'id': 'rice_regular', ...}``.
    """
    if isinstance(entry, str):
        return entry.strip() or None
    if not isinstance(entry, dict):
        return None
    for key in ("name", "item", "food"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested = value.get("name")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return None


class BernAI:
    """Bern — Trainable scikit-learn ML Assistant with live retraining and diagnostics."""

    def __init__(
        self,
        dataset_path: Path = DEFAULT_DATASET_PATH,
        model_path: Path = DEFAULT_MODEL_PATH,
        metrics_path: Path = DEFAULT_METRICS_PATH,
        confidence_threshold: float = 0.35,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        self.confidence_threshold = confidence_threshold
        self._lock = threading.RLock()
        self.sessions: dict[str, dict[str, Any]] = {}
        self.bundle: dict[str, Any] = {}
        self.price_expert = PHPriceExpert()
        self.nutrition_expert = get_shared_expert()
        self._handwriting_reader: Any = None   # built on first use, see handwriting_reader
        self.load_or_train()

    def load_or_train(self) -> None:
        with self._lock:
            if not self.model_path.exists():
                train_bern_model(self.dataset_path, self.model_path, self.metrics_path)
            self.bundle = joblib.load(self.model_path)

    def retrain(self) -> dict[str, Any]:
        with self._lock:
            metrics = train_bern_model(self.dataset_path, self.model_path, self.metrics_path, fast=True)
            self.bundle = joblib.load(self.model_path)
            return metrics

    def predict(self, message: str) -> dict[str, Any]:
        """Run raw ML classification + cosine similarity calibration + feature explanation."""
        start = time.perf_counter()
        with self._lock:
            result = predict_with_bundle(self.bundle, message)
        result["inference_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
        return result

    @property
    def handwriting_reader(self) -> Any:
        """The shared :class:`bern.handwriting.HandwritingReader`, created on first use.

        Built lazily on purpose: the reader pulls in OpenCV and a joblib model, and most
        conversations never touch an image. Importing it inside the property keeps text
        chat working even on a machine where the vision model has not been trained.
        """
        if self._handwriting_reader is None:
            with self._lock:
                if self._handwriting_reader is None:
                    from bern.handwriting import HandwritingReader

                    self._handwriting_reader = HandwritingReader()
        return self._handwriting_reader

    def read_handwriting(self, image_bytes: bytes, backend: str = "auto",
                         case_mode: str = "auto") -> dict[str, Any]:
        """Transcribe a photo of handwriting.

        Args:
            image_bytes: Raw JPEG/PNG/WebP bytes, as uploaded by the client.
            backend: ``auto`` (default), ``glyph``, ``trocr`` or ``tesseract``.
            case_mode: ``auto``, ``upper`` or ``lower``.

        Returns:
            ``{"text": str, "lines": [...], "backend": str, "confidence": float, ...}``
            plus Bern's own ``ai_name`` metadata so responses look like the rest of the API.

        Raises:
            ValueError: if the bytes are not a decodable image.
        """
        reader = self.handwriting_reader
        if backend != reader.requested_backend or case_mode != reader.case_mode:
            from bern.handwriting import HandwritingReader

            result = HandwritingReader(backend=backend, case_mode=case_mode).read_image(image_bytes)
        else:
            result = reader.read_image(image_bytes)
        result["ai_name"] = self.bundle.get("ai_name", "Bern")
        return result

    def build_study_pack(self, text: str, **options: Any) -> dict[str, Any]:
        """Turn notes or lecture text into a reviewer, quiz, mock exam and flashcards.

        Thin wrapper over :func:`bern.study.generate_study_pack` that stamps Bern's
        identity onto the result, matching the other API payloads.
        """
        from bern.study import generate_study_pack

        pack = generate_study_pack(text, **options)
        pack["ai_name"] = self.bundle.get("ai_name", "Bern")
        return pack

    def extract_document(self, data: bytes, filename: str = "") -> dict[str, Any]:
        """Read text out of an uploaded PDF, DOCX, image or text file.

        Raises:
            ValueError: if the file type is unsupported or the PDF is encrypted.
        """
        from bern.text_extract import ExtractionError, extract_text

        try:
            return extract_text(data, filename)
        except ExtractionError as exc:
            raise ValueError(str(exc)) from exc

    @property
    def gating(self) -> dict[str, float]:
        return {**DEFAULT_GATING, **self.bundle.get("gating", {})}

    def chat(self, message: str, session_id: str = "default", engine: str = "auto",
             use_llm: bool = True) -> dict[str, Any]:
        """Answer one message, using what was said earlier in this session.

        Three things happen here that a plain classifier cannot do:

        * **Memory.** Session transcripts are kept, and a follow-up such as "what about the
          other one?" is rewritten into the question it stands for before classification.
        * **Unseen questions.** When local routing declines a message and a language model is
          configured, the model answers instead of Bern giving up.
        * **Explanations.** "Bakit mahal ang bigas?" asks for a reason; a bag-of-words model
          has none to give, so a configured model answers it.

        Args:
            message: What the user typed.
            session_id: Conversation to continue. Sessions are in-memory and per-process.
            engine: ``"local"`` never uses a language model; ``"llm"`` and ``"auto"`` use it
                for the cases above and fall back locally when no key is set or the call
                fails.
            use_llm: Set False for deterministic, offline runs. The frozen benchmark uses
                it, so scoring never depends on a network call.

        Returns:
            The usual reply payload, plus ``conversation`` (what is remembered) and
            ``resolved_as`` (the rewritten question, when one was rewritten).
        """
        session = self.sessions.setdefault(
            session_id, {"user_name": None, "turns": 0, "memory": Memory()}
        )
        memory: Memory = session.setdefault("memory", Memory())

        resolved, subject = memory.resolve(message)
        result = self._answer(resolved, session_id)

        wants_reason = wants_explanation(message)
        if use_llm and engine in ("llm", "auto") and (result.get("needs_training") or wants_reason):
            generated = self._llm_answer(message, result, subject)
            if generated is not None:
                result = generated
            elif wants_reason:
                # No model available, and the message asks for a reason. Answering it with a
                # price table looks like an answer while being a non-answer, so say plainly
                # what Bern cannot do and keep the facts that are still useful.
                result["reply"] = (
                    "**I can't explain why** — I'm a small model and I have no way to reason "
                    "about causes.\n\nHere is what I do know:\n\n" + str(result.get("reply", ""))
                )

        result.setdefault("engine", "local")

        # Only record what actually happened: a declined question must not become the
        # subject of the next "what about that one?".
        if not result.get("needs_training"):
            memory.record(message, str(result.get("intent", "")), str(result.get("reply", "")),
                          list(result.get("entities", [])))

        result["conversation"] = memory.summary()
        result["resolved_as"] = resolved if subject else None
        return result

    def _llm_answer(self, message: str, local: dict[str, Any],
                    subject: str | None) -> dict[str, Any] | None:
        """Try the language model; return ``None`` to keep the local answer.

        The local answer is passed in as grounding, so the model can quote Bern's own price
        and nutrition figures instead of recalling them from training and getting them wrong.
        """
        try:
            from bern import llm
        except Exception:
            return None
        if not llm.is_available():
            return None

        context = ""
        if not wants_explanation(message) and local.get("intent") not in (None, "unknown"):
            context = str(local.get("reply", ""))[:1500]

        question = message if subject is None else f"{message} (about {subject})"
        try:
            answer = llm.TutorSession(notes=context).ask(question)["answer"].strip()
        except Exception:
            # A model that is down must not take the chat down with it.
            return None
        if not answer:
            return None

        return {
            "reply": answer,
            "intent": local.get("intent", "unknown"),
            "confidence": local.get("confidence", 0.0),
            "needs_training": False,
            "engine": "llm",
            "top_predictions": local.get("top_predictions", []),
            "top_signals": local.get("top_signals", []),
            "inference_ms": local.get("inference_ms", 0.0),
            "ai_name": self.bundle.get("ai_name", "Bern"),
            "model_version": self.bundle.get("version", "1.0.0"),
            "trained_at": self.bundle.get("trained_at"),
        }

    def _answer(self, message: str, session_id: str = "default") -> dict[str, Any]:
        """Single-message routing: no memory, no language model. See :meth:`chat`."""
        msg_clean = message.strip()
        if not msg_clean:
            return {
                "reply": "Type a message for Bern to classify and answer!",
                "intent": "empty",
                "confidence": 0.0,
                "needs_training": False,
                "top_predictions": [],
                "top_signals": [],
                "inference_ms": 0.0,
                "ai_name": self.bundle.get("ai_name", "Bern"),
                "model_version": self.bundle.get("version", "1.0.0"),
                "trained_at": self.bundle.get("trained_at"),
            }

        session = self.sessions.setdefault(
            session_id, {"user_name": None, "turns": 0, "memory": Memory()}
        )
        session["turns"] += 1

        # Items this answer is about; reported so the caller can remember them.
        entities: list[str] = []

        pred = self.predict(msg_clean)
        intent = pred["intent"]
        confidence = pred["confidence"]
        max_sim = pred["similarity"]

        # Philippine nutrition routing: calories / macros of foods, dishes, canned goods, McDo & Jollibee, whole meals
        nutri_signal = self.nutrition_expert.has_nutrition_signal(msg_clean)
        if nutri_signal or intent == "ph_nutrition_lookup":
            nutri_reply = self.nutrition_expert.try_answer(msg_clean, min_score=0.5 if nutri_signal else 0.62)
            if nutri_reply:
                for match in re.finditer(r"\*\*([^*\n]{2,40})\*\*", nutri_reply):
                    entities.append(match.group(1).strip())
                    if len(entities) >= 3:
                        break
                return {
                    "reply": nutri_reply,
                    "intent": "ph_nutrition_lookup",
                    "confidence": max(confidence, 0.9) if intent == "ph_nutrition_lookup" else max(confidence, 0.86),
                    "needs_training": False,
                    "entities": entities,
                    "top_predictions": pred["top_predictions"],
                    "top_signals": pred["top_signals"] or ["ph_nutrition_expert"],
                    "inference_ms": pred["inference_ms"],
                    "ai_name": self.bundle.get("ai_name", "Bern"),
                    "model_version": self.bundle.get("version", "1.0.0"),
                    "trained_at": self.bundle.get("trained_at"),
                }

        # Smart Philippine Price routing when user mentions a known PH good + price/budget words
        price_signals = ["magkano", "presyo", "price", "prices", "how much", "cost", "kasya", "mabibili", "palengke", "supermarket", "peso", "pesos", "₱", "php", "buy"]
        has_price_signal = any(re.search(rf"\b{re.escape(w)}\b", msg_clean.lower()) if w.isalpha() else w in msg_clean.lower() for w in price_signals)
        ph_hits = self.price_expert.search_items(msg_clean, limit=1, min_score=0.72) if has_price_signal else []
        ph_basket = self.price_expert.extract_multi_item_basket(msg_clean) if has_price_signal else []
        is_budget_ask = bool(
            re.search(r"(?:₱|php|pesos?|p)\s*[\d,]+|[\d,]+\s*(?:pesos?|php)", msg_clean.lower())
            and any(w in msg_clean.lower() for w in ["buy", "kasya", "budget", "mabibili", "worth", "can i get"])
        )
        sku_hits = self.price_expert.search_skus_for_message(msg_clean, limit=1, min_score=0.5) if (has_price_signal and not ph_hits and not ph_basket) else []
        if (has_price_signal and (ph_hits or ph_basket or sku_hits)) or is_budget_ask:
            has_qty_basket = any(b["qty"] != 1.0 for b in ph_basket) or (len(ph_basket) >= 2 and re.search(r"\d", msg_clean) is not None)
            if is_budget_ask or has_qty_basket:
                routed_intent = "ph_budget_calc"
            elif intent in {"ph_price_lookup", "ph_price_catalog", "ph_budget_calc"}:
                routed_intent = intent
            else:
                routed_intent = "ph_price_lookup"
            priced = self.price_expert.answer_price_query(msg_clean)
            # The items this answer is actually about. Taken from the same hits that routed
            # the message, so a follow-up refers to what was really discussed.
            for hit in list(ph_hits) + list(ph_basket):
                name = _entity_name(hit)
                if name and name not in entities:
                    entities.append(name)
            return {
                "reply": priced,
                "intent": routed_intent,
                "confidence": max(confidence, 0.88),
                "needs_training": False,
                "entities": entities,
                "top_predictions": pred["top_predictions"],
                "top_signals": pred["top_signals"] or ["ph_price_expert"],
                "inference_ms": pred["inference_ms"],
                "ai_name": self.bundle.get("ai_name", "Bern"),
                "model_version": self.bundle.get("version", "1.0.0"),
                "trained_at": self.bundle.get("trained_at"),
            }

        # Check explicit math expression even if worded unexpectedly
        math_expr = extract_math_expression(msg_clean)
        if math_expr is not None and (intent == "calculator" or confidence < 0.45 or is_pure_math(msg_clean)):
            val = safe_eval_math_ast(math_expr)
            if val is not None:
                formatted = int(val) if abs(val - round(val)) < 1e-9 else round(val, 6)
                return {
                    "reply": f"**{math_expr}** = **{formatted}**",
                    "intent": "calculator",
                    "confidence": max(confidence, 0.92) if intent == "calculator" else 0.90,
                    "needs_training": False,
                    "top_predictions": pred["top_predictions"],
                    "top_signals": pred["top_signals"] or ["math_ast"],
                    "inference_ms": pred["inference_ms"],
                    "ai_name": self.bundle.get("ai_name", "Bern"),
                    "model_version": self.bundle.get("version", "1.0.0"),
                    "trained_at": self.bundle.get("trained_at"),
                }

        # Check explicit name assignment first so unusual names aren't marked OOD
        explicit_name = extract_user_name(msg_clean)
        strong_name = extract_user_name(msg_clean, strong_only=True) is not None
        if explicit_name is not None and (intent == "user_name" or strong_name):
            session["user_name"] = explicit_name
            return {
                "reply": f"Nice to meet you, **{explicit_name}**! I'm Bern — I'll remember your name for this session.",
                "intent": "user_name",
                "confidence": max(confidence, 0.90),
                "needs_training": False,
                "top_predictions": pred["top_predictions"],
                "top_signals": pred["top_signals"],
                "inference_ms": pred["inference_ms"],
                "ai_name": self.bundle.get("ai_name", "Bern"),
                "model_version": self.bundle.get("version", "1.0.0"),
                "trained_at": self.bundle.get("trained_at"),
            }

        # Handle unknown / out-of-domain queries honestly
        is_out_of_scope = intent == "out_of_scope"
        needs_training = gating_rejects(pred, self.gating) or is_out_of_scope
        if needs_training:
            if is_out_of_scope:
                unknown_reply = (
                    "That's outside what I've been trained on so far — I focus on **Philippine prices**, **food nutrition facts**, "
                    "**peso counting** and my own API.\n\n"
                    "Click **Teach Bern This** below or visit the **Train Bern** tab if you'd like me to learn it!"
                )
            else:
                unknown_reply = (
                    f"I'm **Bern**, and I don't recognize this topic confidently yet "
                    f"(top guess: `{intent}` at {round(confidence * 100)}%).\n\n"
                    f"Click **Teach Bern This** below or visit the **Train Bern** tab to teach me how to answer this!"
                )
            return {
                "reply": unknown_reply,
                "intent": "unknown",
                "guessed_intent": intent,
                "confidence": confidence,
                "needs_training": True,
                "top_predictions": pred["top_predictions"],
                "top_signals": pred["top_signals"],
                "inference_ms": pred["inference_ms"],
                "ai_name": self.bundle.get("ai_name", "Bern"),
                "model_version": self.bundle.get("version", "1.0.0"),
                "trained_at": self.bundle.get("trained_at"),
            }

        # Dynamic skill handlers for recognized intents
        if intent == "calculator":
            if math_expr:
                val = safe_eval_math_ast(math_expr)
                if val is not None:
                    formatted = int(val) if abs(val - round(val)) < 1e-9 else round(val, 6)
                    reply = f"**{math_expr}** = **{formatted}**"
                else:
                    reply = "I recognized that as a math request, though I couldn't parse the expression. Try something like `calculate 24 * 15`."
            else:
                reply = "Give me numbers and an operator — for example: `calculate 125 * 8` or `15% of 400`!"
        elif intent in {"ph_price_lookup", "ph_price_catalog", "ph_budget_calc"}:
            reply = self.price_expert.answer_price_query(msg_clean)
            # Remember what was just discussed, so "what about the other one?" has something
            # to point at. Recorded from the search results, never parsed back out of prose.
            for item in self.price_expert.search_items(msg_clean, limit=4, min_score=0.45):
                name = _entity_name(item)
                if name and name not in entities:
                    entities.append(name)
        elif intent == "ph_nutrition_lookup":
            reply = self.nutrition_expert.try_answer(msg_clean, min_score=0.4) or self.nutrition_expert.answer_nutrition_query(msg_clean)
            # The matched food is the first bold span of the answer ("**Adobo** ...").
            for match in re.finditer(r"\*\*([^*\n]{2,40})\*\*", reply or ""):
                entities.append(match.group(1).strip())
                if len(entities) >= 3:
                    break
        elif intent == "time_date":
            try:
                now = datetime.now(ZoneInfo("Asia/Manila"))
                tz_label = "Asia/Manila"
            except Exception:
                now = datetime.now().astimezone()
                tz_label = now.tzname() or "Local"
            reply = f"It's currently **{now.strftime('%I:%M %p')}** on **{now.strftime('%A, %B %d, %Y')}** ({tz_label})."
        elif intent == "handwriting_ocr":
            reply = (
                "I can read handwriting! Send me a photo and I'll turn it into text.\n\n"
                "**API:** `POST /api/handwriting` with multipart field `image`, or JSON "
                "`{\"image_base64\": \"data:image/jpeg;base64,...\"}`.\n\n"
                "Block letters on clean paper work best — for cursive, enable the TrOCR backend."
            )
        elif intent == "study_materials":
            reply = (
                "I can turn your notes into a **reviewer**, **quiz**, **mock exam** and "
                "**flashcards** — all built from your own material, nothing invented.\n\n"
                "**API:** `POST /api/study` with multipart `file` (PDF, DOCX, image, or text) "
                "or JSON `{\"text\": \"...\"}`, plus optional `flashcards`, `quiz_items`, "
                "`true_false_items`.\n\n"
                "Handwritten notes work too — they are scanned first, then turned into questions."
            )
        elif intent == "file_scan":
            reply = (
                "Send me the file and I'll pull the text out.\n\n"
                "**API:** `POST /api/extract` with multipart `file`.\n\n"
                "Supported: PDF (text layer), DOCX, plain text, and photos of handwriting."
            )
        elif intent == "user_name":
            new_name = extract_user_name(msg_clean)
            if new_name:
                session["user_name"] = new_name
                reply = f"Nice to meet you, **{new_name}**! I'm Bern — I'll remember your name for this session."
            elif session.get("user_name"):
                reply = f"You told me your name is **{session['user_name']}**!"
            else:
                reply = "I don't know your name yet! Tell me something like `my name is Alex` and I'll remember it."
        else:
            responses = self.bundle["intent_responses"].get(intent, ["I'm Bern!"])
            reply = random.choice(responses)
            if intent == "greeting" and session.get("user_name"):
                reply = f"Hey **{session['user_name']}**! " + reply

        return {
            "reply": reply,
            "intent": intent,
            "confidence": confidence,
            "needs_training": False,
            "entities": entities,
            "top_predictions": pred["top_predictions"],
            "top_signals": pred["top_signals"],
            "inference_ms": pred["inference_ms"],
            "ai_name": self.bundle.get("ai_name", "Bern"),
            "model_version": self.bundle.get("version", "1.0.0"),
            "trained_at": self.bundle.get("trained_at"),
        }

    def get_dataset(self) -> dict[str, Any]:
        with self._lock:
            return load_dataset(self.dataset_path)

    def get_metrics(self) -> dict[str, Any]:
        with self._lock:
            if not self.metrics_path.exists():
                return self.retrain()
            import json
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)

    def teach(
        self,
        tag: str,
        patterns: list[str],
        responses: list[str] | None = None,
        description: str = "",
    ) -> dict[str, Any]:
        """Add or update an intent in Bern's dataset and retrain immediately."""
        clean_tag = re.sub(r"[^a-zA-Z0-9_\- ]", "", tag).strip().lower().replace(" ", "_")
        if not clean_tag:
            raise ValueError("Intent tag cannot be empty.")

        clean_patterns = [p.strip() for p in patterns if p and p.strip()]
        if not clean_patterns:
            raise ValueError("Provide at least one training example phrase.")

        clean_responses = [r.strip() for r in (responses or []) if r and r.strip()]

        with self._lock:
            dataset = load_dataset(self.dataset_path)
            intents = dataset.setdefault("intents", [])
            existing = next((item for item in intents if item.get("tag") == clean_tag), None)

            if existing:
                seen_p = {p.lower() for p in existing.get("patterns", [])}
                for p in clean_patterns:
                    if p.lower() not in seen_p:
                        existing.setdefault("patterns", []).append(p)
                        seen_p.add(p.lower())
                if clean_responses:
                    seen_r = set(existing.get("responses", []))
                    for r in clean_responses:
                        if r not in seen_r:
                            existing.setdefault("responses", []).append(r)
                            seen_r.add(r)
                if description.strip():
                    existing["description"] = description.strip()
            else:
                if not clean_responses:
                    raise ValueError("New intents require at least one response for Bern to reply with.")
                intents.append(
                    {
                        "tag": clean_tag,
                        "description": description.strip() or f"Custom trained skill: {clean_tag}",
                        "patterns": clean_patterns,
                        "responses": clean_responses,
                    }
                )

            # a phrase can only mean one thing: teaching Bern something it used to treat as out_of_scope (or another
            # intent) moves the identical phrase to the new intent instead of leaving contradictory examples behind
            norm_p = lambda t: re.sub(r"\s+", " ", t.strip().lower())  # noqa: E731
            taught = {norm_p(p) for p in clean_patterns}
            reassigned: list[dict[str, str]] = []
            for item in intents:
                if item.get("tag") == clean_tag:
                    continue
                kept = []
                for pattern in item.get("patterns", []):
                    if norm_p(pattern) in taught:
                        reassigned.append({"from": str(item.get("tag")), "phrase": pattern})
                    else:
                        kept.append(pattern)
                item["patterns"] = kept

            save_dataset(dataset, self.dataset_path)
            metrics = self.retrain()
            return {"status": "trained", "tag": clean_tag, "metrics": metrics, "reassigned": reassigned}

    def delete_intent(self, tag: str) -> dict[str, Any]:
        with self._lock:
            dataset = load_dataset(self.dataset_path)
            intents = dataset.get("intents", [])
            filtered = [item for item in intents if item.get("tag") != tag]
            if len(filtered) == len(intents):
                raise ValueError(f"Intent '{tag}' not found.")
            if len(filtered) < 2:
                raise ValueError("Bern needs at least 2 intents to train a classifier.")
            dataset["intents"] = filtered
            save_dataset(dataset, self.dataset_path)
            metrics = self.retrain()
            return {"status": "deleted", "tag": tag, "metrics": metrics}
