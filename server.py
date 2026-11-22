from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import tornado.ioloop
import tornado.web

from bern.engine import BernAI
from bern.money_vision import PhilippineMoneyCounter, generate_sample_ph_money_image

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
RN_DIR = BASE_DIR / "react-native"

bern_ai = BernAI()
money_counter = PhilippineMoneyCounter()


def llm_available() -> bool:
    """Whether an optional language model is configured. Never raises."""
    try:
        from bern import llm

        return llm.is_available()
    except Exception:
        return False


class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self) -> None:
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def options(self, *args: Any, **kwargs: Any) -> None:
        self.set_status(204)
        self.finish()

    def write_json(self, payload: Any, status_code: int = 200) -> None:
        self.set_status(status_code)
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.finish(json.dumps(payload, ensure_ascii=False))

    def parse_json_body(self) -> dict[str, Any]:
        if not self.request.body:
            return {}
        try:
            data = json.loads(self.request.body.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object.")
            return data
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON payload: {exc}") from exc


class HomeHandler(BaseHandler):
    def head(self) -> None:
        index_path = WEB_DIR / "index.html"
        if not index_path.exists():
            self.set_status(404)
            self.finish()
            return
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.set_header("Cache-Control", "no-store")
        self.finish()

    def get(self) -> None:
        index_path = WEB_DIR / "index.html"
        if not index_path.exists():
            self.write_json(
                {"error": "Web UI not found", "expected": str(index_path),
                 "hint": "Check out the full repository, or use the API directly — see README."},
                status_code=404,
            )
            return
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.set_header("Cache-Control", "no-store")
        self.finish(index_path.read_text(encoding="utf-8"))


class HealthHandler(BaseHandler):
    def get(self) -> None:
        metrics = bern_ai.get_metrics()
        self.write_json(
            {
                "status": "ok",
                "ai_name": metrics.get("ai_name", "Bern"),
                "version": metrics.get("version", "1.0.0"),
                "num_intents": metrics.get("num_intents"),
                "num_examples": metrics.get("num_examples"),
                "test_accuracy": metrics.get("test_accuracy"),
                "ph_price_items": len(bern_ai.price_expert.items),
                "nutrition_foods": len(bern_ai.nutrition_expert.foods),
                "money_vision_ready": True,
                # Reported so a client can tell the user what is possible before they type.
                # With no key, explanations and unseen questions are answered locally or
                # declined honestly rather than generated.
                "language_model": llm_available(),
            }
        )


class ChatHandler(BaseHandler):
    def post(self) -> None:
        try:
            body = self.parse_json_body()
            message = str(body.get("message", ""))
            session_id = str(body.get("session_id", "default"))
            # "auto" uses a language model only when one is configured, and falls back to
            # the local answer otherwise, so an install with no key is unchanged.
            engine = str(body.get("engine", "auto")).lower()
            if engine not in ("local", "llm", "auto"):
                self.write_json({"error": "engine must be 'local', 'llm' or 'auto'."},
                                status_code=400)
                return
            result = bern_ai.chat(message=message, session_id=session_id, engine=engine)
            self.write_json(result)
        except ValueError as exc:
            self.write_json({"error": str(exc)}, status_code=400)


class PredictHandler(BaseHandler):
    def post(self) -> None:
        try:
            body = self.parse_json_body()
            text = str(body.get("text", body.get("message", ""))).strip()
            if not text:
                self.write_json({"error": "Missing 'text' field."}, status_code=400)
                return
            self.write_json(bern_ai.predict(text))
        except ValueError as exc:
            self.write_json({"error": str(exc)}, status_code=400)


class MetricsHandler(BaseHandler):
    def get(self) -> None:
        self.write_json(bern_ai.get_metrics())


class TrainingReportHandler(BaseHandler):
    """GET /api/training-report - model selection, held-out benchmark (before vs after) and the development log."""

    def get(self) -> None:
        path = BASE_DIR / "models" / "training_report.json"
        if not path.exists():
            self.write_json({"error": "No training report yet. Run: python3 scripts/train_bern.py"}, status_code=404)
            return
        self.write_json(json.loads(path.read_text(encoding="utf-8")))


class DatasetHandler(BaseHandler):
    def get(self) -> None:
        self.write_json(bern_ai.get_dataset())


class LearnHandler(BaseHandler):
    """POST /api/learn - state a fact; it is checked against Bern's data before storage."""

    def post(self) -> None:
        try:
            body = self.parse_json_body()
            message = str(body.get("message") or body.get("text") or "").strip()
            if not message:
                self.write_json({"error": "Provide a message stating a fact."}, status_code=400)
                return
            result = bern_ai.learn_claim(
                message,
                session_id=str(body.get("session_id") or "default"),
                override=bool(body.get("override")),
            )
            self.write_json(result)
        except ValueError as exc:
            self.write_json({"error": str(exc)}, status_code=400)


class LearnedHandler(BaseHandler):
    """GET /api/learned - what has been learned, and how each claim was rated."""

    def get(self) -> None:
        from bern import memory

        self.write_json({
            "claims": memory.stored_claims(),
            "counts": memory.count_by_verdict(),
        })


class TrainHandler(BaseHandler):
    def post(self) -> None:
        try:
            body = self.parse_json_body()
            tag = body.get("tag")
            if tag:
                patterns = body.get("patterns", [])
                if isinstance(patterns, str):
                    patterns = [line.strip() for line in patterns.splitlines() if line.strip()]
                responses = body.get("responses", [])
                if isinstance(responses, str):
                    responses = [line.strip() for line in responses.splitlines() if line.strip()]
                description = str(body.get("description", ""))
                res = bern_ai.teach(
                    tag=str(tag),
                    patterns=list(patterns),
                    responses=list(responses),
                    description=description,
                )
                self.write_json(res)
            else:
                metrics = bern_ai.retrain()
                self.write_json({"status": "retrained", "metrics": metrics})
        except ValueError as exc:
            self.write_json({"error": str(exc)}, status_code=400)


class IntentDetailHandler(BaseHandler):
    def delete(self, tag: str) -> None:
        try:
            res = bern_ai.delete_intent(tag)
            self.write_json(res)
        except ValueError as exc:
            self.write_json({"error": str(exc)}, status_code=400)


class PHPricesHandler(BaseHandler):
    def get(self) -> None:
        q = self.get_argument("q", default="")
        category = self.get_argument("category", default="")
        self.write_json(bern_ai.price_expert.get_catalog(query=q, category=category))


class PHSkuCatalogHandler(BaseHandler):
    def get(self) -> None:
        q = self.get_argument("q", default="")
        category = self.get_argument("category", default="")
        limit = min(50, max(1, int(self.get_argument("limit", default="20"))))
        items = bern_ai.price_expert.search_skus(query=q, category=category, limit=limit)
        self.write_json({"count": len(items), "total_skus": len(bern_ai.price_expert.skus), "items": items})


class DatasetsHandler(BaseHandler):
    def get(self) -> None:
        summary = bern_ai.price_expert.get_dataset_summary()
        summary["money_vision_trained_real_images"] = money_counter.real_train_images
        summary["nutrition"] = bern_ai.nutrition_expert.summary()
        self.write_json(summary)


class NutritionSearchHandler(BaseHandler):
    """GET /api/nutrition?q=garlic&source=philfct&type=generic&category=Vegetables&limit=10"""

    def get(self) -> None:
        q = self.get_argument("q", default="").strip()
        expert = bern_ai.nutrition_expert
        if not q:
            self.write_json({"query": q, "count": 0, "items": [], "summary": expert.summary()})
            return
        try:
            limit = min(50, max(1, int(self.get_argument("limit", default="10"))))
        except ValueError:
            self.write_json({"error": "limit must be an integer"}, status_code=400)
            return
        items = expert.search(
            q, limit=limit,
            source=self.get_argument("source", default="") or None,
            kind=self.get_argument("type", default="") or None,
            category=self.get_argument("category", default="") or None,
        )
        self.write_json({"query": q, "count": len(items), "items": items})


class NutritionItemHandler(BaseHandler):
    """GET /api/nutrition/item?id=philfct:D084[&grams=200]"""

    def get(self) -> None:
        food_id = self.get_argument("id", default="")
        expert = bern_ai.nutrition_expert
        rec = expert.get(food_id)
        if not rec:
            self.write_json({"error": f"Unknown food id: {food_id}"}, status_code=404)
            return
        grams = self.get_argument("grams", default="")
        if grams:
            try:
                g = float(grams)
            except ValueError:
                self.write_json({"error": "grams must be a number"}, status_code=400)
                return
            nutrients, label, used = expert.portion(expert.by_id[food_id], grams=g)
            rec["portion"] = {"label": label, "grams": used, "nutrients": nutrients}
        self.write_json(rec)


class NutritionMealHandler(BaseHandler):
    """POST /api/nutrition/meal  {"text": "2 pc chickenjoy and 1 cup rice"}  or  {"items": [{"name": "adobo", "qty": 1}, ...]}"""

    def post(self) -> None:
        try:
            body = self.parse_json_body()
        except ValueError as exc:
            self.write_json({"error": str(exc)}, status_code=400)
            return
        expert = bern_ai.nutrition_expert
        items = body.get("items")
        if not items and body.get("text"):
            items = [{"name": p["name"], "qty": p["qty"], "unit": p["unit"], "grams": p["grams"]}
                     for p in expert.parse_meal_text(str(body["text"]))]
        if not isinstance(items, list) or not items:
            self.write_json({"error": "Provide `text` (e.g. '2 pc chickenjoy and 1 cup rice') or an `items` array."}, status_code=400)
            return
        self.write_json(expert.estimate_meal(items))


class NutritionSummaryHandler(BaseHandler):
    def get(self) -> None:
        self.write_json(bern_ai.nutrition_expert.summary())


class PHBudgetHandler(BaseHandler):
    def post(self) -> None:
        try:
            body = self.parse_json_body()
            amount = float(body.get("budget_php", 0))
            self.write_json(bern_ai.price_expert.suggest_basket_for_budget(amount))
        except (ValueError, TypeError) as exc:
            self.write_json({"error": f"Invalid budget_php value: {exc}"}, status_code=400)


class CountMoneyHandler(BaseHandler):
    def post(self) -> None:
        try:
            image_bytes: bytes | None = None

            # Option A: multipart/form-data file upload ("image")
            if "image" in self.request.files and self.request.files["image"]:
                image_bytes = self.request.files["image"][0]["body"]
            else:
                # Option B: JSON {"image_base64": "data:image/jpeg;base64,..."}
                body = self.parse_json_body()
                b64_raw = str(body.get("image_base64", "")).strip()
                if b64_raw:
                    if "," in b64_raw and b64_raw.startswith("data:"):
                        b64_raw = b64_raw.split(",", 1)[1]
                    image_bytes = base64.b64decode(b64_raw)

            if not image_bytes:
                self.write_json(
                    {"error": "Provide a photo via multipart 'image' field or JSON 'image_base64'."},
                    status_code=400,
                )
                return

            result = money_counter.count_from_bytes(image_bytes)
            if result["total_php"] > 0:
                result["buying_power"] = bern_ai.price_expert.suggest_basket_for_budget(result["total_php"])
            self.write_json(result)
        except Exception as exc:
            self.write_json({"error": str(exc)}, status_code=400)


class HandwritingHandler(BaseHandler):
    """Transcribe handwriting from an uploaded photo.

    Accepts the same two input shapes as ``CountMoneyHandler`` so clients only need one
    upload helper, and returns the reader's result plus the active AI name.
    """

    def post(self) -> None:
        try:
            image_bytes: bytes | None = None

            # Option A: multipart/form-data file upload ("image")
            if "image" in self.request.files and self.request.files["image"]:
                image_bytes = self.request.files["image"][0]["body"]
            else:
                # Option B: JSON {"image_base64": "data:image/jpeg;base64,..."}
                body = self.parse_json_body()
                b64_raw = str(body.get("image_base64", "")).strip()
                if b64_raw:
                    if "," in b64_raw and b64_raw.startswith("data:"):
                        b64_raw = b64_raw.split(",", 1)[1]
                    image_bytes = base64.b64decode(b64_raw)

            if not image_bytes:
                self.write_json(
                    {"error": "Provide a photo via multipart 'image' field or JSON 'image_base64'."},
                    status_code=400,
                )
                return

            body = {}
            backend = "auto"
            case_mode = "auto"
            try:
                body = self.parse_json_body()
            except Exception:
                body = {}
            if isinstance(body, dict):
                backend = str(body.get("backend", "auto"))
                case_mode = str(body.get("case_mode", "auto"))

            self.write_json(bern_ai.read_handwriting(image_bytes, backend=backend, case_mode=case_mode))
        except Exception as exc:
            self.write_json({"error": str(exc)}, status_code=400)


class ExtractHandler(BaseHandler):
    """Turn an uploaded document into plain text.

    Accepts multipart field ``file`` (PDF, DOCX, image, or text) or JSON ``text`` which is
    passed through. Useful on its own, and it is what powers the study endpoints.
    """

    def post(self) -> None:
        try:
            data, filename = self._read_upload()
            if data is None:
                body = self.parse_json_body()
                pasted = str(body.get("text", ""))
                if not pasted.strip():
                    self.write_json(
                        {"error": "Provide a file via multipart 'file' or JSON 'text'."},
                        status_code=400,
                    )
                    return
                self.write_json({"text": pasted, "source_type": "text", "engine": "pasted",
                                 "characters": len(pasted), "lines": pasted.count("\n") + 1,
                                 "words": len(pasted.split())})
                return

            from bern.text_extract import ExtractionError, extract_text

            try:
                self.write_json(extract_text(data, filename))
            except ExtractionError as exc:
                self.write_json({"error": str(exc)}, status_code=400)
        except Exception as exc:
            self.write_json({"error": str(exc)}, status_code=400)

    def _read_upload(self) -> tuple[bytes | None, str]:
        """Return the uploaded file bytes and name, or ``(None, "")``."""
        if "file" in self.request.files and self.request.files["file"]:
            upload = self.request.files["file"][0]
            return upload["body"], str(upload.get("filename", ""))
        return None, ""


class StudyMaterialsHandler(BaseHandler):
    """Build reviewers, quizzes and flashcards from notes, a lecture file, or pasted text.

    The heavy lifting is in :mod:`bern.study`; this handler only chooses the source.
    """

    def post(self) -> None:
        try:
            from bern.study import generate_study_pack
            from bern.text_extract import ExtractionError, extract_text

            body: dict[str, Any] = {}
            try:
                body = self.parse_json_body()
            except Exception:
                body = {}
            if not isinstance(body, dict):
                body = {}

            source_text = str(body.get("text", ""))
            source_type = "text"

            if "file" in self.request.files and self.request.files["file"]:
                upload = self.request.files["file"][0]
                try:
                    extracted = extract_text(upload["body"], str(upload.get("filename", "")))
                except ExtractionError as exc:
                    self.write_json({"error": str(exc)}, status_code=400)
                    return
                source_text = extracted["text"]
                source_type = extracted["source_type"]
            elif not source_text.strip():
                self.write_json(
                    {"error": "Provide a file via multipart 'file' or JSON 'text'."},
                    status_code=400,
                )
                return

            if not source_text.strip():
                self.write_json(
                    {"error": "No text could be extracted. For a scanned PDF, export the "
                              "pages as images and send them through /api/handwriting."},
                    status_code=400,
                )
                return

            # "auto" is the safe default: it uses a language model only when one is
            # configured and falls back to the local generator otherwise, so an install with
            # no key behaves exactly as before.
            engine = str(body.get("engine", "auto")).lower()
            if engine not in ("local", "llm", "auto"):
                self.write_json(
                    {"error": "engine must be 'local', 'llm' or 'auto'."}, status_code=400
                )
                return

            pack = generate_study_pack(
                source_text,
                flashcards=int(body.get("flashcards", 20)),
                quiz_items=int(body.get("quiz_items", 10)),
                true_false_items=int(body.get("true_false_items", 8)),
                seed=int(body.get("seed", 7)),
                engine=engine,
            )
            pack["source_stats"]["source_type"] = source_type
            pack["ai_name"] = bern_ai.bundle.get("ai_name", "Bern")
            self.write_json(pack)
        except Exception as exc:
            self.write_json({"error": str(exc)}, status_code=400)


class TutorHandler(BaseHandler):
    """Answer follow-up questions about a set of notes, remembering the conversation.

    Requires a configured language model (``BERN_LLM_API_KEY``). Without one this returns 503
    with an explanation instead of pretending to tutor: a scripted reply that ignores the
    question is worse than an honest refusal, and the local study pack is still available at
    ``/api/study``.

    Sessions are held in memory and keyed by ``session_id``. They are lost on restart, which
    is acceptable for a single-user local tool and avoids a storage dependency.
    """

    _sessions: dict[str, Any] = {}
    _max_sessions = 64

    def post(self) -> None:
        try:
            from bern import llm

            body = self.parse_json_body()
            if not isinstance(body, dict):
                body = {}

            if not llm.is_available():
                self.write_json(
                    {"error": "Tutoring needs a language model. Set BERN_LLM_API_KEY and "
                              "restart. Everything else in Bern works without it.",
                     "engine": "none"},
                    status_code=503,
                )
                return

            question = str(body.get("question", "")).strip()
            if not question:
                self.write_json({"error": "Provide a 'question'."}, status_code=400)
                return

            session_id = str(body.get("session_id", "default"))
            notes = str(body.get("notes", ""))

            session = self._sessions.get(session_id)
            if session is None or (notes and notes != session.notes):
                # New session, or the caller swapped the notes under an existing one.
                if len(self._sessions) >= self._max_sessions:
                    oldest = next(iter(self._sessions))
                    self._sessions.pop(oldest, None)
                session = llm.TutorSession(notes=notes)
                self._sessions[session_id] = session

            result = session.ask(question)
            self.write_json({
                "answer": result["answer"],
                "engine": result["engine"],
                "grounded": result["grounded"],
                "session_id": session_id,
                "turns": len(session.history) // 2,
            })
        except Exception as exc:
            self.write_json({"error": str(exc)}, status_code=400)

    def delete(self) -> None:
        """Forget a session. ``DELETE /api/tutor?session_id=...``"""
        session_id = self.get_query_argument("session_id", "default")
        self._sessions.pop(session_id, None)
        self.write_json({"cleared": session_id})


class SampleHandwritingHandler(BaseHandler):
    """Serve the committed sample handwriting image for the web console's demo button.

    Returns a plain 404 with an explanation when the file is absent, rather than raising:
    the sample is a convenience, and its absence must not look like a server fault.
    """

    def get(self) -> None:
        path = BASE_DIR / "examples" / "sample_handwriting.png"
        if not path.exists():
            self.write_json({"error": "No sample image in examples/sample_handwriting.png."},
                            status_code=404)
            return
        self.set_header("Content-Type", "image/png")
        self.set_header("Cache-Control", "no-store")
        self.finish(path.read_bytes())


class SampleMoneyPhotoHandler(BaseHandler):
    def get(self) -> None:
        jpg_bytes = generate_sample_ph_money_image()
        self.set_header("Content-Type", "image/jpeg")
        self.set_header("Cache-Control", "no-store")
        self.finish(jpg_bytes)


class ReactNativeFilesHandler(BaseHandler):
    """Serve the bundled React Native client files as JSON.

    Degrades to a 404 with an explanation instead of a stack trace when the directory has
    not been checked out — the handler used to read the files unconditionally.
    """

    def get(self) -> None:
        api_ts = RN_DIR / "bernApi.ts"
        if not api_ts.exists():
            self.write_json(
                {"error": f"React Native client files not found in {RN_DIR}. "
                          "They ship with the repository — check out the full tree."},
                status_code=404,
            )
            return
        api_source = api_ts.read_text(encoding="utf-8")
        files = {"bernApi.ts": api_source}
        for name in ("BernChatScreen.tsx", "BernMoneyScannerScreen.tsx", "BernNutritionScreen.tsx"):
            candidate = RN_DIR / name
            if candidate.exists():
                files[name] = candidate.read_text(encoding="utf-8")
        self.write_json({"files": files})


def make_app() -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/", HomeHandler),
            (r"/index\.html", HomeHandler),
            (r"/api/health", HealthHandler),
            (r"/api/chat", ChatHandler),
            (r"/api/predict", PredictHandler),
            (r"/api/metrics", MetricsHandler),
            (r"/api/dataset", DatasetHandler),
            (r"/api/training-report", TrainingReportHandler),
            (r"/api/train", TrainHandler),
            (r"/api/learn", LearnHandler),
            (r"/api/learned", LearnedHandler),
            (r"/api/intents/([^/]+)", IntentDetailHandler),
            (r"/api/ph-prices", PHPricesHandler),
            (r"/api/ph-prices/skus", PHSkuCatalogHandler),
            (r"/api/ph-prices/budget", PHBudgetHandler),
            (r"/api/nutrition", NutritionSearchHandler),
            (r"/api/nutrition/item", NutritionItemHandler),
            (r"/api/nutrition/meal", NutritionMealHandler),
            (r"/api/nutrition/summary", NutritionSummaryHandler),
            (r"/api/datasets", DatasetsHandler),
            (r"/api/count-money", CountMoneyHandler),
            (r"/api/handwriting", HandwritingHandler),
            (r"/api/extract", ExtractHandler),
            (r"/api/study", StudyMaterialsHandler),
            (r"/api/tutor", TutorHandler),
            (r"/examples/sample_handwriting\.png", SampleHandwritingHandler),
            (r"/api/sample-money-photo", SampleMoneyPhotoHandler),
            (r"/api/react-native-files", ReactNativeFilesHandler),
        ]
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app = make_app()
    app.listen(port, address="0.0.0.0")
    print(f"Bern AI Server listening on http://0.0.0.0:{port}", flush=True)
    tornado.ioloop.IOLoop.current().start()
