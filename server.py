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
            self.write_json({"error": "Web UI not found"}, status_code=404)
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
            }
        )


class ChatHandler(BaseHandler):
    def post(self) -> None:
        try:
            body = self.parse_json_body()
            message = str(body.get("message", ""))
            session_id = str(body.get("session_id", "default"))
            result = bern_ai.chat(message=message, session_id=session_id)
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


class SampleMoneyPhotoHandler(BaseHandler):
    def get(self) -> None:
        jpg_bytes = generate_sample_ph_money_image()
        self.set_header("Content-Type", "image/jpeg")
        self.set_header("Cache-Control", "no-store")
        self.finish(jpg_bytes)


class ReactNativeFilesHandler(BaseHandler):
    def get(self) -> None:
        api_ts = (RN_DIR / "bernApi.ts").read_text(encoding="utf-8")
        chat_tsx = (RN_DIR / "BernChatScreen.tsx").read_text(encoding="utf-8")
        scanner_tsx = (RN_DIR / "BernMoneyScannerScreen.tsx").read_text(encoding="utf-8")
        files = {
            "bernApi.ts": api_ts,
            "BernChatScreen.tsx": chat_tsx,
            "BernMoneyScannerScreen.tsx": scanner_tsx,
        }
        nutrition_path = RN_DIR / "BernNutritionScreen.tsx"
        if nutrition_path.exists():
            files["BernNutritionScreen.tsx"] = nutrition_path.read_text(encoding="utf-8")
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
