from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PH_PRICES_PATH = BASE_DIR / "data" / "ph_prices.json"
DEFAULT_REGIONAL_PATH = BASE_DIR / "data" / "ph_regional_prices.json"
DEFAULT_SKU_CATALOG_PATH = BASE_DIR / "data" / "ph_sku_catalog.json"

REGION_ALIASES: dict[str, tuple[str, str]] = {
    "ncr": ("National Capital region", "NCR / Metro Manila"),
    "metro manila": ("National Capital region", "NCR / Metro Manila"),
    "manila": ("National Capital region", "NCR / Metro Manila"),
    "quezon city": ("National Capital region", "NCR / Metro Manila"),
    "makati": ("National Capital region", "NCR / Metro Manila"),
    "calabarzon": ("Region IV-A", "Region IV-A (Calabarzon)"),
    "laguna": ("Region IV-A", "Region IV-A (Calabarzon)"),
    "cavite": ("Region IV-A", "Region IV-A (Calabarzon)"),
    "batangas": ("Region IV-A", "Region IV-A (Calabarzon)"),
    "rizal": ("Region IV-A", "Region IV-A (Calabarzon)"),
    "quezon": ("Region IV-A", "Region IV-A (Calabarzon)"),
    "central luzon": ("Region III", "Region III (Central Luzon)"),
    "pampanga": ("Region III", "Region III (Central Luzon)"),
    "bulacan": ("Region III", "Region III (Central Luzon)"),
    "nueva ecija": ("Region III", "Region III (Central Luzon)"),
    "ilocos": ("Region I", "Region I (Ilocos)"),
    "pangasinan": ("Region I", "Region I (Ilocos)"),
    "cagayan": ("Region II", "Region II (Cagayan Valley)"),
    "cordillera": ("Cordillera Administrative region", "CAR (Cordillera)"),
    "baguio": ("Cordillera Administrative region", "CAR (Cordillera)"),
    "benguet": ("Cordillera Administrative region", "CAR (Cordillera)"),
    "mimaropa": ("Region IV-B", "Region IV-B (MIMAROPA)"),
    "palawan": ("Region IV-B", "Region IV-B (MIMAROPA)"),
    "bicol": ("Region V", "Region V (Bicol)"),
    "albay": ("Region V", "Region V (Bicol)"),
    "western visayas": ("Region VI", "Region VI (Western Visayas)"),
    "iloilo": ("Region VI", "Region VI (Western Visayas)"),
    "bacolod": ("Region VI", "Region VI (Western Visayas)"),
    "central visayas": ("Region VII", "Region VII (Central Visayas)"),
    "cebu": ("Region VII", "Region VII (Central Visayas)"),
    "bohol": ("Region VII", "Region VII (Central Visayas)"),
    "eastern visayas": ("Region VIII", "Region VIII (Eastern Visayas)"),
    "tacloban": ("Region VIII", "Region VIII (Eastern Visayas)"),
    "zamboanga": ("Region IX", "Region IX (Zamboanga Peninsula)"),
    "northern mindanao": ("Region X", "Region X (Northern Mindanao)"),
    "cagayan de oro": ("Region X", "Region X (Northern Mindanao)"),
    "davao": ("Region XI", "Region XI (Davao Region)"),
    "soccsksargen": ("Region XII", "Region XII (SOCCSKSARGEN)"),
    "caraga": ("Region XIII", "Region XIII (Caraga)"),
    "barmm": ("Autonomous region in Muslim Mindanao", "BARMM"),
    "mindanao": ("Region XI", "Region XI (Davao Region)"),
    "visayas": ("Region VII", "Region VII (Central Visayas)"),
}

WFP_COMMODITY_ALIASES: list[tuple[list[str], str]] = [
    (["well milled", "well-milled", "sinandomeng"], "Rice (well milled)"),
    (["special rice", "dinorado", "jasmine"], "Rice (special)"),
    (["bigas", "rice", "regular rice"], "Rice (regular, milled)"),
    (["pork", "baboy", "liempo", "kasim"], "Meat (pork)"),
    (["chicken", "manok"], "Meat (chicken, whole)"),
    (["beef", "baka"], "Meat (beef)"),
    (["egg", "eggs", "itlog"], "Eggs"),
    (["bangus", "milkfish"], "Fish (milkfish)"),
    (["tilapia"], "Fish (tilapia)"),
    (["galunggong", "roundscad", "gg"], "Fish (roundscad)"),
    (["onion", "onions", "sibuyas"], "Onions (red)"),
    (["garlic", "bawang"], "Garlic"),
    (["tomato", "tomatoes", "kamatis"], "Tomatoes"),
    (["cabbage", "repolyo"], "Cabbage"),
    (["eggplant", "talong"], "Eggplants"),
    (["ampalaya", "bitter"], "Bitter melon"),
    (["calamansi", "kalamansi"], "Calamansi"),
    (["banana", "saging", "lakatan"], "Bananas (lakatan)"),
    (["ginger", "luya"], "Ginger"),
    (["carrot", "carrots"], "Carrots"),
    (["potato", "patatas"], "Potatoes (Irish)"),
]


_TAGALOG_CATEGORY_WORDS = {
    "gulay": "vegetables", "prutas": "fruits", "karne": "meat", "isda": "seafood", "gatas": "dairy", "itlog": "eggs",
    "tinapay": "bread", "bigas": "rice", "mantika": "cooking", "pampalasa": "condiments",
}


def fmt_php(val: float) -> str:
    if abs(val - round(val)) < 1e-6:
        return f"₱{int(round(val)):,}"
    return f"₱{val:,.2f}"


class PHPriceExpert:
    """Philippine Common Goods Price Knowledge Base, SKU Catalog & Basket Estimator."""

    def __init__(
        self,
        prices_path: Path = DEFAULT_PH_PRICES_PATH,
        regional_path: Path = DEFAULT_REGIONAL_PATH,
        sku_path: Path = DEFAULT_SKU_CATALOG_PATH,
    ) -> None:
        self.prices_path = Path(prices_path)
        self.regional_path = Path(regional_path)
        self.sku_path = Path(sku_path)
        self.data: dict[str, Any] = {}
        self.items: list[dict[str, Any]] = []
        self.regional: dict[str, Any] = {}
        self.skus: list[dict[str, Any]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.item_matrix: Any = None
        self.sku_vectorizer: TfidfVectorizer | None = None
        self.sku_matrix: Any = None
        self.reload()

    def reload(self) -> None:
        with open(self.prices_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.items = self.data.get("items", [])
        corpus = [
            " ".join([item["name"], item["category"], " ".join(item.get("aliases", [])), item.get("notes", "")])
            for item in self.items
        ]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
        self.item_matrix = self.vectorizer.fit_transform(corpus) if corpus else None

        if self.regional_path.exists():
            self.regional = json.loads(self.regional_path.read_text(encoding="utf-8"))
        else:
            self.regional = {}

        if self.sku_path.exists():
            sku_blob = json.loads(self.sku_path.read_text(encoding="utf-8"))
            self.skus = sku_blob.get("skus", [])
        else:
            self.skus = []

        if self.skus:
            sku_corpus = [
                " ".join([s.get("name", ""), s.get("brand", ""), s.get("category", "")])
                for s in self.skus
            ]
            self.sku_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
            self.sku_matrix = self.sku_vectorizer.fit_transform(sku_corpus)
        else:
            self.sku_vectorizer = None
            self.sku_matrix = None

    def get_catalog(self, query: str = "", category: str = "") -> dict[str, Any]:
        items = self.items
        if category:
            cat_l = category.strip().lower()
            items = [i for i in items if cat_l in i["category"].lower()]
        if query:
            matches = self.search_items(query, limit=16, min_score=0.16)
            items = [m["item"] for m in matches]
        return {
            "currency": self.data.get("currency", "PHP"),
            "symbol": self.data.get("symbol", "₱"),
            "last_updated": self.data.get("last_updated"),
            "sources": self.data.get("sources", []),
            "count": len(items),
            "items": items,
            "categories": sorted({i["category"] for i in self.items}),
            "sku_count": len(self.skus),
            "canned_sku_count": sum(1 for s in self.skus if "canned" in s.get("category", "")),
        }

    def search_skus(self, query: str = "", category: str = "", limit: int = 12, min_score: float = 0.20) -> list[dict[str, Any]]:
        if not self.skus:
            return []
        q = query.strip().lower()
        cat_l = category.strip().lower()

        if not q:
            pool = [s for s in self.skus if (not cat_l or cat_l in s.get("category", "").lower())]
            # Prioritize SKUs that include price benchmarks
            pool.sort(key=lambda s: (0 if s.get("srp_php") is not None else 1, s.get("name", "")))
            return pool[:limit]

        if self.sku_vectorizer is None or self.sku_matrix is None:
            return []

        q_vec = self.sku_vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.sku_matrix).ravel()
        q_tokens = [t for t in re.findall(r"\b\w+\b", q) if len(t) >= 3]

        scored: list[tuple[float, dict[str, Any]]] = []
        for idx, sku in enumerate(self.skus):
            if cat_l and cat_l not in sku.get("category", "").lower():
                continue
            score = float(sims[idx])
            text_l = f"{sku.get('name', '')} {sku.get('brand', '')}".lower()
            for tok in q_tokens:
                if re.search(rf"\b{re.escape(tok)}\b", text_l):
                    score += 0.18
            if sku.get("srp_php") is not None:
                score += 0.06
            if score >= min_score:
                scored.append((score, sku))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [{**sku, "score": round(sc, 4)} for sc, sku in scored[:limit]]

    def get_dataset_summary(self) -> dict[str, Any]:
        wfp = self.regional.get("wfp", {})
        da = self.regional.get("da_latest", {})
        ext_dir = BASE_DIR / "data" / "external"
        vision_meta = ext_dir / "ph_banknotes_coins" / "metadata.csv"
        vision_count = 0
        if vision_meta.exists():
            vision_count = max(0, sum(1 for _ in vision_meta.open("r", encoding="utf-8")) - 1)
        return {
            "curated_items": len(self.items),
            "categories": sorted({i["category"] for i in self.items}),
            "sku_catalog_total": len(self.skus),
            "canned_goods_skus": sum(1 for s in self.skus if "canned" in s.get("category", "")),
            "wfp_total_rows": wfp.get("total_rows", 0),
            "wfp_date_range": wfp.get("date_range", []),
            "wfp_commodities_2026": wfp.get("commodity_count", 0),
            "da_total_rows": da.get("total_rows", 0),
            "da_latest_date": da.get("latest_date"),
            "da_latest_items": len(da.get("items", [])),
            "money_vision_real_images": vision_count,
            "sources": self.data.get("sources", []),
        }

    def search_items(self, query: str, limit: int = 5, min_score: float = 0.22) -> list[dict[str, Any]]:
        q = query.strip().lower()
        if not q or not self.items or self.vectorizer is None:
            return []

        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.item_matrix).ravel()

        results: list[dict[str, Any]] = []
        q_tokens = set(re.findall(r"\b\w+\b", q))

        for idx, item in enumerate(self.items):
            score = float(sims[idx])
            aliases = [a.lower() for a in item.get("aliases", [])] + [item["name"].lower()]
            alias_hit = False
            best_alias_len = 0
            best_alias_words = 0
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", q):
                    alias_hit = True
                    if len(alias) > best_alias_len:
                        best_alias_len = len(alias)
                        best_alias_words = len(alias.split())
                elif any(tok in alias.split() for tok in q_tokens if len(tok) >= 4):
                    score += 0.10

            if alias_hit:
                score = max(score, 0.76 + min(0.22, best_alias_len * 0.012) + 0.04 * best_alias_words)

            if score >= min_score:
                results.append({"item": item, "score": round(score, 4)})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    def format_item_summary(self, item: dict[str, Any], qty: float = 1.0) -> str:
        p_min = item["palengke_min"] * qty
        p_max = item["palengke_max"] * qty
        s_min = item["supermarket_min"] * qty
        s_max = item["supermarket_max"] * qty
        qty_prefix = f"{int(qty) if qty == int(qty) else qty} × " if qty != 1.0 else ""

        palengke_label = "Sari-sari / Palengke" if item["category"] in ("Canned & Instant", "Street Foods & Snacks") else "Palengke / Market"
        lines = [
            f"**{qty_prefix}{item['name']}** ({item['unit']}):",
            f"  • {palengke_label}: **{fmt_php(p_min)}–{fmt_php(p_max)}**",
            f"  • Supermarket: **{fmt_php(s_min)}–{fmt_php(s_max)}**",
        ]
        if item.get("notes"):
            lines.append(f"  • _Tip:_ {item['notes']}")
        return "\n".join(lines)

    def lookup_regional_price(self, query: str) -> str | None:
        wfp_comms = self.regional.get("wfp", {}).get("commodities", {})
        if not wfp_comms:
            return None
        ql = query.lower()

        matched_region: tuple[str, str] | None = None
        for alias, target in REGION_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", ql):
                matched_region = target
                break

        if not matched_region and not re.search(r"\b(?:region|regional|province|wfp)\b", ql):
            return None

        matched_comm: str | None = None
        for keywords, comm_name in WFP_COMMODITY_ALIASES:
            if any(re.search(rf"\b{re.escape(kw)}\b", ql) for kw in keywords):
                if comm_name in wfp_comms:
                    matched_comm = comm_name
                    break

        if not matched_comm:
            # Default to Rice if region asked with general food/prices
            if matched_region and re.search(r"\b(?:rice|bigas|food|prices?|market)\b", ql):
                matched_comm = "Rice (regular, milled)"
            else:
                return None

        info = wfp_comms[matched_comm]
        snap_date = self.regional.get("wfp", {}).get("snapshot_date", "2026-09-15")
        unit = info["unit"]
        if matched_region:
            reg_key, reg_label = matched_region
            reg_stats = info.get("regions", {}).get(reg_key)
            if reg_stats:
                return (
                    f"**2026 Regional Price — {matched_comm}** in **{reg_label}** _(WFP/PSA data, {snap_date})_:\n"
                    f"  • Regional Median: **{fmt_php(reg_stats['median'])}/{unit}** "
                    f"(market range: {fmt_php(reg_stats['min'])}–{fmt_php(reg_stats['max'])}, across {reg_stats['markets']} markets)\n"
                    f"  • National Median: **{fmt_php(info['national_median'])}/{unit}**"
                )

        # Compare major regions
        sample_regions = [
            ("National Capital region", "NCR"),
            ("Region III", "Central Luzon"),
            ("Region IV-A", "Calabarzon"),
            ("Region VII", "Central Visayas"),
            ("Region XI", "Davao Region"),
        ]
        lines = [f"**2026 Philippine Regional Prices — {matched_comm}** _(per {unit}, WFP/PSA {snap_date})_:"]
        for rkey, rlabel in sample_regions:
            rstats = info.get("regions", {}).get(rkey)
            if rstats:
                lines.append(
                    f"  • **{rlabel}**: median **{fmt_php(rstats['median'])}** ({fmt_php(rstats['min'])}–{fmt_php(rstats['max'])})"
                )
        lines.append(f"  • **National Median**: **{fmt_php(info['national_median'])}/{unit}**")
        return "\n".join(lines)

    _PRICE_FILLER = re.compile(
        r"\b(price|prices|presyo|magkano|how much|cost|per|bundle|pack|the|of|ang|ng|po|is|ba|ngayon|today|current|latest|what|whats|"
        r"philippine|philippines|pilipinas|pinoy|list|listahan|please|pls|show|me|all|common|grocery|groceries|goods|ipakita|gusto|kong|makita|mga|sa)\b",
        re.IGNORECASE)

    def search_skus_for_message(self, message: str, limit: int = 5, min_score: float = 0.45) -> list[dict[str, Any]]:
        """SKU search that ignores price-question filler and generic words ("tuyo price per bundle" -> "tuyo";
        "philippine price list please" -> nothing, so it can't match a vegetable called Philippine Spinach)."""
        core = re.sub(r"\s+", " ", self._PRICE_FILLER.sub(" ", message)).strip()
        return self.search_skus(core, limit=limit, min_score=min_score) if core else []

    def extract_multi_item_basket(self, message: str) -> list[dict[str, Any]]:
        segments = re.split(r",|\band\b|\bat\b|\bsaka\b|\bplus\b|\+|&", message, flags=re.IGNORECASE)
        found: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for seg in segments:
            seg_clean = seg.strip()
            if not seg_clean:
                continue
            qty_match = re.search(
                r"(\d+(?:\.\d+)?)\s*(?:kg|kilo|kilos|l|liter|liters|pc|pcs|pack|packs|can|cans|tray|trays|x)?",
                seg_clean,
                re.IGNORECASE,
            )
            qty = 1.0
            if qty_match:
                num = float(qty_match.group(1))
                if not re.search(rf"{re.escape(qty_match.group(1))}\s*(?:g|ml|in-1|kg\s+lpg)\b", seg_clean, re.IGNORECASE):
                    if 0 < num <= 100 and num != 91:
                        qty = num

            matches = self.search_items(seg_clean, limit=1, min_score=0.45)
            if matches:
                item = matches[0]["item"]
                if item["id"] not in seen_ids:
                    seen_ids.add(item["id"])
                    found.append({"item": item, "qty": qty, "score": matches[0]["score"]})
        return found

    def estimate_basket(self, items_with_qty: list[dict[str, Any]]) -> dict[str, Any]:
        total_palengke_min = 0.0
        total_palengke_max = 0.0
        total_super_min = 0.0
        total_super_max = 0.0
        total_typical = 0.0
        lines: list[dict[str, Any]] = []

        for entry in items_with_qty:
            item = entry["item"]
            qty = float(entry.get("qty", 1.0))
            total_palengke_min += item["palengke_min"] * qty
            total_palengke_max += item["palengke_max"] * qty
            total_super_min += item["supermarket_min"] * qty
            total_super_max += item["supermarket_max"] * qty
            total_typical += item["typical"] * qty
            lines.append(
                {
                    "id": item["id"],
                    "name": item["name"],
                    "qty": qty,
                    "unit": item["unit"],
                    "palengke_range": [round(item["palengke_min"] * qty, 2), round(item["palengke_max"] * qty, 2)],
                    "supermarket_range": [round(item["supermarket_min"] * qty, 2), round(item["supermarket_max"] * qty, 2)],
                    "typical_subtotal": round(item["typical"] * qty, 2),
                }
            )

        return {
            "lines": lines,
            "palengke_total_range": [round(total_palengke_min, 2), round(total_palengke_max, 2)],
            "supermarket_total_range": [round(total_super_min, 2), round(total_super_max, 2)],
            "typical_total": round(total_typical, 2),
        }

    def suggest_basket_for_budget(self, php_amount: float) -> dict[str, Any]:
        if php_amount <= 0:
            return {"budget": php_amount, "items": [], "estimated_cost": 0.0}

        priority_order = [
            ("rice_well_milled", 2 if php_amount >= 300 else 1),
            ("eggs_piece", 6 if php_amount >= 150 else 2),
            ("cooking_oil", 1),
            ("whole_chicken", 1),
            ("sardines", 2),
            ("pancit_canton", 3),
            ("red_onion", 0.5),
            ("garlic", 0.25),
            ("soy_sauce", 1),
            ("pork_kasim", 1),
            ("rice_5kg", 1),
            ("eggs_tray", 1),
        ]

        id_map = {item["id"]: item for item in self.items}
        remaining = php_amount
        chosen: list[dict[str, Any]] = []
        spent = 0.0
        skip_ids = {"rice_well_milled", "eggs_piece"} if php_amount >= 900 else {"rice_5kg", "eggs_tray"}

        for item_id, qty in priority_order:
            if item_id in skip_ids or item_id not in id_map:
                continue
            item = id_map[item_id]
            cost = item["typical"] * qty
            if cost <= remaining:
                chosen.append({"item": item, "qty": qty, "subtotal": round(cost, 2)})
                remaining -= cost
                spent += cost

        return {
            "budget": round(php_amount, 2),
            "items": chosen,
            "estimated_cost": round(spent, 2),
            "estimated_change": round(php_amount - spent, 2),
        }

    def answer_price_query(self, message: str) -> str:
        q = message.strip()
        ql = q.lower()

        # 1. Check budget query: "what can 500 pesos buy" / "kasya ba 1000"
        budget_match = re.search(r"(?:₱|php|pesos?|p)\s*([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*(?:pesos?|php)", ql)
        if budget_match and any(w in ql for w in ["buy", "kasya", "budget", "mabibili", "worth", "can i get"]):
            raw_num = (budget_match.group(1) or budget_match.group(2)).replace(",", "")
            amount = float(raw_num)
            plan = self.suggest_basket_for_budget(amount)
            if not plan["items"]:
                return f"With **{fmt_php(amount)}**, you can grab a couple of small essentials like **pandesal** ({fmt_php(3)}–{fmt_php(5)}/pc), **3-in-1 coffee** ({fmt_php(7.75)}–{fmt_php(12)}), or **instant mami** ({fmt_php(9)}–{fmt_php(14)})."
            lines = [f"Here is a sample Philippine grocery basket for **{fmt_php(amount)}** (2026 typical prices):"]
            for entry in plan["items"]:
                it = entry["item"]
                q_str = int(entry["qty"]) if entry["qty"] == int(entry["qty"]) else entry["qty"]
                lines.append(f"- **{q_str} × {it['name']}** ({it['unit']}) ≈ **{fmt_php(entry['subtotal'])}**")
            lines.append(f"\nEstimated total: **{fmt_php(plan['estimated_cost'])}** (estimated change: **{fmt_php(plan['estimated_change'])}**).")
            return "\n".join(lines)

        # 2. Check regional WFP price lookup (e.g., "magkano bigas sa Calabarzon", "pork price in Cebu")
        regional_ans = self.lookup_regional_price(q)
        if regional_ans:
            return regional_ans

        # 3. Canned foods / delata dedicated summary
        if any(re.search(rf"\b{re.escape(kw)}\b", ql) for kw in ["canned food", "canned foods", "canned goods", "delata", "de lata"]):
            canned_items = [
                it for it in self.items
                if it["category"] == "Canned & Instant"
                and any(w in it["name"].lower() for w in ["sardine", "tuna", "corned", "loaf", "luncheon", "beans", "vienna", "liver", "tomato", "fruit"])
            ]
            lines = ["**2026 Philippine Canned Foods Price Guide** _(DTI May 2026 SRP & Sari-Sari/Supermarket benchmarks)_:"]
            for it in canned_items[:10]:
                lines.append(
                    f"- **{it['name']}** ({it['unit']}): "
                    f"Sari-sari **{fmt_php(it['palengke_min'])}–{fmt_php(it['palengke_max'])}** | "
                    f"Supermarket **{fmt_php(it['supermarket_min'])}–{fmt_php(it['supermarket_max'])}**"
                )
            lines.append(f"\n_Backed by {len(self.skus):,} Philippine retail SKUs ({sum(1 for s in self.skus if 'canned' in s.get('category',''))} canned food items) in Bern's SKU catalog._")
            return "\n".join(lines)

        # 4. Check multi-item basket estimation
        basket = self.extract_multi_item_basket(q)
        if len(basket) >= 2 or (len(basket) == 1 and basket[0]["qty"] != 1.0):
            est = self.estimate_basket(basket)
            lines = ["Here's your **2026 Philippine Price Estimate** (DTI SRP & DA Bantay Presyo benchmarks):"]
            for row in est["lines"]:
                q_val = row["qty"]
                q_lbl = int(q_val) if q_val == int(q_val) else q_val
                lines.append(
                    f"- **{q_lbl} × {row['name']}**: "
                    f"Palengke **{fmt_php(row['palengke_range'][0])}–{fmt_php(row['palengke_range'][1])}** | "
                    f"Supermarket **{fmt_php(row['supermarket_range'][0])}–{fmt_php(row['supermarket_range'][1])}**"
                )
            lines.append(
                f"\n**Estimated Total:** Palengke **{fmt_php(est['palengke_total_range'][0])}–{fmt_php(est['palengke_total_range'][1])}** · "
                f"Supermarket **{fmt_php(est['supermarket_total_range'][0])}–{fmt_php(est['supermarket_total_range'][1])}** "
                f"(typical midpoint: **{fmt_php(est['typical_total'])}**)."
            )
            return "\n".join(lines)

        # 5. Category or general overview query (English + Tagalog words: gulay, prutas, karne, isda ...)
        ql_cat = ql
        for tagalog, english in _TAGALOG_CATEGORY_WORDS.items():
            ql_cat = re.sub(rf"\b{tagalog}\b", english, ql_cat)
        ql_stem = " ".join(w[:-1] if len(w) > 4 and w.endswith("s") else w for w in re.findall(r"\w+", ql_cat))
        for cat in sorted({i["category"] for i in self.items}):
            cat_words = [w[:-1] if w.endswith("s") else w for w in re.findall(r"\b\w+\b", cat.lower()) if len(w) > 3]
            if any(re.search(rf"\b{re.escape(w)}\b", ql_stem) for w in cat_words) and any(
                re.search(rf"\b{re.escape(kw)}\b", ql_cat)
                for kw in ["list", "all", "prices", "price", "compare", "category", "mga presyo", "show", "listahan", "presyo", "ipakita", "mga"]
            ):
                cat_items = [i for i in self.items if i["category"] == cat]
                lines = [f"**Philippine {cat} Prices (2026 Benchmarks):**"]
                for it in cat_items:
                    lines.append(
                        f"- **{it['name']}** ({it['unit']}): "
                        f"Palengke **{fmt_php(it['palengke_min'])}–{fmt_php(it['palengke_max'])}** | "
                        f"Supermarket **{fmt_php(it['supermarket_min'])}–{fmt_php(it['supermarket_max'])}**"
                    )
                return "\n".join(lines)

        # 6. Specific single-item lookup
        matches = self.search_items(q, limit=3, min_score=0.34)
        if matches:
            sections = [self.format_item_summary(m["item"]) for m in matches]
            return (
                "**2026 Philippine Price Reference** _(DTI May 2026 SRP / DA Nov 2026 / WFP)_:\n\n"
                + "\n\n".join(sections)
            )

        # 7. Brand/SKU catalog lookup (e.g., "King Cup", "Young's Town", "San Marino Chili", "Star Corned Beef")
        sku_hits = self.search_skus_for_message(q, limit=5, min_score=0.45)
        if sku_hits:
            lines = ["**Philippine Grocery SKU Matches** _(DTI SRP / Open Food Facts PH / Retail Catalog)_:"]
            priced = False
            for s in sku_hits:
                price_part = ""
                if s.get("srp_php") is not None:
                    price_part = f" — SRP **{fmt_php(float(s['srp_php']))}**"
                    priced = True
                if s.get("wholesale_php") is not None:
                    price_part += f" (Wholesale ~{fmt_php(float(s['wholesale_php']))})"
                unit_part = f" [{s['unit']}]" if s.get("unit") else ""
                lines.append(f"- **{s['name']}**{unit_part}{price_part} _({s.get('source', '')})_")
            if not priced:
                lines.append("\n_I know these products, but my data has no verified price for them yet - check the shelf tag or teach me a price!_")
            return "\n".join(lines)

        # 8. Overview of common Philippine staples ONLY when user asked for general/list/overview prices
        overview_kws = ["list", "all", "common", "goods", "grocery", "groceries", "palengke", "supermarket", "dti", "srp", "bantay", "guide", "philippine", "philippines", "mga presyo", "overview", "menu",
                        "listahan", "ipakita", "presyo ng lahat", "lahat ng presyo"]
        if any(re.search(rf"\b{re.escape(kw)}\b", ql) for kw in overview_kws):
            featured_ids = ["rice_regular", "eggs_tray", "whole_chicken", "pork_liempo", "cooking_oil", "sardines", "corned_beef", "lpg_11kg"]
            id_map = {i["id"]: i for i in self.items}
            lines = [
                "Here are common **2026 Philippine Benchmark Prices** (Palengke vs Supermarket, per DTI SRP / DA / DOE):"
            ]
            for fid in featured_ids:
                if fid in id_map:
                    it = id_map[fid]
                    lines.append(
                        f"- **{it['name']}** ({it['unit']}): "
                        f"Palengke **{fmt_php(it['palengke_min'])}–{fmt_php(it['palengke_max'])}** | "
                        f"Supermarket **{fmt_php(it['supermarket_min'])}–{fmt_php(it['supermarket_max'])}**"
                    )
            lines.append("\nAsk me about any specific item (e.g. `magkano Century Tuna?`, `canned foods list`, `bigas sa Calabarzon`, or `2kg rice, 1L oil, 1 tray eggs`)!")
            return "\n".join(lines)

        return (
            "I'm specialized in **common Philippine goods** (rice, canned foods, bread, eggs, milk, chicken, pork, fish, vegetables, condiments, LPG, and fuel), "
            "and I didn't find that specific item in my 2026 Philippine price catalog yet.\n\n"
            "Try asking about items like `bigas`, `itlog`, `sardinas`, `Argentina corned beef`, `Century Tuna`, `pork liempo`, `sibuyas`, or `11kg LPG` — "
            "or teach me new items in **Train Bern**!"
        )
