"""Template-based training phrases generated from Bern's own catalogs.

Why: the hand-written phrases teach Bern *how people ask*; the catalogs know *what they ask about*. Crossing the two
gives the classifier hundreds of extra examples in which

  * the same food appears with price, nutrition and off-topic wording ("price of adobo" is not a price Bern knows,
    "calories ng adobo" is nutrition, "how to cook adobo" is out of scope), so the classifier learns to follow the
    *wording* and not the food name;
  * non-grocery prices (iphone, laptop, tesla ...) are explicitly out of scope.

Generated phrases are used for TRAINING ONLY - they are never part of the evaluation split or the held-out benchmark.

Usage:  python3 -m bern.augment          # (re)builds data/augmented_examples.json
"""
from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
AUGMENTED_PATH = BASE_DIR / "data" / "augmented_examples.json"
BENCHMARK_PATH = BASE_DIR / "data" / "eval" / "bern_benchmark.json"

PRICE_EN = ["price of {x}", "how much is {x}", "{x} price", "cost of {x}", "how much does {x} cost", "current price of {x}", "{x} price today", "what's the price of {x}"]
PRICE_TL = ["magkano ang {x}", "presyo ng {x}", "magkano po ang {x}", "magkano na ang {x} ngayon", "{x} magkano", "ano ang presyo ng {x}", "magkano ba ang {x}"]
NUTRI_EN = ["calories in {x}", "{x} calories", "nutrition facts of {x}", "how many calories does {x} have", "protein in {x}", "{x} nutrition", "macros of {x}", "kcal of {x}", "how much fat is in {x}"]
NUTRI_TL = ["ilang calories ang {x}", "calories ng {x}", "nutrisyon ng {x}", "ilan ang calories sa {x}", "gaano karaming protina ang {x}", "kcal ng {x}"]
COOK_EN = ["how to cook {x}", "recipe for {x}", "best way to cook {x}", "how do i make {x}", "how long to cook {x}", "ingredients for {x}"]
COOK_TL = ["paano magluto ng {x}", "paano lutuin ang {x}", "recipe ng {x}", "ano ang sangkap ng {x}"]
NON_GROCERY = [
    "iphone 15", "a samsung phone", "a laptop", "a macbook", "a ps5", "a tesla", "a honda civic", "a rolex", "a gold ring", "airpods", "a smart tv", "an aircon",
    "a motorcycle", "a bicycle", "a wedding ring", "a plane ticket to cebu", "a bus ticket to baguio", "a condo in makati", "a bitcoin", "a share of ayala stock",
    "a movie ticket", "a concert ticket", "college tuition", "a haircut", "a hotel room in boracay", "a grab ride to bgc", "a pair of nike shoes", "a gaming pc",
]
NON_GROCERY_TL = ["iphone", "laptop", "motor", "bagong cellphone", "tiket sa eroplano", "condo sa manila", "sapatos na nike", "pamasahe sa grab", "kuryente", "tuition fee"]

# plain nouns that read naturally after "calories in ..." / "magkano ang ..."
EXTRA_FOODS = ["rice", "egg", "chicken", "pork", "beef", "fish", "milk", "bread", "banana", "mango", "tomato", "onion", "garlic", "tuna", "sardines", "corned beef",
               "noodles", "pandesal", "kangkong", "tofu", "hotdog", "longganisa", "tapa", "tocino", "bangus", "tilapia", "galunggong", "ampalaya", "kamote", "saging"]


def _clean(name: str) -> str:
    name = re.sub(r"\([^)]*\)", " ", name)
    name = re.sub(r"\s+", " ", name).strip(" ,-/")
    return name.lower()


def _price_names() -> list[str]:
    from bern.ph_prices import PHPriceExpert

    names: list[str] = []
    for it in PHPriceExpert().items:
        for cand in [it["name"]] + list(it.get("aliases", []))[:4]:
            c = _clean(cand)
            if 2 < len(c) <= 28 and c not in names:
                names.append(c)
    return names


def _dish_and_food_names() -> tuple[list[str], list[str]]:
    from bern.ph_nutrition import get_shared_expert

    expert = get_shared_expert()
    dishes: list[str] = []
    foods: list[str] = []
    for f in expert.foods:
        if f["source"] == "dish" and f["type"] == "dish":
            dishes.append(_clean(f["name"].split("(")[0]))
            for a in f.get("aliases", [])[:2]:
                dishes.append(_clean(a))
        elif f["source"] == "restaurant" and f.get("brand") in {"McDonald's", "Jollibee"} and f.get("featured"):
            base = _clean(f["name"])
            foods.append(f"{f['brand'].lower()} {base}" if f["brand"].lower() not in base else base)
        elif f["source"] == "philfct" and expert._docs[expert.foods.index(f)]["featured"] > 1.0:  # noqa: SLF001 - curated staples
            foods.append(_clean(f["name"]))
    uniq = lambda xs: list(dict.fromkeys(x for x in xs if 2 < len(x) <= 34))  # noqa: E731
    return uniq(dishes), uniq(foods + EXTRA_FOODS)


def build_augmented_examples(seed: int = 7) -> list[dict[str, str]]:
    rng = random.Random(seed)
    price_names = _price_names()
    dishes, foods = _dish_and_food_names()
    out: list[dict[str, str]] = []

    def add(intent: str, templates: list[str], names: list[str], per_name: int) -> None:
        for name in names:
            for tpl in rng.sample(templates, k=min(per_name, len(templates))):
                out.append({"text": tpl.format(x=name), "intent": intent})

    pick = lambda names, k: rng.sample(names, k=min(k, len(names)))  # noqa: E731 - cap each family so generated data never dominates
    add("ph_price_lookup", PRICE_EN, pick(price_names, 70), 1)
    add("ph_price_lookup", PRICE_TL, pick(price_names, 70), 1)
    nutri_names = foods + dishes
    add("ph_nutrition_lookup", NUTRI_EN, pick(nutri_names, 75), 1)
    add("ph_nutrition_lookup", NUTRI_TL, pick(nutri_names, 75), 1)
    add("out_of_scope", COOK_EN, pick(dishes + foods[:25], 45), 1)
    add("out_of_scope", COOK_TL, pick(dishes, 30), 1)
    add("out_of_scope", PRICE_EN, NON_GROCERY, 1)
    add("out_of_scope", PRICE_TL, NON_GROCERY_TL, 1)

    # keep it clean: no duplicates, and never an exact copy of a held-out benchmark message
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())  # noqa: E731
    held_out = {norm(i["text"]) for i in json.loads(BENCHMARK_PATH.read_text("utf-8"))["items"]} if BENCHMARK_PATH.exists() else set()
    seen, clean = set(), []
    for ex in out:
        key = norm(ex["text"])
        if key in seen or key in held_out:
            continue
        seen.add(key)
        clean.append(ex)
    return clean


def save_augmented(path: Path = AUGMENTED_PATH, seed: int = 7) -> dict[str, Any]:
    examples = build_augmented_examples(seed)
    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex["intent"]] = counts.get(ex["intent"], 0) + 1
    payload = {
        "description": "Template phrases generated from Bern's catalogs - used for training only, never for evaluation",
        "seed": seed,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": counts,
        "examples": examples,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), "utf-8")
    return payload


def load_augmented(path: Path = AUGMENTED_PATH) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    return json.loads(Path(path).read_text("utf-8")).get("examples", [])


if __name__ == "__main__":
    payload = save_augmented()
    print(f"wrote {AUGMENTED_PATH} - {sum(payload['counts'].values())} examples {payload['counts']}")
    rng = random.Random(1)
    for ex in rng.sample(payload["examples"], 12):
        print(f"  [{ex['intent']}] {ex['text']}")
