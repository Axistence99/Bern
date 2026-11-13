"""Philippine food & nutrition-facts expert for Bern.

Unifies five open / published sources behind one search + answer API:

  philfct     DOST-FNRI Philippine Food Composition Tables (1,542 foods, lab / calculated, per 100 g edible portion)
  usda        USDA FoodData Central (SR Legacy + Foundation + FNDDS, CC0) - ground beef, McDonald's USA, fast-food chicken ...
  off         Open Food Facts Philippines (packaged / canned / milk products, ODbL, label data entered by the community)
  dish        Filipino dishes (adobo, pares, bopis, sisig ...) estimated from standard recipes using FNRI + USDA ingredient values
  restaurant  McDonald's PH / Jollibee published or reported values (per serving)

Every record carries a ``quality`` tag so answers can say how trustworthy a number is.
"""
from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXT_DIR = DATA_DIR / "external"
PHILFCT_PATH = EXT_DIR / "philfct_fnri_foods.json.gz"
USDA_PATH = EXT_DIR / "usda_fdc_core.json.gz"
OFF_PATH = EXT_DIR / "openfoodfacts_ph_nutrition.json.gz"
CURATED_PATH = DATA_DIR / "ph_nutrition_curated.json"

NUTRIENT_FIELDS: list[tuple[str, str, str]] = [
    ("kcal", "Energy", "kcal"),
    ("protein_g", "Protein", "g"),
    ("fat_g", "Fat", "g"),
    ("sat_fat_g", "Saturated fat", "g"),
    ("carbs_g", "Carbohydrates", "g"),
    ("sugars_g", "Sugars", "g"),
    ("fiber_g", "Fiber", "g"),
    ("sodium_mg", "Sodium", "mg"),
    ("cholesterol_mg", "Cholesterol", "mg"),
    ("calcium_mg", "Calcium", "mg"),
    ("iron_mg", "Iron", "mg"),
    ("potassium_mg", "Potassium", "mg"),
    ("vitamin_c_mg", "Vitamin C", "mg"),
]
NUTRIENT_KEYS = [k for k, _, _ in NUTRIENT_FIELDS]
CORE_KEYS = ["kcal", "protein_g", "fat_g", "carbs_g", "sodium_mg"]

SOURCE_LABELS = {
    "philfct": "DOST-FNRI Philippine Food Composition Tables",
    "usda": "USDA FoodData Central",
    "off": "Open Food Facts (community label data)",
    "dish": "Bern standard-recipe estimate",
    "restaurant": "Restaurant-published / reported values",
}
QUALITY_LABELS = {
    "lab_analysis": "FNRI lab-analysed / calculated values",
    "usda_reference": "USDA reference database",
    "manufacturer_label": "Printed product label, entered by the Open Food Facts community",
    "recipe_estimate": "Estimated from a standard recipe - real values vary with recipe and portion",
    "official_published": "Published by the restaurant chain",
    "reported_secondary": "Reported on third-party sites - unverified, may differ from today's menu",
}

PHILFCT_GROUPS = {
    "A": "Rice, Bread, Noodles & Cereals",
    "B": "Starchy Roots & Tubers",
    "C": "Nuts, Beans & Seeds",
    "D": "Vegetables",
    "E": "Fruits",
    "F": "Meat & Poultry",
    "G": "Fish & Seafood",
    "H": "Eggs",
    "J": "Milk & Dairy",
    "K": "Fats & Oils",
    "M": "Sugar & Sweets",
    "N": "Condiments & Spices",
    "P": "Alcoholic Beverages",
    "Q": "Beverages",
    "R": "Mixed Dishes & Prepared Foods",
    "S": "Baby Foods",
    "T": "Miscellaneous",
}

# PhilFCT abbreviations -> words (applied on whole words only)
_ABBREV = [
    (r"\bcnd\b", "canned"), (r"\bpwdr\b", "powder"), (r"\bsce\b", "sauce"), (r"\binst\b", "instant"),
    (r"\bflvr\b", "flavor"), (r"\bflvd\b", "flavored"), (r"\brecon\b", "reconstituted"), (r"\bsw\b", "sweet"),
    (r"\bsr\b", "sour"), (r"\blvs\b", "leaves"), (r"\bw/o\b", "without"), (r"\bw/", "with "),
    (r"\bunckd\b", "uncooked"), (r"\bckd\b", "cooked"), (r"\bbbq\b", "barbecue"), (r"\bprep\b", "prepared"),
    (r"\bgrtd\b", "grated"), (r"\bsweetn\b", "sweetened"), (r"\bcond\b", "condensed"), (r"\bfort\b", "fortified"),
    (r"\breg\b", "regular"), (r"\benr\b", "enriched"), (r"\bstr\b", "strained"), (r"\bpetioles\b", "stalks"),
    (r"\bMLP\b", "milk, lard & pancake mix"), (r"\bdrnk\b", "drink"), (r"\bchoc\b", "chocolate"),
]

# Taglish / local words -> English tokens that the English-named sources use
_SYNONYMS: dict[str, list[str]] = {
    "bawang": ["garlic"], "sibuyas": ["onion"], "sibuyas tagalog": ["shallot"], "gatas": ["milk"], "giniling": ["ground"],
    "baka": ["beef"], "karne": ["meat"], "baboy": ["pork"], "manok": ["chicken"], "itlog": ["egg"], "isda": ["fish"],
    "kanin": ["rice", "boiled"], "bigas": ["rice", "well", "milled"], "sardinas": ["sardine"], "mantika": ["oil"], "asukal": ["sugar"],
    "asin": ["salt"], "suka": ["vinegar"], "toyo": ["soy", "sauce"], "patis": ["fish", "sauce"], "saging": ["banana"],
    "kamatis": ["tomato"], "talong": ["eggplant"], "ampalaya": ["bitter", "melon"], "kamote": ["sweet", "potato"],
    "tokwa": ["soybean", "curd"], "hipon": ["shrimp"], "pusit": ["squid"], "alimango": ["crab"], "tahong": ["mussel"],
    "mani": ["peanut"], "lata": ["canned"], "delata": ["canned"], "evap": ["evaporated"], "condensada": ["condensed"],
    "kondensada": ["condensed"], "mcdo": ["mcdonald"], "mcdonalds": ["mcdonald"], "mcdonald": ["mcdonald"],
    "jabee": ["jollibee"], "chowking": ["chowking"], "pandesal": ["pan", "de", "sal"], "liempo": ["pork", "belly"],
    "kasim": ["pork", "picnic"], "lechon": ["roast", "pork"], "tinapa": ["smoked"], "kalabasa": ["squash"],
    "repolyo": ["cabbage"], "sitaw": ["string", "bean"], "kangkong": ["swamp", "cabbage"], "labanos": ["radish"],
    "mangga": ["mango"], "pinya": ["pineapple"], "niyog": ["coconut"], "gata": ["coconut", "milk"], "bangus": ["milkfish"],
    "tilapia": ["tilapia"], "galunggong": ["round", "scad"], "dilis": ["anchovy"], "tuyo": ["dried"], "luya": ["ginger"],
    "fishball": ["fish", "ball"], "meatloaf": ["meat", "loaf"], "coke": ["cola"], "kape": ["coffee"], "tsaa": ["tea"],
    "halohalo": ["halu", "halo"], "tokneneng": ["quail", "egg"],
}

_STOPWORDS = {
    "a", "an", "and", "are", "about", "ang", "ano", "ba", "be", "calorie", "calories", "carbs", "carb", "carbohydrate",
    "carbohydrates", "can", "content", "contain", "contains", "do", "does", "fact", "facts", "for", "get", "give",
    "has", "have", "how", "ilan", "ilang", "in", "info", "information", "is", "it", "kcal", "magkano", "many", "mga",
    "much", "na", "nang", "ng", "nutrient", "nutrients", "nutrisyon", "nutrition", "nutritional", "nutritive", "of",
    "on", "per", "please", "pls", "sa", "sustansya", "tell", "the", "there", "this", "to", "value", "values", "what",
    "whats", "what's", "with", "yung", "ung", "100g", "100", "gram", "grams", "serving", "servings", "size", "macros",
    "macro", "protein", "proteins", "fat", "fats", "sodium", "cholesterol", "fiber", "fibre", "i", "me", "my", "show",
    "list", "meron", "mayroon", "kumain", "kain", "eat", "eating", "ate", "get", "intake", "total", "healthy",
    "laman", "taglay", "gaano", "karami", "karaming", "dami", "isang", "isa", "ilan",
    "high", "low", "good", "bad", "fattening", "fatty", "rich", "enough", "too", "lots", "source", "mataas", "mababa", "which", "more",
    "less", "better", "worse", "safe", "unhealthy",
    # household units / quantities ("kcal of a cup of rice", "2 slices of bread") describe the amount, not the food
    "cup", "cups", "slice", "slices", "piece", "pieces", "pc", "pcs", "bowl", "bowls", "glass", "glasses", "tbsp", "tsp", "spoon",
    "tablespoon", "teaspoon", "plate", "plates", "gram", "grams", "kilo", "kilos", "kg", "ounce", "oz", "ml", "order", "orders",
}
_PREP_WORDS = {
    "boiled", "fried", "broiled", "barbecue", "roasted", "cooked", "baked", "sauteed", "steamed", "smoked",
    "stewed", "grilled", "toasted",
}
# very common Philippine staples get a ranking boost for short, generic queries (matched on the PhilFCT source name)
_FEATURED_RE = re.compile(
    r"^(Garlic bulb|Onion, Bombay bulb|Onion, garden shallot bulb|Onion, spring|Ginger|Tomato|Eggplant|Carrot|Cabbage, green|"
    r"Potato|Sweet potato, (white|yellow)|Swamp cabbage lvs|Squash fruit|Bitter melon/gourd fruit|"
    r"Rice, well-milled(, boiled|, fried)?|Rice, undermilled|Egg, chicken, whole(, boiled)?|Egg, duck, whole|"
    r"Milk, (cow|carabao|evaporated|recombined|pwdr, full cream|pwdr, skim|sweetn, cond, filled|goat)|"
    r"Bread, pan de sal|Bread, white, loaf|Noodles, wheat, canton, dry|Noodles, wheat, inst, w/ flvr|"
    r"Pork belly|Pork Boston butt|Pork liver|Pork lung|Pork heart|Beef brisket|Beef chuck|Beef, lean|Chicken breast|Chicken thigh & leg|Chicken wing|"
    r"Chicken leg/drumstick|Corned beef, cnd|Luncheon meat, cnd|Sausage, hotdog|Sardines, in tomato sce, cnd|"
    r"Milkfish|Tilapia|Round scad|Shrimp, banana prawn|Banana, (saba|lakatan)|Mango, ripe|Calamansi/Philippine lemon|"
    r"Soy sauce|Fish sauce|Catsup, banana|Sugar, brown|Oil, coconut|Oil, corn|Peanut butter|"
    r"Sausage, Vienna|Meat loaf|Potato, French-fried|Hamburger sandwich, reg|Cheeseburger sandwich)$", re.I)
# finer ordering among staples (exact PhilFCT source name, lower-case -> multiplier)
_FEATURED_BOOST = {
    "onion, bombay bulb": 1.36, "onion, garden shallot bulb": 1.3, "onion, spring": 1.2,
    "milk, cow": 1.42, "milk, evaporated": 1.36, "milk, pwdr, full cream": 1.32, "milk, sweetn, cond, filled": 1.3,
    "milk, carabao": 1.22, "milk, recombined": 1.22, "milk, goat": 1.12, "milk, pwdr, skim": 1.2,
    "rice, well-milled, boiled": 1.75, "rice, well-milled": 1.28, "rice, well-milled, fried": 1.6, "rice, undermilled": 1.1,
    "egg, chicken, whole": 1.3, "egg, chicken, whole, boiled": 1.22, "pork belly": 1.3, "chicken breast": 1.25,
    "garlic bulb": 1.3, "corned beef, cnd": 1.3, "sardines, in tomato sce, cnd": 1.3, "luncheon meat, cnd": 1.3,
    "egg, duck, fertilized, boiled": 1.4, "vinegar, coconut": 1.3, "horseradish tree lvs": 1.35, "sugar, white, refined": 1.4,
    "mango, manila super, ripe": 1.3, "pineapple": 1.2, "papaya fruit, ripe": 1.2, "soybean cheese, hard curd": 1.25,
    "soybean cheese, soft curd": 1.25, "fish ball": 1.3, "halu-halo, w/ ice": 1.4, "halu-halo, cnd": 1.0,
}
# extra search aliases for items whose official names differ from everyday Philippine wording (lower-case source name)
_EXTRA_ALIASES = {
    "oil, coconut": ["cooking oil", "mantika", "langis"], "oil, corn": ["cooking oil", "mantika", "langis"],
    "halu-halo, w/ ice": ["halo halo", "halo-halo"], "halu-halo, cnd": ["halo halo canned"],
    "soybean cheese, soft curd": ["tofu", "tokwa", "bean curd"], "soybean cheese, hard curd": ["tofu", "tokwa", "bean curd"],
    "bread, pan de sal": ["pandesal", "pan de sal"], "banana, saba, w/ sugar, wrapped, fried": ["turon", "banana lumpia"],
    "banana, saba, w/ sugar, fried": ["banana cue", "banana q", "bananacue"], "sausage, hotdog": ["hotdog", "hot dog", "tender juicy"],
    "egg, duck, fertilized, boiled": ["balut", "whole balut"], "potato, french-fried": ["french fries", "fries"],
    "noodles, wheat, canton, dry": ["pancit canton", "canton"], "noodles, rice": ["bihon", "pancit bihon"],
    "rice, well-milled, fried": ["garlic rice", "sinangag", "fried rice"], "rice, well-milled, boiled": ["steamed rice", "plain rice", "cooked rice"],
}
_NUTRIENT_WORDS = {
    "protein": "protein_g", "proteins": "protein_g", "carb": "carbs_g", "carbs": "carbs_g", "carbohydrate": "carbs_g",
    "carbohydrates": "carbs_g", "fat": "fat_g", "fats": "fat_g", "sodium": "sodium_mg", "salt": "sodium_mg",
    "cholesterol": "cholesterol_mg", "fiber": "fiber_g", "fibre": "fiber_g", "sugar": "sugars_g", "sugars": "sugars_g",
    "calcium": "calcium_mg", "iron": "iron_mg", "potassium": "potassium_mg", "vitamin c": "vitamin_c_mg",
}

NUTRITION_STRONG_RE = re.compile(
    r"\b(calories?|kcal|nutrition(?:al)?|nutrients?|nutrisyon|nutrisyon|sustansya|macros?|how many cals?|ilang cals?)\b", re.I)
NUTRITION_WEAK_RE = re.compile(
    r"\b(protein|proteins|carbs?|carbohydrates?|sodium|cholesterol|fib(?:er|re)|protina|"
    r"(?:fat|sugar) content|how much (?:fat|sugar|protein|carbs?|sodium))\b", re.I)
_PRICE_WORDS_RE = re.compile(r"\b(magkano|presyo|price|prices|cost|peso|pesos|php)\b|₱", re.I)

# household measures for PhilFCT items (PhilFCT itself is per-100 g only). (name regex, [(label, grams), ...])
_HOUSEHOLD: list[tuple[str, list[tuple[str, float]]]] = [
    (r"^garlic bulb$", [("1 clove", 3), ("1 head (about 10 cloves)", 30)]),
    (r"^onion, bombay bulb$", [("1 medium onion", 110), ("1 tbsp chopped", 10)]),
    (r"^onion, garden shallot bulb$", [("1 medium bulb", 15), ("1 tbsp chopped", 10)]),
    (r"^onion, spring$", [("1 stalk", 15)]),
    (r"^ginger$", [("1 thumb (2 cm)", 10)]),
    (r"^milk, (cow|carabao|goat)$", [("1 glass (240 mL)", 244)]),
    (r"^milk, evaporated", [("1/2 cup (125 mL)", 126), ("1 tbsp", 16)]),
    (r"^milk, sweetn", [("2 tbsp", 40), ("1 tbsp", 20)]),
    (r"^milk, pwdr", [("1 glass (4 tbsp powder)", 32)]),
    (r"^milk, recombined", [("1 glass (240 mL)", 244)]),
    (r"^egg, chicken, whole", [("1 medium egg", 50)]),
    (r"^egg, duck, whole", [("1 egg", 70)]),
    (r"^egg, quail", [("1 egg", 9)]),
    (r"^egg, duck, fertilized, boiled", [("1 balut", 70)]),
    (r"^rice, well-milled, boiled$", [("1 cup cooked", 160)]),
    (r"^rice, well-milled, fried$", [("1 cup garlic rice", 150)]),
    (r"^rice, undermilled, boiled", [("1 cup cooked", 160)]),
    (r"^bread, pan de sal$", [("1 piece", 35)]),
    (r"^bread, pan de monay$", [("1 piece", 50)]),
    (r"^bread, sweet roll$", [("1 piece ensaymada", 60)]),
    (r"^bread, white, loaf$", [("1 slice", 30)]),
    (r"^noodles, wheat, canton", [("1 pack (60 g dry)", 60)]),
    (r"^noodles, wheat, inst", [("1 pack (60 g dry)", 60)]),
    (r"^instant noodles", [("1 pack (60 g dry)", 60)]),
    (r"^banana, saba$", [("1 piece", 80)]),
    (r"^banana, lakatan$", [("1 piece", 100)]),
    (r"^oil, ", [("1 tbsp", 14)]),
    (r"^sugar, brown$", [("1 tbsp", 12)]),
    (r"^soy sauce$", [("1 tbsp", 16)]),
    (r"^fish sauce$", [("1 tbsp", 18)]),
    (r"^vinegar", [("1 tbsp", 15)]),
    (r"^catsup", [("1 tbsp", 17)]),
    (r"^mayonnaise$", [("1 tbsp", 14)]),
    (r"^peanut butter$", [("1 tbsp", 16)]),
    (r"^corned beef, cnd$", [("1 serving (1/3 can)", 55)]),
    (r"^luncheon meat, cnd$", [("1 slice", 40)]),
    (r"^sausage, (hotdog|vienna)", [("1 piece", 45)]),
    (r"^sardines?, .*cnd$", [("1 serving (1/3 can)", 55)]),
    (r"^(tuna|milkfish|mackerel).*cnd$", [("1 serving (1/3 can)", 55)]),
    (r"^rice cake, (puto|cuchinta|bibingka)", [("1 piece", 40)]),
    (r"^halu-halu", [("1 regular glass", 300)]),
    (r"^banana, saba, w/ sugar, wrapped, fried$", [("1 piece turon", 80)]),
    (r"^fish ball", [("1 piece", 12)]),
    (r"^steamed bun", [("1 piece siopao", 100)]),
    (r"^meat pie", [("1 piece empanada", 90)]),
]
_HOUSEHOLD_RE = [(re.compile(rx, re.I), items) for rx, items in _HOUSEHOLD]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ----------------------------------------------------------------------------------------------- helpers
def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us")):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("'", "").replace("’", "")
    return [_stem(t) for t in _TOKEN_RE.findall(text)]


def _expand_abbrev(name: str) -> str:
    out = name
    for rx, repl in _ABBREV:
        out = re.sub(rx, repl, out, flags=re.I)
    return re.sub(r"\s+", " ", out).replace(" ,", ",").strip(" ,")


def _round(value: float | None, nd: int = 1) -> float | None:
    if value is None:
        return None
    return round(float(value), nd)


def _read_gz_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


_INT_KEYS = {"kcal", "sodium_mg", "cholesterol_mg", "calcium_mg", "potassium_mg"}


def _scale(per: dict[str, float], factor: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in per.items():
        if v is None:
            continue
        val = v * factor
        if k in _INT_KEYS:
            out[k] = round(val) if val >= 10 else round(val, 1)
        else:
            out[k] = round(val, 1)
    return out


# ----------------------------------------------------------------------------------------------- loaders
def load_philfct(path: Path = PHILFCT_PATH) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    out = []
    for f in _read_gz_json(path)["foods"]:
        per = {k: v for k, v in f["n"].items() if k in NUTRIENT_KEYS}
        if "kcal" not in per:
            continue
        name = _expand_abbrev(f["name"])
        name = name[0].upper() + name[1:] if name else name
        servings: list[dict[str, Any]] = []
        for rx, items in _HOUSEHOLD_RE:
            if rx.search(f["name"]):
                servings = [{"label": lab, "grams": g} for lab, g in items]
                break
        out.append({
            "id": f"philfct:{f['id']}",
            "source": "philfct",
            "type": "generic",
            "name": name,
            "source_name": f["name"],
            "local_name": f["alt"],
            "brand": "",
            "category": PHILFCT_GROUPS.get(f["group_code"], f["group"]),
            "per_100g": per,
            "servings": servings or [{"label": "100 g", "grams": 100.0}],
            "quality": "lab_analysis",
            "source_ref": f["url"] or f"PhilFCT {f['id']}",
            "notes": (f"Scientific name: {f['sci']}. " if f["sci"] else "") +
                     (f"Edible portion: {f['ep']:g}% (values are per 100 g edible portion)." if f["ep"] else
                      "Values are per 100 g edible portion."),
            "aliases": _EXTRA_ALIASES.get(f["name"].lower(), []),
            "extra": {"philfct_id": f["id"], "group_code": f["group_code"]},
        })
    return out


_USDA_ALIASES = {169655: ["sugar", "white sugar", "asukal"], 174036: ["ground beef", "giniling na baka"]}
_USDA_RESTAURANTS = {
    "MCDONALD'S": "McDonald's", "BURGER KING": "Burger King", "WENDY'S": "Wendy's", "TACO BELL": "Taco Bell",
    "PIZZA HUT": "Pizza Hut", "KFC": "KFC", "DOMINO'S": "Domino's", "SUBWAY": "Subway", "ARBY'S": "Arby's",
    "CARL'S JR": "Carl's Jr.", "LITTLE CAESARS": "Little Caesars", "DAIRY QUEEN": "Dairy Queen",
    "CHICK-FIL-A": "Chick-fil-A", "POPEYES": "Popeyes", "STARBUCKS": "Starbucks", "DUNKIN": "Dunkin'",
}


def _usda_default_serving(portions: list[list[Any]]) -> list[dict[str, Any]]:
    if not portions:
        return [{"label": "100 g", "grams": 100.0}]
    items = [{"label": str(lab), "grams": float(g)} for lab, g in portions]
    prio = [r"clove|slice|piece|item|patty|egg|medium|serving|fillet|link|nugget|sandwich|burger", r"tbsp|tablespoon|tsp|cup|oz"]
    for rx in prio:
        for i, it in enumerate(items):
            if re.search(rx, it["label"], re.I):
                return [items[i]] + items[:i] + items[i + 1:]
    return items


def load_usda(path: Path = USDA_PATH) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    out = []
    ds_label = {"sr_legacy": "SR Legacy", "foundation": "Foundation Foods", "fndds": "FNDDS survey foods"}
    for f in _read_gz_json(path)["foods"]:
        per = {k: v for k, v in f["n"].items() if k in NUTRIENT_KEYS}
        name = f["name"].strip()
        brand, ftype = "", "generic"
        head = name.split(",", 1)[0].strip().upper()
        if head in _USDA_RESTAURANTS:
            brand = _USDA_RESTAURANTS[head]
            name = name.split(",", 1)[1].strip() if "," in name else name
            name = name[:1].upper() + name[1:]
            ftype = "restaurant"
        out.append({
            "id": f"usda:{f['fdc_id']}",
            "source": "usda",
            "aliases": _USDA_ALIASES.get(f["fdc_id"], []),
            "type": ftype,
            "name": name,
            "source_name": f["name"],
            "local_name": "",
            "brand": brand,
            "category": f.get("category") or "",
            "per_100g": per,
            "servings": _usda_default_serving(f.get("portions", [])),
            "quality": "usda_reference",
            "source_ref": f"https://fdc.nal.usda.gov/fdc-app.html#/food-details/{f['fdc_id']}/nutrients",
            "notes": f"USDA {ds_label.get(f['dataset'], f['dataset'])}" + (" (US menu item - Philippine recipe/portion may differ)." if ftype == "restaurant" else "."),
            "extra": {"fdc_id": f["fdc_id"], "dataset": f["dataset"], "region": "US" if ftype == "restaurant" else ""},
        })
    return out


_OFF_CATEGORY_MAP = [
    (("canned-foods", "canned-fishes", "canned-meats", "canned-seafood", "canned-vegetables", "canned-fruits"), "Canned Goods"),
    (("sardines", "tunas", "corned", "luncheon", "meat-loaf"), "Canned Goods"),
    (("milks", "dairies", "dairy", "yogurts", "cheeses", "evaporated-milks", "condensed-milks", "powdered-milks"), "Milk & Dairy"),
    (("instant-noodles", "noodles", "pastas", "instant-foods"), "Noodles & Pasta"),
    (("biscuits", "cookies", "crackers", "snacks", "chips", "crisps", "salty-snacks", "sweet-snacks", "confectioneries", "chocolates"), "Snacks & Sweets"),
    (("beverages", "sodas", "juices", "teas", "coffees", "energy-drinks", "waters"), "Beverages"),
    (("sauces", "condiments", "soy-sauces", "vinegars", "seasonings", "spreads"), "Sauces & Condiments"),
    (("breads", "pastries", "cakes", "bakery"), "Bread & Bakery"),
    (("meats", "sausages", "hams", "hotdogs", "processed-meats", "frozen-foods"), "Meat & Frozen"),
    (("cereals", "rices", "breakfast-cereals", "oats"), "Cereals & Rice"),
    (("desserts", "ice-creams", "frozen-desserts"), "Desserts"),
]


def _off_category(cats: Iterable[str], name: str) -> str:
    text = " ".join(cats).lower() + " " + name.lower()
    for keys, label in _OFF_CATEGORY_MAP:
        if any(k in text for k in keys):
            return label
    return "Packaged Foods"


def load_off(path: Path = OFF_PATH) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    out, seen = [], set()
    for p in _read_gz_json(path)["products"]:
        per = {k: v for k, v in p["per_100g"].items() if k in NUTRIENT_KEYS}
        key = (p["brand"].lower(), p["name"].lower(), per.get("kcal"))
        if key in seen:
            continue
        seen.add(key)
        servings = []
        size_txt = p.get("serving_size") or ""
        if p.get("serving_g"):
            label = size_txt if size_txt else f"1 serving ({p['serving_g']:g} g)"
            servings = [{"label": label, "grams": float(p["serving_g"])}]
        servings.append({"label": "100 g", "grams": 100.0})
        out.append({
            "id": f"off:{p['code']}",
            "source": "off",
            "type": "product",
            "name": p["name"],
            "source_name": p["name"],
            "local_name": "",
            "brand": p["brand"],
            "category": _off_category(p["categories"], p["name"]),
            "per_100g": per,
            "servings": servings,
            "quality": "manufacturer_label",
            "source_ref": f"https://world.openfoodfacts.org/product/{p['code']}",
            "notes": (f"Pack size: {p['quantity']}. " if p["quantity"] else "") + "Label data entered by the Open Food Facts community (ODbL).",
            "extra": {"barcode": p["code"], "nutriscore": p.get("nutriscore", ""), "nova": p.get("nova"),
                      "off_categories": p["categories"][:5]},
        })
    return out


def load_curated(path: Path = CURATED_PATH) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text("utf-8"))
    return data.get("foods", [])


# ----------------------------------------------------------------------------------------------- expert
class PHNutritionExpert:
    """Search and explain nutrition facts for Philippine foods, products, dishes and restaurant items."""

    SOURCE_WEIGHT = {"dish": 1.06, "restaurant": 1.03, "philfct": 1.0, "usda": 0.92, "off": 0.88}

    def __init__(self, philfct: Path = PHILFCT_PATH, usda: Path = USDA_PATH, off: Path = OFF_PATH,
                 curated: Path = CURATED_PATH) -> None:
        self.paths = {"philfct": philfct, "usda": usda, "off": off, "curated": curated}
        self.foods: list[dict[str, Any]] = []
        self.foods += load_curated(curated)
        self.foods += load_philfct(philfct)
        self.foods += load_off(off)
        self.foods += load_usda(usda)
        self.by_id: dict[str, dict[str, Any]] = {f["id"]: f for f in self.foods}
        self._build_index()

    # -- index -------------------------------------------------------------------------------------
    def _build_index(self) -> None:
        self._docs: list[dict[str, Any]] = []
        postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        df: Counter[str] = Counter()
        for i, f in enumerate(self.foods):
            name_core = re.sub(r"\([^)]*\)", " ", f["name"])
            core_toks = tokenize(name_core)
            full_toks = tokenize(f["name"])
            local_toks = tokenize(f.get("local_name", ""))
            alias_toks = tokenize(" ".join(f.get("aliases", [])))
            brand_toks = tokenize(f.get("brand", ""))
            cat_toks = tokenize(f.get("category", ""))
            doc = {
                "name_toks": set(core_toks), "name_len": len(set(core_toks)),
                "prep": bool(set(core_toks) & _PREP_WORDS), "name_norm": " ".join(core_toks),
                "first": core_toks[0] if core_toks else "", "brand_toks": set(brand_toks),
                "local_len": max(1, len(set(local_toks))) if local_toks else 0,
                "alias_toks": set(alias_toks), "local_toks": set(local_toks),
                "raw": "raw" in core_toks,
                "featured": (_FEATURED_BOOST.get(f.get("source_name", "").lower(),
                                                 1.2 if _FEATURED_RE.match(f.get("source_name", "")) else 1.0)
                             if f["source"] == "philfct" else 1.0),
                "us_menu": f.get("type") == "restaurant" and f.get("extra", {}).get("region") == "US" and f["source"] == "usda",
            }
            self._docs.append(doc)
            weights: dict[str, float] = {}
            for toks, w in ((cat_toks, 0.35), (full_toks, 0.8), (local_toks, 0.9), (alias_toks, 0.95), (brand_toks, 0.9), (core_toks, 1.0)):
                for t in toks:
                    if weights.get(t, 0.0) < w:
                        weights[t] = w
            for t, w in weights.items():
                postings[t].append((i, w))
                df[t] += 1
        n = max(1, len(self.foods))
        self._postings = dict(postings)
        self._idf = {t: math.log(1.0 + n / (1.0 + c)) for t, c in df.items()}

    # -- query handling ----------------------------------------------------------------------------
    @staticmethod
    def nutrient_focus(message: str) -> str | None:
        low = message.lower()
        if "vitamin c" in low:
            return "vitamin_c_mg"
        for word, key in _NUTRIENT_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", low):
                return key
        return None

    @staticmethod
    def has_nutrition_signal(message: str) -> bool:
        """True for 'calories in garlic', 'protein in egg' ... but not for 'magkano protein powder?'."""
        if NUTRITION_STRONG_RE.search(message):
            return True
        return bool(NUTRITION_WEAK_RE.search(message)) and not _PRICE_WORDS_RE.search(message)

    _DAILY_RE = re.compile(r"\b(should|need|needed|dapat|kailangan|recommended|ideal|enough)\b.*\b(per day|a day|each day|daily|araw|kada araw)\b", re.I)

    def try_answer(self, message: str, min_score: float = 0.5) -> str | None:
        """Answer only when the message really maps to a food/topic/meal; otherwise None."""
        if self._DAILY_RE.search(message) and re.search(r"calor|kcal|protein|carb|fat|sodium", message, re.I):
            return ("**Daily needs depend on age, sex, weight and activity**, so I can't give you a personal number. "
                    "Nutrition labels use a **2,000 kcal/day** reference diet, and the DOST-FNRI Philippine Dietary Reference Intakes "
                    "(PDRI) list exact values by age and sex - a registered nutritionist-dietitian can tailor this for you.\n\n"
                    "I *can* tell you what a food contains - try `calories in kanin`, `protein in itlog` or `Jollibee Chickenjoy calories`, "
                    "or total a meal: `calories of 2 pc chickenjoy and 1 cup rice`.")
        if self.answer_topic(message):
            return self.answer_nutrition_query(message)
        parsed = self.parse_meal_text(message)
        if len(parsed) >= 2:
            meal = self.answer_meal(message)
            if meal:
                return meal
        if re.search(r"\b(vs\.?|versus|compare|kumpara)\b", message.lower()):
            cmp = self.answer_compare(message)
            if cmp:
                return cmp
        found = self.search(message, limit=1)
        if found and found[0]["score"] >= min_score:
            return self.answer_nutrition_query(message)
        return None

    def _query_groups(self, query: str, keep_local: bool = False) -> tuple[list[list[str]], bool]:
        """Turn a query into AND-groups of OR-alternatives, e.g. 'kanin' -> [['rice', 'kanin'], ['boiled']]."""
        low = query.lower()
        for phrase, alts in _SYNONYMS.items():          # multi-word local phrases first
            if " " in phrase and phrase in low:
                low = low.replace(phrase, " ".join(alts))
        raw_toks = _TOKEN_RE.findall(low.replace("'", "").replace("’", ""))
        prep_in_query = any(_stem(t) in _PREP_WORDS or t in _PREP_WORDS for t in raw_toks)
        groups: list[list[str]] = []
        self._last_n_raw = 0
        for raw in raw_toks:
            t = _stem(raw)
            if raw in _STOPWORDS or t in _STOPWORDS or (raw.isdigit() and len(raw) > 2) or len(raw) < 2:
                continue
            self._last_n_raw += 1
            syn = [_stem(x) for x in _SYNONYMS.get(raw, [])]      # look up the *unstemmed* local word
            if syn and not (keep_local and len(set(syn)) > 1):
                # local word -> English tokens that must all match; keep the local word as an alternative
                for i, st in enumerate(dict.fromkeys(syn)):
                    groups.append([st, t] if i == 0 and t not in syn else [st])
            else:
                groups.append([t])
        return groups, prep_in_query

    def search(self, query: str, limit: int = 10, source: str | None = None, kind: str | None = None,
               category: str | None = None, min_score: float = 0.34) -> list[dict[str, Any]]:
        """Best matches for a (Taglish) food query.  Multi-word local synonyms ("kanin" -> "rice boiled") are tried
        both expanded and as the raw local word, so FNRI names and packaged-product names are both found."""
        merged: dict[int, float] = {}
        variants = [False]
        if any(len(set(_SYNONYMS.get(t, []))) > 1 for t in _TOKEN_RE.findall(query.lower())):
            variants.append(True)
        for keep_local in variants:
            for di, score in self._search_once(query, source, kind, category, min_score, keep_local):
                if score > merged.get(di, 0.0):
                    merged[di] = score
        ranked = sorted(merged.items(), key=lambda x: (-x[1], self.foods[x[0]]["name"]))[:limit]
        return [self._public(self.foods[di], score) for di, score in ranked]

    def _search_once(self, query: str, source: str | None, kind: str | None, category: str | None,
                     min_score: float, keep_local: bool) -> list[tuple[int, float]]:
        groups, prep_in_query = self._query_groups(query, keep_local)
        n_raw = self._last_n_raw
        if not groups:
            return []
        n_docs = max(1, len(self.foods))
        default_idf = math.log(1.0 + n_docs)
        # each group is weighted by the idf of its first (English) alternative so rare local words don't dominate
        group_idf = [self._idf.get(g[0], default_idf) for g in groups]
        denom = sum(group_idf) or 1.0
        scores: dict[int, list[float]] = {}
        local_hit: set[int] = set()
        for gi, group in enumerate(groups):
            for ai, alt in enumerate(group):
                posting = self._postings.get(alt)
                if not posting:
                    continue
                idf = min(self._idf[alt], group_idf[gi])
                for di, w in posting:
                    arr = scores.get(di)
                    if arr is None:
                        arr = scores[di] = [0.0] * len(groups)
                    val = idf * w * (1.0 if ai == 0 else 0.95)
                    if val > arr[gi]:
                        arr[gi] = val
                        if w in (0.9, 0.95):
                            local_hit.add(di)
                        else:
                            local_hit.discard(di)
        q_norm = " ".join(g[0] for g in groups)
        q_tokens = {a for g in groups for a in g}
        generic_query = n_raw <= 2
        results: list[tuple[float, int]] = []
        for di, arr in scores.items():
            f = self.foods[di]
            if source and f["source"] != source:
                continue
            if kind and f["type"] != kind:
                continue
            if category and category.lower() not in f.get("category", "").lower():
                continue
            matched = sum(1 for v in arr if v > 0)
            if len(groups) >= 2 and matched < max(2, math.ceil(len(groups) * 0.6)):
                continue
            cov = sum(arr) / denom
            doc = self._docs[di]
            eff_len = doc["name_len"]
            if di in local_hit:                       # matched through the Filipino / alias name -> judge by that name's length
                eff_len = min(eff_len, max(doc["local_len"], 2))
            extra_tokens = max(0, eff_len - matched)
            score = cov / (1.0 + 0.09 * extra_tokens)
            if doc["prep"] and not prep_in_query:
                score *= 0.9
            if doc["first"] in q_tokens:
                score *= 1.08
            brand_hit = bool(doc["brand_toks"] & q_tokens)
            if f["source"] == "off":
                if q_norm and q_norm == doc["name_norm"] and brand_hit:
                    score *= 1.1
            elif q_norm and q_norm == doc["name_norm"]:
                score *= 1.08 if doc["us_menu"] else 1.25
            elif q_norm and q_norm in doc["name_norm"]:
                score *= 1.06
            score *= self.SOURCE_WEIGHT.get(f["source"], 1.0)
            if f["type"] in {"product", "restaurant"} and not brand_hit:
                score *= 0.8 if doc["us_menu"] else 0.9         # generic wording prefers generic foods
            if f["source"] == "off":
                filled = sum(1 for k in ("protein_g", "fat_g", "carbs_g", "sodium_mg", "sugars_g") if k in f["per_100g"])
                score *= 0.9 + 0.02 * filled
                if not f.get("brand"):
                    score *= 0.96
            if generic_query and doc["featured"] > 1.0:
                score *= doc["featured"]
            if doc["raw"] and not prep_in_query and f["source"] == "usda":
                score *= 1.14
            if f["type"] == "meal" and not (q_tokens & {"meal", "rice", "kanin", "with", "combo", "overload", "sinangag", "silog", "bigas"}):
                score *= 0.86
            if f.get("boost"):
                score *= float(f["boost"])
            elif f.get("featured"):
                score *= 1.08
            if score >= min_score:
                results.append((score, di))
        return [(di, score) for score, di in results]

    # -- serialization -----------------------------------------------------------------------------
    def _public(self, f: dict[str, Any], score: float | None = None) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "id": f["id"],
            "name": f["name"],
            "local_name": f.get("local_name", ""),
            "brand": f.get("brand", ""),
            "source": f["source"],
            "source_label": SOURCE_LABELS.get(f["source"], f["source"]),
            "type": f["type"],
            "category": f.get("category", ""),
            "quality": f["quality"],
            "quality_label": QUALITY_LABELS.get(f["quality"], f["quality"]),
            "basis": "per_serving" if not f.get("per_100g") else "per_100g",
            "per_100g": f.get("per_100g"),
            "servings": f.get("servings", []),
            "notes": f.get("notes", ""),
            "source_ref": f.get("source_ref", ""),
        }
        if f.get("per_serving"):
            rec["per_serving"] = f["per_serving"]
            rec["serving_label"] = f.get("serving_label", "1 serving")
        if f.get("range"):
            rec["range"] = f["range"]
        if f.get("region"):
            rec["region"] = f["region"]
        elif f.get("extra", {}).get("region"):
            rec["region"] = f["extra"]["region"]
        if f.get("reported"):
            rec["reported"] = f["reported"]
        if f.get("ingredients"):
            rec["ingredients"] = f["ingredients"]
        if f.get("reference_id"):
            rec["reference_id"] = f["reference_id"]
            ref = self.by_id.get(f["reference_id"])
            if ref:
                rec["reference"] = {"id": ref["id"], "name": (ref.get("brand", "") + " " + ref["name"]).strip(),
                                    "region": ref.get("extra", {}).get("region", ""), "serving": self.default_serving(ref)}
        if score is not None:
            rec["score"] = round(min(score, 1.5), 3)
        default = self.default_serving(f)
        if default:
            rec["serving"] = default
        return rec

    def get(self, food_id: str) -> dict[str, Any] | None:
        f = self.by_id.get(food_id)
        return self._public(f) if f else None

    def default_serving(self, f: dict[str, Any]) -> dict[str, Any] | None:
        """Nutrients for the food's default serving (None when only per-100 g without weights)."""
        if f.get("per_serving"):
            return {"label": f.get("serving_label", "1 serving"), "grams": f.get("serving_g"),
                    "nutrients": f["per_serving"]}
        per = f.get("per_100g")
        servings = f.get("servings") or []
        if per and servings:
            s = servings[0]
            if abs(s["grams"] - 100.0) < 1e-6:
                return None
            return {"label": s["label"], "grams": s["grams"], "nutrients": _scale(per, s["grams"] / 100.0)}
        return None

    def portion(self, f: dict[str, Any], grams: float | None = None, count: float = 1.0) -> tuple[dict[str, float], str, float | None]:
        """Return (nutrients, label, grams) for ``grams`` of a food or ``count`` default servings."""
        if grams is not None and f.get("per_100g"):
            return _scale(f["per_100g"], grams / 100.0), f"{grams:g} g", grams
        if f.get("per_serving"):
            lab = f.get("serving_label", "serving")
            return _scale(f["per_serving"], count), (lab if count == 1 else f"{count:g} x {lab}"), (f.get("serving_g") or 0) * count or None
        per = f.get("per_100g") or {}
        servings = f.get("servings") or [{"label": "100 g", "grams": 100.0}]
        s = servings[0]
        factor = s["grams"] * count / 100.0
        label = s["label"] if count == 1 else f"{count:g} x {s['label']}"
        return _scale(per, factor), label, s["grams"] * count

    # -- stats -------------------------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        by_source = Counter(f["source"] for f in self.foods)
        by_type = Counter(f["type"] for f in self.foods)
        cats = Counter(f.get("category", "") for f in self.foods if f["source"] in {"philfct", "dish", "restaurant", "off"})
        return {
            "total_foods": len(self.foods),
            "by_source": dict(by_source),
            "by_type": dict(by_type),
            "philfct_foods": by_source.get("philfct", 0),
            "usda_foods": by_source.get("usda", 0),
            "open_food_facts_products": by_source.get("off", 0),
            "dishes": sum(1 for f in self.foods if f["source"] == "dish" and f["type"] == "dish"),
            "meals": sum(1 for f in self.foods if f["type"] == "meal"),
            "restaurant_items": by_source.get("restaurant", 0),            # McDonald's PH + Jollibee (curated)
            "us_chain_items_usda": sum(1 for f in self.foods if f["type"] == "restaurant" and f["source"] == "usda"),
            "top_categories": cats.most_common(14),
            "source_labels": SOURCE_LABELS,
            "quality_labels": QUALITY_LABELS,
        }

    # -- formatting --------------------------------------------------------------------------------
    @staticmethod
    def _n(v: Any) -> str:
        """Compact number: 213, 18.4, 0.3 (no 18.44 / 3.87 noise)."""
        if v is None:
            return "?"
        v = float(v)
        if abs(v) >= 100:
            return f"{round(v):d}"
        if abs(v) >= 10:
            return f"{round(v, 1):g}"
        return f"{round(v, 1):g}"

    @classmethod
    def _fmt_line(cls, nutr: dict[str, float], focus: str | None = None) -> str:
        parts = []
        if "kcal" in nutr:
            parts.append(f"**{cls._n(nutr['kcal'])} kcal**")
        short = [("protein_g", "Protein", "g"), ("fat_g", "Fat", "g"), ("carbs_g", "Carbs", "g"),
                 ("fiber_g", "Fiber", "g"), ("sugars_g", "Sugar", "g"), ("sodium_mg", "Sodium", "mg")]
        for key, label, unit in short:
            if key in nutr and nutr[key] is not None:
                txt = f"{label} {cls._n(nutr[key])} {unit}"
                parts.append(f"**{txt}**" if key == focus else txt)
        if focus and focus not in {k for k, *_ in short} and focus in nutr:
            label = next(l for k, l, _ in NUTRIENT_FIELDS if k == focus)
            unit = next(u for k, _, u in NUTRIENT_FIELDS if k == focus)
            parts.append(f"**{label} {cls._n(nutr[focus])} {unit}**")
        return " | ".join(parts)

    def format_card(self, rec: dict[str, Any], focus: str | None = None, detailed: bool = True) -> str:
        title = f"**{rec['name']}**"
        if rec.get("brand") and rec["brand"].lower() not in rec["name"].lower():
            title = f"**{rec['brand']} {rec['name']}**" if rec["type"] in {"product", "restaurant"} else title
        local = rec.get("local_name") or ""
        if local and local.lower() not in rec["name"].lower() and rec["name"].lower() not in local.lower():
            title += f" ({local})"
        tags = [rec["source_label"]]
        if rec.get("region"):
            tags.append(f"{rec['region']} menu")
        lines = [f"{title} - _{' | '.join(tags)}_"]
        if rec.get("per_100g"):
            lines.append(f"- Per 100 g: {self._fmt_line(rec['per_100g'], focus)}")
        serv = rec.get("serving")
        if serv:
            grams = f" (~{self._n(serv['grams'])} g)" if serv.get("grams") and serv["label"].find("g)") < 0 and serv["label"].find(" g") < 0 else ""
            lines.append(f"- Per {serv['label']}{grams}: {self._fmt_line(serv['nutrients'], focus)}")
        if detailed:
            rng = rec.get("range")
            if rng:
                lines.append(f"- Typical range: {self._n(rng['kcal_low'])}-{self._n(rng['kcal_high'])} kcal per 100 g ({rng.get('basis', 'depends on cut / recipe')})")
            rep = rec.get("reported")
            if rep:
                lines.append(f"- Reported values: {self._n(rep['kcal_min'])}-{self._n(rep['kcal_max'])} kcal across sources ({rep.get('note', 'unverified')})")
            if rec.get("ingredients"):
                lines.append("- Based on: " + ", ".join(rec["ingredients"][:7]) + (" ..." if len(rec["ingredients"]) > 7 else ""))
            ref = rec.get("reference")
            if ref and ref.get("serving") and "protein_g" not in (rec.get("serving") or {}).get("nutrients", {}):
                lines.append(f"- Macros (US {ref['name']} per {ref['serving']['label']}, USDA): "
                             f"{self._fmt_line({k: v for k, v in ref['serving']['nutrients'].items() if k in {'kcal', 'protein_g', 'fat_g', 'carbs_g'}}, focus)}")
            lines.append(f"- _Data quality: {rec['quality_label']}._")
            if rec.get("notes") and rec["type"] in {"dish", "meal", "restaurant"}:
                lines.append(f"- _{rec['notes']}_")
        return "\n".join(lines)

    # -- natural language --------------------------------------------------------------------------
    _SPLIT_RE = re.compile(r"\s*(?:,|\+|&|;|\band\b|\bat\b|\bsaka\b|\bplus\b|\bwith\b|\bw/\b|\bwith a\b)\s*", re.I)
    _QTY_RE = re.compile(
        r"^\s*(?:(?P<qty>\d+(?:\.\d+)?|\d+\s*/\s*\d+|an?|one|two|three|four|five|six|half|isa|dalawa|tatlo)\s*(?:x\s*)?)?"
        r"(?:(?P<unit>pcs?|pieces?|piece|pc|cups?|servings?|serving|g|grams?|gm|kg|ml|tbsp|tsp|slices?|bowls?|plates?|cans?|packs?|glass(?:es)?|orders?|bottles?)\b\.?\s*)?"
        r"(?:of\s+)?(?P<name>.+?)\s*$", re.I)
    _WORD_NUM = {"a": 1, "an": 1, "one": 1, "isa": 1, "two": 2, "dalawa": 2, "three": 3, "tatlo": 3, "four": 4, "five": 5, "six": 6, "half": 0.5}

    def parse_meal_text(self, text: str) -> list[dict[str, Any]]:
        # NB: keep "meal" / "breakfast" - they are part of food names ("Pares meal", "Big Breakfast")
        cleaned = re.sub(r"\b(calories?|kcal|nutrition|facts?|in|of my|ng|sa|ilang|how many|how much|what('?s| is)|total|lunch|dinner|"
                         r"almusal|tanghalian|hapunan|kumain ako ng|i ate|i had|ate|had|kain|ng aking)\b", " ", text, flags=re.I)
        cleaned = re.sub(r"[?!.]+$", "", cleaned.strip())
        cleaned = re.sub(r"^\s*(of|ng|sa|ang|a|the)\s+", "", cleaned, flags=re.I)
        items: list[dict[str, Any]] = []
        for part in self._SPLIT_RE.split(cleaned):
            part = part.strip()
            if not part:
                continue
            m = self._QTY_RE.match(part)
            if not m or not m.group("name"):
                continue
            q_raw = (m.group("qty") or "").strip().lower()
            if not q_raw:
                qty = 1.0
            elif "/" in q_raw:
                a, b = [float(x) for x in q_raw.split("/")]
                qty = a / b if b else 1.0
            elif q_raw in self._WORD_NUM:
                qty = float(self._WORD_NUM[q_raw])
            else:
                qty = float(q_raw)
            unit = (m.group("unit") or "").lower()
            grams = None
            if unit in {"g", "gram", "grams", "gm"}:
                grams, qty = qty, 1.0
            elif unit == "kg":
                grams, qty = qty * 1000, 1.0
            name = re.sub(r"^(of|ng)\s+", "", m.group("name").strip(), flags=re.I)
            name = self._MEAL_ALIASES.get(name.lower(), name)
            items.append({"text": part, "name": name, "qty": qty, "unit": unit, "grams": grams})
        return items

    _MEAL_ALIASES = {
        "rice": "rice well-milled boiled", "steamed rice": "rice well-milled boiled", "plain rice": "rice well-milled boiled",
        "white rice": "rice well-milled boiled", "kanin": "rice well-milled boiled", "extra rice": "rice well-milled boiled",
        "garlic rice": "rice well-milled fried", "sinangag": "rice well-milled fried", "fried rice": "rice well-milled fried",
        "egg": "egg chicken whole", "itlog": "egg chicken whole", "fried egg": "egg chicken whole",
        "coke": "cola", "softdrink": "cola", "soft drink": "cola",
    }
    _UNIT_WORD = {"cups": "cup", "cup": "cup", "tbsp": "tbsp", "tsp": "tsp", "slices": "slice", "slice": "slice",
                  "pieces": "piece", "piece": "piece", "pcs": "piece", "pc": "piece", "glasses": "glass", "glass": "glass",
                  "packs": "pack", "pack": "pack", "bowls": "bowl", "bowl": "bowl", "plates": "plate", "plate": "plate",
                  "cans": "can", "can": "can"}

    def estimate_meal(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """items: [{name | id, qty?, unit?, grams?}] -> per-item breakdown + totals (+ which nutrients every item had)."""
        lines, unmatched = [], []
        totals: dict[str, float] = {}
        have: Counter[str] = Counter()
        for it in items:
            rec = None
            if it.get("id"):
                rec = self.by_id.get(it["id"])
            if rec is None and it.get("name"):
                wanted = self._MEAL_ALIASES.get(str(it["name"]).strip().lower(), str(it["name"]))  # "rice" -> cooked rice
                found = self.search(wanted, limit=1, min_score=0.3)
                rec = self.by_id.get(found[0]["id"]) if found else None
            if rec is None:
                unmatched.append(it.get("name") or it.get("id"))
                continue
            count = float(it.get("qty") or 1.0)
            grams = it.get("grams")
            word = self._UNIT_WORD.get((it.get("unit") or "").lower())
            chosen = None
            if grams is None and word and rec.get("per_100g"):
                chosen = next((sv for sv in rec.get("servings", []) if word in sv["label"].lower() and sv.get("grams")), None)
            if chosen:
                g = chosen["grams"] * count
                nutr = _scale(rec["per_100g"], g / 100.0)
                label, grams = (chosen["label"] if count == 1 else f"{count:g} x {chosen['label']}"), g
            else:
                nutr, label, grams = self.portion(rec, grams=grams, count=count)
            for k, v in nutr.items():
                totals[k] = totals.get(k, 0.0) + (v or 0.0)
                have[k] += 1
            lines.append({"id": rec["id"], "name": rec["name"], "brand": rec.get("brand", ""), "portion": label,
                          "grams": grams, "source": rec["source"], "quality": rec["quality"], "nutrients": nutr})
        n = len(lines)
        return {"items": lines, "unmatched": unmatched, "totals": {k: _round(v, 1) for k, v in totals.items()},
                "complete": sorted(k for k, c in have.items() if c == n and n > 0)}

    def answer_meal(self, message: str, focus: str | None = None) -> str | None:
        parsed = self.parse_meal_text(message)
        if len(parsed) < 2:
            return None
        result = self.estimate_meal([{"name": p["name"], "qty": p["qty"], "unit": p["unit"], "grams": p["grams"]} for p in parsed])
        if len(result["items"]) < 2:
            return None
        out = [f"**Meal estimate** ({len(result['items'])} items):"]
        for ln in result["items"]:
            n = ln["nutrients"]
            nm = f"{ln['brand']} {ln['name']}".strip() if ln.get("brand") and ln["brand"].lower() not in ln["name"].lower() else ln["name"]
            bits = [f"{self._n(n.get('kcal', 0))} kcal"]
            for key, lab in (("protein_g", "P"), ("fat_g", "F"), ("carbs_g", "C")):
                if key in n:
                    bits.append(f"{lab} {self._n(n[key])} g")
            out.append(f"- **{nm}** ({ln['portion']}) - " + " | ".join(bits))
        t, complete = result["totals"], set(result["complete"])
        parts = [f"**Total: {self._n(t.get('kcal', 0))} kcal**"]
        partial = False
        for key, lab, unit in (("protein_g", "Protein", "g"), ("fat_g", "Fat", "g"), ("carbs_g", "Carbs", "g"), ("sodium_mg", "Sodium", "mg")):
            if key in t:
                if key in complete:
                    parts.append(f"{lab} {self._n(t[key])} {unit}")
                else:
                    parts.append(f"{lab} at least {self._n(t[key])} {unit}")
                    partial = True
        out.append("\n" + " | ".join(parts))
        if partial:
            out.append("_Some items only list calories (restaurant values), so the other totals are lower bounds._")
        if result["unmatched"]:
            out.append(f"_Couldn't find: {', '.join(str(u) for u in result['unmatched'])}._")
        out.append("_Estimates only - portions and recipes vary._")
        return "\n".join(out)

    def answer_compare(self, message: str) -> str | None:
        stripped = re.sub(r"^(compare|which has more \w+)[:,]?\s*", "", message.strip(), flags=re.I)
        seps = r"vs\.?|versus|or|kumpara sa|compared to|o" + (r"|and|at" if re.match(r"\s*compare\b", message, re.I) else "")
        m = re.split(rf"\s+(?:{seps})\s+", stripped, maxsplit=1)
        if len(m) != 2:
            return None
        recs = []
        for part in m:
            found = self.search(re.sub(r"\b(calories?|nutrition|in|of|ang|ba)\b", " ", part, flags=re.I), limit=1, min_score=0.3)
            if not found:
                return None
            recs.append(found[0])
        a, b = recs
        def row(rec):
            per = rec.get("per_100g") or (rec.get("serving") or {}).get("nutrients") or rec.get("per_serving") or {}
            basis = "100 g" if rec.get("per_100g") else rec.get("serving_label", "serving")
            return per, basis
        pa, ba = row(a)
        pb, bb = row(b)
        out = [f"**{a['name']}** (per {ba}) vs **{b['name']}** (per {bb}):"]
        for key, label, unit in NUTRIENT_FIELDS[:8]:
            if key in pa or key in pb:
                out.append(f"- {label}: {pa.get(key, '-')} {unit} vs {pb.get(key, '-')} {unit}")
        return "\n".join(out)

    # -- topic overviews ("canned foods", "jollibee", "street food" ...) ------------------------------
    _TOPIC_FILLER = re.compile(r"\b(what|whats|are|is|the|calories?|kcal|nutrition(al)?|facts?|info|of|in|for|show|list|me|give|ng|sa|ang|mga|ba|"
                               r"ilan|ilang|tell|about|all|menu|items?|please|pls|top|popular|common)\b|[?!.,']", re.I)
    _TOPICS = [
        ("canned", re.compile(r"^(canned( (foods?|goods?|products?))?|de lata|lata|canned goods)$")),
        ("mcdo", re.compile(r"^(mcdo|mcdonalds?|mcdonald s|mcdo menu)$")),
        ("jollibee", re.compile(r"^(jollibee|jabee|jollibee menu)$")),
        ("fastfood", re.compile(r"^(fast ?foods?|fastfood)$")),
        ("dishes", re.compile(r"^(filipino|pinoy|philippine)( (dishes|foods?|ulam|viands?|recipes?))?$|^(ulam|viands?|carinderia( foods?)?|pinoy ulam)$")),
        ("street", re.compile(r"^(street ?foods?|merienda|pinoy snacks?|filipino snacks?|pinoy street ?foods?)$")),
        ("milk", re.compile(r"^(milk|gatas|milk products?|dairy)$")),
    ]
    _TOPIC_TITLES = {
        "canned": "Canned foods (FNRI lab values; ask for a brand such as `Century Tuna` or `Argentina corned beef`)",
        "mcdo": "McDonald's Philippines (official 2010 flyer - kcal per item)",
        "jollibee": "Jollibee (reported values - US guide and PH menu sites)",
        "fastfood": "Fast food in the Philippines",
        "dishes": "Popular Filipino dishes (recipe estimates)",
        "street": "Street food & merienda",
        "milk": "Milk & dairy",
    }
    _TOPIC_QUERIES = {
        "canned": ["sardines in tomato sauce canned", "corned beef canned", "luncheon meat canned", "meat loaf", "vienna sausage",
                   "milkfish canned", "tuna spread canned", "pork adobo canned", "beef caldereta canned"],
        "street": ["kwek-kwek", "taho", "fish ball fried", "isaw pork barbecue", "balut", "turon banana wrapped fried",
                   "banana saba sugar fried", "lumpiang shanghai", "puto white", "bibingka"],
        "milk": ["milk cow", "milk evaporated", "milk sweetened condensed", "milk powder full cream", "milk carabao", "yoghurt", "bear brand milk", "alaska milk"],
    }

    def _topic_items(self, topic: str) -> list[dict[str, Any]]:
        if topic in {"mcdo", "jollibee", "fastfood"}:
            wanted = {"mcdo": ["McDonald's"], "jollibee": ["Jollibee"], "fastfood": ["McDonald's", "Jollibee"]}[topic]
            items = [f for f in self.foods if f["source"] == "restaurant" and f.get("brand") in wanted and f.get("featured")
                     and f.get("region", "") in {"PH", ""}] + \
                    [f for f in self.foods if f["source"] == "restaurant" and f.get("brand") == "Jollibee" and f.get("featured") and f.get("region") == "US"
                     and topic != "mcdo"]
            seen, out = set(), []
            for f in items:
                if f["name"] not in seen:
                    seen.add(f["name"])
                    out.append(self._public(f))
            return out
        if topic == "dishes":
            return [self._public(f) for f in self.foods if f["source"] == "dish" and f["type"] == "dish" and f.get("featured")]
        out = []
        for q in self._TOPIC_QUERIES.get(topic, []):
            hit = self.search(q, limit=1, min_score=0.3)
            if hit and hit[0]["id"] not in {o["id"] for o in out}:
                out.append(hit[0])
        return out

    def answer_topic(self, message: str) -> str | None:
        cleaned = re.sub(r"\s+", " ", self._TOPIC_FILLER.sub(" ", message.lower())).strip()
        for topic, rx in self._TOPICS:
            if rx.match(cleaned):
                items = self._topic_items(topic)
                if not items:
                    return None
                out = [f"**{self._TOPIC_TITLES[topic]}**"]
                for r in items[:12]:
                    brand = f"{r['brand']} " if r.get("brand") and r["brand"].lower() not in r["name"].lower() else ""
                    if r.get("per_100g"):
                        base = f"{self._n(r['per_100g'].get('kcal'))} kcal/100 g"
                        serv = r.get("serving")
                        if serv:
                            base += f" | {self._n(serv['nutrients'].get('kcal'))} kcal per {serv['label']}"
                    else:
                        ps = r.get("per_serving") or {}
                        base = f"{self._n(ps.get('kcal'))} kcal per {r.get('serving_label', 'serving')}"
                    out.append(f"- {brand}{r['name']} - {base}")
                if topic == "canned":
                    n = sum(1 for f in self.foods if f["source"] == "off" and f.get("category") == "Canned Goods")
                    out.append(f"\n_Plus {n} branded canned products from Open Food Facts (Century Tuna, Ligo, 555, Argentina, Purefoods, CDO ...)._")
                out.append("\n_Ask about any item for its full nutrition facts._")
                return "\n".join(out)
        return None

    def answer_nutrition_query(self, message: str, limit_alternatives: int = 4) -> str:
        focus = self.nutrient_focus(message)
        low = message.lower()
        topic = self.answer_topic(message)
        if topic:
            return topic
        if re.search(r"\b(vs\.?|versus|compare|kumpara)\b", low):
            cmp = self.answer_compare(message)
            if cmp:
                return cmp
        meal = self.answer_meal(message, focus)
        if meal:
            return meal
        results = self.search(message, limit=12)
        if not results:
            return ("I couldn't find that food in my Philippine nutrition database yet. Try a simpler name, e.g. "
                    "`calories in garlic`, `nutrition facts of pork adobo`, `Jollibee Chickenjoy calories`, "
                    "`McDo Big Mac nutrition`, `ground beef protein` or a canned item like `Century tuna`.")
        best = results[0]
        out = [self.format_card(best, focus)]
        # tied / similar alternatives, preferring variety of sources
        alts, seen = [], {best["name"].lower()}
        for r in results[1:]:
            if r["name"].lower() in seen or r["score"] < 0.55 * best["score"]:
                continue
            seen.add(r["name"].lower())
            alts.append(r)
        picked: list[dict[str, Any]] = []
        used_sources: set[str] = {best["source"]}
        for r in alts:  # first pass: new sources
            if r["source"] not in used_sources and len(picked) < limit_alternatives:
                picked.append(r)
                used_sources.add(r["source"])
        for r in alts:  # second pass: fill
            if r not in picked and len(picked) < limit_alternatives:
                picked.append(r)
        if picked:
            out.append("\n**Similar matches:**")
            for r in picked:
                if r.get("per_100g"):
                    base = f"{self._n(r['per_100g'].get('kcal'))} kcal/100 g"
                else:
                    ps = r.get("per_serving") or {}
                    base = f"{self._n(ps.get('kcal'))} kcal per {r.get('serving_label', 'serving')}"
                brand = f"{r['brand']} " if r.get("brand") and r["brand"].lower() not in r["name"].lower() else ""
                extra = ""
                if focus and r.get("per_100g") and focus in r["per_100g"]:
                    unit = next(u for k, _, u in NUTRIENT_FIELDS if k == focus)
                    extra = f", {focus.split('_')[0]} {self._n(r['per_100g'][focus])} {unit}"
                out.append(f"- {brand}{r['name']} - {base}{extra} _({r['source_label'].split(' (')[0]})_")
        out.append("\n_Tip: ask for a whole meal, e.g. `calories of 2 pc Chickenjoy and 1 cup rice`._")
        return "\n".join(out)


_SHARED: PHNutritionExpert | None = None


def get_shared_expert() -> PHNutritionExpert:
    """Process-wide lazily-built expert (loading + indexing ~20k foods takes a few seconds)."""
    global _SHARED
    if _SHARED is None:
        _SHARED = PHNutritionExpert()
    return _SHARED
