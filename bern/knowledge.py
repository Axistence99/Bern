"""Check a learned claim against data Bern already holds, before it is stored.

Learning from whatever arrives is how a model quietly gets worse. Bern's price and nutrition
tables are small and checked against named sources; a fact typed into the chat box is neither.
Saving both into the same place means a mistyped price outranks a real one, because the latest
thing taught is the thing that answers.

So a claim is checked first, and there are only three verdicts:

* ``verified``   - the reference data agrees. Safe to store.
* ``disputed``  - the reference data holds a different value. Stored only if a human says so,
                   and the disagreement is reported either way.
* ``unverified`` - nothing in Bern's data covers this. Stored, but flagged as unverified and
                   never presented as fact.

What this is not: a general fact-checker. There is no truth service here. Bern can only check
the two kinds of number it actually holds - Philippine prices and nutrition figures - and
everything else is *unverified*, not *true*. Saying otherwise would be the exact failure this
module exists to prevent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
PRICE_PATH = BASE_DIR / "data" / "ph_prices.json"
NUTRITION_PATH = BASE_DIR / "data" / "ph_nutrition_curated.json"

#: Numbers in a claim, with an optional unit. "rice is 65 per kilo", "450 kcal", "₱120".
_NUMBER = re.compile(r"(?:₱|php\s*)?(\d+(?:\.\d+)?)\s*(kcal|calories|cal|per kilo|/kg|per kg|kg|g|grams)?",
                     re.IGNORECASE)


@dataclass
class Verdict:
    """The result of checking one claim."""

    status: str          # verified | disputed | unverified
    checked_against: str = ""
    detail: str = ""
    reference_value: str = ""

    @property
    def blocks_saving(self) -> bool:
        return self.status == "disputed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checked_against": self.checked_against,
            "detail": self.detail,
            "reference_value": self.reference_value,
        }


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _numbers(text: str) -> list[float]:
    return [float(m.group(1)) for m in _NUMBER.finditer(text)]


#: Words that must not be treated as an item name on their own. "chicken joy is 300 kcal" was
#: matching the raw-chicken price entry on the single word "chicken" and comparing a calorie
#: figure against a peso figure - a false dispute built from a false match.
_TOO_GENERIC = frozenset({
    "rice", "egg", "meat", "chicken", "pork", "beef", "fish", "sugar", "salt", "oil", "milk",
    "water", "flour", "onion", "garlic", "coffee", "bread", "soap", "coke", "cola",
})


def _match_items(text: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the reference entries whose name or alias appears in the claim.

    Matches must be at least two words, or a single word that is specific enough to name one
    thing ("adobo", "bigas"). Comparing a nutrition figure against a price entry produces a
    confident, wrong verdict, which is worse than no verdict.
    """
    lowered = text.lower()
    hits: list[tuple[int, dict[str, Any]]] = []
    for item in items:
        names = [str(item.get("name", ""))] + [str(a) for a in item.get("aliases", [])]
        for name in names:
            needle = name.lower().strip()
            if not needle:
                continue
            if needle in _TOO_GENERIC and not _unit_confirms(needle, lowered):
                continue
            if len(needle) < 4 and " " not in needle:
                continue
            if needle in lowered:
                hits.append((len(needle), item))
                break
    # Longest match wins: "chicken joy" beats "chicken", "pork adobo" beats "pork".
    hits.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in hits]


#: Units that identify which table a claim is about. A peso-per-kilo claim cannot be about
#: Chickenjoy; a kcal claim cannot be about raw chicken.
_WEIGHT_UNITS = ("per kilo", "/kg", "per kg", "a kilo", "per kilogram")
_ENERGY_UNITS = ("kcal", "calorie", "calories", "cal")


def _unit_confirms(name: str, lowered: str) -> bool:
    """Whether the claim's unit makes a generic name safe to match."""
    has_weight = any(u in lowered for u in _WEIGHT_UNITS)
    has_energy = any(u in lowered for u in _ENERGY_UNITS)
    # Weight-priced staples read naturally as "per kilo"; calories read as kcal.
    return has_weight and not has_energy


def check_claim(text: str) -> Verdict:
    """Check one claim against Bern's price and nutrition reference data.

    Only numbers are checked. A claim with no number in it cannot be checked against these
    tables at all, and is reported ``unverified`` rather than passed as correct.
    """
    statement = (text or "").strip()
    if not statement:
        return Verdict("unverified", detail="Nothing to check.")

    claimed = _numbers(statement)
    if not claimed:
        return Verdict("unverified", detail="No figure in this to compare against Bern's data.")

    value = claimed[0]

    prices = _load(PRICE_PATH).get("items", [])
    price_hits = _match_items(statement, prices)
    if price_hits:
        item = price_hits[0]
        lo = item.get("palengke_min")
        hi = item.get("palengke_max")
        sup_lo, sup_hi = item.get("supermarket_min"), item.get("supermarket_max")
        spent = "palengke" if "supermarket" not in statement.lower() else "supermarket"
        rng_lo, rng_hi = (sup_lo, sup_hi) if spent == "supermarket" else (lo, hi)
        if rng_lo is None or rng_hi is None:
            return Verdict("unverified", "Philippine prices", f"No price range stored for {item.get('name')}.")
        low, high = float(rng_lo), float(rng_hi)
        span = f"{spent} ₱{low:g}–{high:g}"
        if low * 0.85 <= value <= high * 1.15:
            return Verdict("verified", "Philippine prices", f"{item.get('name')} is {span}.", span)
        return Verdict(
            "disputed", "Philippine prices",
            f"You said ₱{value:g}, but Bern's {item.get('name')} is {span} "
            f"({_load(PRICE_PATH).get('last_updated', 'undated')}).",
            span,
        )

    foods = _load(NUTRITION_PATH).get("foods", [])
    food_hits = _match_items(statement, foods)
    if food_hits:
        food = food_hits[0]
        per_100 = food.get("per_100g") or {}
        kcal = per_100.get("calories") or per_100.get("kcal")
        if kcal is None:
            return Verdict("unverified", "Nutrition", f"No calorie figure stored for {food.get('name')}.")
        if value > 1500:
            return Verdict("unverified", "Nutrition", "Most figures over 1500 are not kcal per 100 g.")
        low, high = float(kcal) * 0.8, float(kcal) * 1.25
        ref = f"{float(kcal):g} kcal per 100 g"
        if low <= value <= high:
            return Verdict("verified", "Nutrition", f"{food.get('name')} is {ref}.", ref)
        return Verdict(
            "disputed", "Nutrition",
            f"You said {value:g}, but Bern's {food.get('name')} is {ref}.", ref,
        )

    return Verdict("unverified", detail="Bern holds no price or nutrition entry for this.")
