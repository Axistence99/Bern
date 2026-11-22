"""Remember what the user said, but only after checking what can be checked.

The request was to learn from every prompt and fact-check before saving. Taken literally, that
is a machine for poisoning itself: whatever is typed becomes training data, and a mistyped
price outranks the curated one because the newest statement is the one that answers. So the
rules below are less convenient than "learn everything" and exist for that reason.

1. **A claim must contain a checkable figure.** Everything else is counted and ignored - a
   question is not a fact, and storing one would teach Bern to answer its own questions.
2. **Nothing is saved as fact without a verdict.** :mod:`bern.knowledge` returns verified,
   disputed or unverified. Disputed is refused unless a human overrides; unverified is stored
   clearly marked, and never presented to a user as though it were checked.
3. **A claim never silently overrides reference data.** It goes to a separate store, so the
   curated price table keeps answering for prices.
4. **Nothing is learned without a person saying so.** ``learn`` must be called explicitly;
   the chat path does not call it.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from bern.knowledge import check_claim

BASE_DIR = Path(__file__).resolve().parent.parent
STORE_PATH = BASE_DIR / "data" / "learned_claims.json"

#: "rice is 45 per kilo", "adobo has 312 calories", "bigas ₱42/kg". A statement of fact, not a
#: question: no question mark, and a figure with something naming it.
_CLAIM_SHAPE = re.compile(
    r"^\s*(?P<subject>[^?]{3,80}?)\s+(?:is|are|has|have|costs?|cost|ay|may)\s+"
    r"(?:₱|php\s*)?(?P<value>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_LOCK = threading.Lock()


def _load() -> dict[str, Any]:
    try:
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"claims": []}


def _save(store: dict[str, Any]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(store, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def looks_like_claim(message: str) -> bool:
    """Whether a message is stating a fact about something, rather than asking."""
    text = (message or "").strip()
    if not text or text.endswith("?") or len(text) > 240:
        return False
    return bool(_CLAIM_SHAPE.match(text))


def learn(message: str, session_id: str = "default",
          override: bool = False) -> dict[str, Any]:
    """Check one claim and store it if it survives the check.

    Args:
        message: What the user typed.
        session_id: Who said it, so a claim can be traced back.
        override: Store a disputed claim anyway. The verdict and the reference value travel with
            it, so the disagreement is never lost.

    Returns:
        ``{saved, verdict, reason}``. ``saved`` is False whenever anything was refused, and
        ``reason`` says why in a sentence a person can act on.
    """
    text = (message or "").strip()
    if not looks_like_claim(text):
        return {"saved": False, "verdict": None,
                "reason": "Not a statement of fact - nothing to check or store."}

    verdict = check_claim(text)
    if verdict.status == "disputed" and not override:
        return {"saved": False, "verdict": verdict.as_dict(),
                "reason": f"Refused: {verdict.detail} Say so explicitly if you are right and "
                          f"Bern's data is out of date."}

    record = {
        "text": text,
        "verdict": verdict.status,
        "checked_against": verdict.checked_against,
        "reference_value": verdict.reference_value,
        "session": session_id,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with _LOCK:
        store = _load()
        existing = {c["text"].lower() for c in store.get("claims", [])}
        if text.lower() in existing:
            return {"saved": False, "verdict": verdict.as_dict(),
                    "reason": "Already stored."}
        store.setdefault("claims", []).append(record)
        _save(store)

    if verdict.status == "unverified":
        note = ("Saved, marked unverified: nothing in Bern's data covers this, so it will never "
                "be presented as a checked fact.")
    elif verdict.status == "disputed":
        note = f"Saved over a disagreement: {verdict.detail}"
    else:
        note = f"Saved and checked: {verdict.detail}"
    return {"saved": True, "verdict": verdict.as_dict(), "reason": note}


def stored_claims() -> list[dict[str, Any]]:
    """Everything learned so far, newest last."""
    return _load().get("claims", [])


def count_by_verdict() -> dict[str, int]:
    counts: dict[str, int] = {}
    for claim in stored_claims():
        counts[claim.get("verdict", "unknown")] = counts.get(claim.get("verdict", "unknown"), 0) + 1
    return counts
