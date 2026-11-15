"""Conversation memory: what Bern remembers between turns, without a language model.

A chat box that forgets after every message is not a chat box. Bern's intent classifier sees
one message at a time and has no idea that "what about the other one?" refers to anything, so
this module supplies the missing context with two small, testable pieces:

* **A transcript.** The last few turns, kept per session and capped so a long conversation
  cannot grow without bound.
* **Reference resolution.** "What about the other one?", "iyon din", "same for that" are
  rewritten into the full question they stand for, using the entities the previous answer was
  actually about.

The resolution is rule-based on purpose. A pronoun resolver that guesses would be worse than
one that declines: when this module cannot tell what a reference means it returns ``None`` and
the caller answers the message as written, which is the honest outcome.

The entities are recorded by the caller from the answer it just produced, never parsed back
out of prose. Guessed entities are how a follow-up ends up about the wrong food.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Turns kept per session. Beyond this the oldest are dropped: the point of the transcript is
#: resolving an immediate follow-up, and no follow-up refers back forty messages.
MAX_TURNS = 12

#: Discussed items kept for "the other one". A handful is plenty; more is just noise.
MAX_ENTITIES = 6

#: Phrases that point at something already mentioned. Grouped by what they resolve to.
_OTHER = re.compile(
    r"\b(?:the\s+other\s+one|the\s+other|other\s+one|the\s+rest|"
    r"yung\s+isa|yong\s+isa|iyong\s+isa|ang\s+isa|isa\s+pa)\b",
    re.IGNORECASE,
)
_SAME = re.compile(
    r"\b(?:same|same\s+for|same\s+with|ulit|ulitin|pareho|parehas|"
    r"how\s+about\s+(?:that|it|this)|what\s+about\s+(?:that|it|this)|"
    r"and\s+(?:that|it)|at\s+yon)\b",
    re.IGNORECASE,
)
_PRONOUN = re.compile(r"\b(?:it|that|this|those|these|iyon|iyan|ito|yon|yan)\b", re.IGNORECASE)

#: A message longer than this is a new question, not a follow-up gesture.
_MAX_FOLLOW_UP_WORDS = 8


@dataclass
class Memory:
    """Per-session transcript plus the last entities a reply was about."""

    turns: list[dict[str, Any]] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    last_entity: str | None = None
    #: Intent of the last answered turn, used to rebuild a follow-up in the shape the
    #: classifier understands.
    last_intent: str = ""

    #: Subject the previous follow-up resolved to, so "and that one?" does not simply repeat
    #: the same answer twice in a row.
    last_resolved: str | None = None

    def record(self, message: str, intent: str, reply: str,
               entities: list[str] | None = None) -> None:
        """Append one exchange. ``entities`` are the items the reply was actually about."""
        clean = [str(item).strip() for item in (entities or []) if str(item).strip()]
        self.turns.append({
            "message": message,
            "intent": intent,
            "entities": clean,
        })
        del self.turns[:-MAX_TURNS]

        if intent:
            self.last_intent = intent

        # Entities accumulate as a rolling list rather than being replaced, because "the other
        # one" means the item discussed *before* this one. Replacing the set would leave
        # nothing to refer back to after a single-item question.
        for item in clean:
            if item in self.entities:
                self.entities.remove(item)
            self.entities.append(item)
        del self.entities[:-MAX_ENTITIES]
        if self.entities:
            self.last_entity = self.entities[-1]

    def resolve(self, message: str) -> tuple[str, str | None]:
        """Rewrite a follow-up into the question it stands for.

        Returns:
            ``(rewritten_message, subject)``. When nothing can be resolved the message is
            returned unchanged and ``subject`` is ``None``, so the caller can answer it
            normally instead of being handed a guess.
        """
        text = message.strip()
        if not text or not self.entities:
            return message, None

        words = text.split()
        if len(words) > _MAX_FOLLOW_UP_WORDS:
            # Long enough to be its own question; a pronoun inside it is not a reference.
            return message, None

        if _OTHER.search(text):
            # "The other one" is the most recently discussed item that is not the current
            # subject. With only one item ever discussed there is no other, and guessing
            # would send the follow-up to the same subject while sounding like it moved on.
            other = next((item for item in reversed(self.entities) if item != self.last_entity), None)
            if other is None:
                return message, None
            self.last_entity = other
            self.last_resolved = other
            return self._substitute(text, other), other

        if _SAME.search(text) or (_PRONOUN.search(text) and len(words) <= 4):
            subject = self.last_entity or self.entities[-1]
            # Asking "and that one?" straight after a follow-up that already answered about
            # this same subject means the other item, not the same answer a second time.
            if subject == self.last_resolved:
                other = next((item for item in reversed(self.entities) if item != subject), None)
                if other is not None:
                    subject = other
            self.last_entity = subject
            self.last_resolved = subject
            return self._substitute(text, subject), subject

        return message, None

    def _substitute(self, text: str, subject: str) -> str:
        """Keep the question, replace the pointer with the subject.

        The rewritten form is what the classifier sees, and the classifier was trained on
        real sentences — "magkano ang <item>" scores far better than "what about that one".
        """
        lowered = text.lower()
        if "magkano" in lowered or "presyo" in lowered or "price" in lowered or "how much" in lowered:
            return f"magkano ang {subject}"
        if "calorie" in lowered or "kcal" in lowered or "nutrition" in lowered or "protein" in lowered:
            return f"calories in {subject}"
        if "bakit" in lowered or "why" in lowered or "explain" in lowered:
            return f"bakit {subject}"
        # No verb of its own ("iyon din", "same"): reuse the shape of the previous question,
        # because the classifier recognises "magkano ang X" and not "iyon din".
        if self.last_intent.startswith("ph_price"):
            return f"magkano ang {subject}"
        if self.last_intent.startswith("ph_nutrition"):
            return f"calories in {subject}"
        return f"{text} ({subject})"

    def summary(self) -> dict[str, Any]:
        """Small snapshot for the chat response, so the UI can show what is remembered."""
        return {
            "turns": len(self.turns),
            "subject": self.last_entity,
            "discussed": list(self.entities),
        }


#: Words that ask for reasoning rather than a lookup. Used to decide whether an explanation
#: is wanted; with a language model that is possible, without one it is not.
_EXPLANATION = re.compile(
    r"(?:\bwhy\b|\bexplain\b|\breason\b|\bcause\b|\bhow\s+come\b|"
    r"\bbakit\b|\bpaano\b|\banong\s+dahilan\b|\bsaan\s+nagmula\b|\bipaliwanag\b)",
    re.IGNORECASE,
)


def wants_explanation(message: str) -> bool:
    """True when the message asks for reasoning, which a bag-of-words model cannot supply."""
    return bool(_EXPLANATION.search(message))
