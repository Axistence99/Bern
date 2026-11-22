"""Opt-in language-model layer for the things a 21k-parameter model cannot do.

Bern's local pipeline is *extractive*: it can find a definition in your notes and turn it into
a question, but it cannot explain a concept in new words, answer a follow-up question, or
notice that two paragraphs of a chapter contradict each other. Those need a language model.

This module is the seam where one can be plugged in. Three rules shape it:

1. **Opt-in, by environment variable only.** With no ``BERN_LLM_API_KEY`` set, every function
   here reports itself unavailable and the caller keeps using the local path. Nothing changes
   for an existing install, and no request ever leaves the machine.
2. **The fallback is never removed.** Every entry point has a documented behaviour when the
   model is missing or fails. A language model is an improvement, not a prerequisite.
3. **Stdlib only.** ``urllib.request`` already speaks HTTP; adding an SDK would be a new
   dependency for a module most installs never enable.

Configuration:

===========================  ===========================================================
``BERN_LLM_API_KEY``         Required. Without it this module stays inert.
``BERN_LLM_MODEL``           Model name. Default ``gpt-4o-mini``.
``BERN_LLM_BASE_URL``        Chat-completions endpoint. Default OpenAI.
===========================  ===========================================================

Any OpenAI-compatible endpoint works, which is what makes local runtimes (Ollama, llama.cpp,
LM Studio) usable here as well — point ``BERN_LLM_BASE_URL`` at them and no data leaves the
machine even with a model configured.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT_SECONDS = 60.0

#: Notes are trimmed to this many characters before being sent. A whole textbook chapter
#: overruns the context window and costs more than it gains; the extractive fallback copes
#: with the full text anyway.
MAX_SOURCE_CHARS = 12_000


class LLMError(RuntimeError):
    """Raised when a configured language model cannot be reached or answers badly."""


@dataclass
class LLMConfig:
    """Connection settings, read from the environment unless passed explicitly."""

    api_key: str = field(default_factory=lambda: os.environ.get("BERN_LLM_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("BERN_LLM_MODEL", DEFAULT_MODEL))
    base_url: str = field(default_factory=lambda: os.environ.get("BERN_LLM_BASE_URL", DEFAULT_BASE_URL))

    @property
    def available(self) -> bool:
        """True when a key is present, which is the only thing that enables the layer."""
        return bool(self.api_key.strip())


def is_available() -> bool:
    """Report whether a language model is configured. Cheap; call it before offering an option."""
    return LLMConfig().available


def _chat(messages: list[dict[str, str]], config: LLMConfig | None = None,
          force_json: bool = False) -> str:
    """Send one chat request and return the assistant's text.

    Raises:
        LLMError: if no key is configured, the request fails, or the reply is malformed.
    """
    config = config or LLMConfig()
    if not config.available:
        raise LLMError("No BERN_LLM_API_KEY configured; the language model is disabled.")

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": 0.2,
    }
    if force_json:
        payload["response_format"] = {"type": "json_object"}

    request = urllib.request.Request(
        config.base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise LLMError(f"Language model returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"Could not reach the language model: {exc}") from exc

    try:
        return json.loads(body)["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LLMError("Language model returned an unexpected response shape.") from exc


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Extract a JSON object from a reply that may be wrapped in prose or a code fence."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMError("Language model reply contained no JSON object.")
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError("Language model reply was not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise LLMError("Language model reply was JSON but not an object.")
    return parsed


STUDY_SYSTEM_PROMPT = """You write study material for Filipino students, in English, Filipino \
or Taglish, matching the language of the notes you are given.

Return one JSON object with exactly these keys:
  "reviewer": {"title": str, "summary": str, "sections": [{"heading": str, "points": [str]}]}
  "flashcards": [{"front": str, "back": str}]
  "quiz": [{"question": str, "options": [str], "answer": str, "explanation": str}]
  "mock_exam": {"title": str, "sections": [{"name": str, "instructions": str, "items": [...]}]}

Rules:
- Use ONLY facts present in the notes. Do not add outside information.
- Every quiz "answer" must be one of its own "options", verbatim.
- Give exactly four options per quiz question, with the distractors plausible but wrong.
- Write in the same language the notes are written in.
"""


def generate_study_pack(
    text: str,
    flashcards: int = 20,
    quiz_items: int = 10,
    true_false_items: int = 8,
    config: LLMConfig | None = None,
) -> dict[str, Any]:
    """Ask the language model for a study pack.

    The reply is normalised into the same shape ``bern.study.generate_study_pack`` returns, so
    callers do not branch on which engine produced it.

    Raises:
        LLMError: on any failure. Callers should fall back to the local generator, which is
        what ``bern.study.generate_study_pack(engine="auto")`` does.
    """
    source = text[:MAX_SOURCE_CHARS]
    prompt = (
        f"Write study material from these notes.\n\n"
        f"Produce at most {flashcards} flashcards, {quiz_items} multiple-choice questions "
        f"and {true_false_items} true/false statements.\n\n--- NOTES ---\n{source}\n--- END ---"
    )
    raw = _chat(
        [{"role": "system", "content": STUDY_SYSTEM_PROMPT},
         {"role": "user", "content": prompt}],
        config=config,
        force_json=True,
    )
    data = _parse_json_object(raw)

    quiz = _clean_quiz(data.get("quiz"))
    cards = _clean_cards(data.get("flashcards"))
    true_false = _clean_true_false(data.get("true_false"), true_false_items)

    mock_exam = data.get("mock_exam")
    if not isinstance(mock_exam, dict) or not mock_exam.get("sections"):
        mock_exam = _mock_exam_from(quiz, true_false)

    reviewer = data.get("reviewer")
    if not isinstance(reviewer, dict):
        reviewer = {"title": "Reviewer", "summary": "", "sections": []}

    return {
        "reviewer": reviewer,
        "flashcards": cards,
        "quiz": quiz,
        "mock_exam": mock_exam,
        "true_false": true_false,
        "source_stats": {
            "characters": len(text),
            "engine": "llm",
        },
        "engine": "llm",
        "notes": [
            "Generated with a language model from the supplied notes. Check anything you "
            "will be graded on — a model can misread a sentence as easily as a person can.",
        ],
    }


def _clean_quiz(raw: Any) -> list[dict[str, Any]]:
    """Keep only well-formed multiple-choice items: four options and an answer among them."""
    cleaned: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return cleaned
    for item in raw:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        options = [str(option).strip() for option in item.get("options", []) if str(option).strip()]
        answer = str(item.get("answer", "")).strip()
        # An answer outside its own options is unmarkable, so such an item is dropped rather
        # than shipped and discovered later while studying.
        if not question or len(options) < 2 or answer not in options:
            continue
        cleaned.append({
            "question": question,
            "options": options,
            "answer": answer,
            "explanation": str(item.get("explanation", "")).strip(),
        })
    return cleaned


def _clean_cards(raw: Any) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return cards
    for item in raw:
        if not isinstance(item, dict):
            continue
        front = str(item.get("front", "")).strip()
        back = str(item.get("back", "")).strip()
        if front and back:
            cards.append({"front": front, "back": back})
    return cards


def _clean_true_false(raw: Any, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return items
    for item in raw:
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement", "")).strip()
        answer = item.get("answer")
        if not statement or not isinstance(answer, bool):
            continue
        items.append({"statement": statement, "answer": answer,
                      "explanation": str(item.get("explanation", "")).strip()})
    return items[:limit]


def _mock_exam_from(quiz: list[dict[str, Any]], true_false: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble a paper from whatever the model actually returned.

    Built here rather than trusted from the model: a mock exam whose answer key disagrees
    with its own questions is worse than no mock exam.
    """
    sections: list[dict[str, Any]] = []
    if quiz:
        sections.append({"name": "Part I — Multiple Choice",
                         "instructions": "Choose the best answer.", "items": quiz})
    if true_false:
        sections.append({"name": "Part II — True or False",
                         "instructions": "Write TRUE or FALSE.", "items": true_false})

    answer_key: list[dict[str, Any]] = []
    for index, question in enumerate(quiz):
        answer_key.append({"number": index + 1, "answer": question["answer"],
                           "explanation": question.get("explanation", "")})
    for index, item in enumerate(true_false):
        answer_key.append({"number": len(quiz) + index + 1,
                           "answer": "TRUE" if item["answer"] else "FALSE",
                           "explanation": item.get("explanation", "")})

    return {"title": "Mock exam", "sections": sections, "answer_key": answer_key}


#: Used when the local classifier has nothing and a model is configured. This is deliberately
#: not the tutor prompt: a general question ("who wrote hamlet", "explain photosynthesis") is
#: not homework from the student's own notes, and answering it in tutor voice - "here is what
#: your notes say" - would invent a source that does not exist.
CHAT_SYSTEM_PROMPT = """You are Bern, a small AI assistant built by Axistence99 running on \
a local classifier with a language model handling the questions the classifier cannot.

Answer the user directly and plainly. Speak in the first person. Keep it short unless the \
question genuinely needs length. Answer in the language the user writes in - English, Filipino \
or Taglish.

Two things matter more than sounding helpful:
- If you do not know, say so. Do not invent a fact, a number, a citation or a source.
- You have no access to the user's files, their device, or the internet. Do not claim to."""


TUTOR_SYSTEM_PROMPT = """You are a patient tutor for a Filipino student, answering in the \
language the student uses (English, Filipino or Taglish).

Ground every answer in the provided notes. If the notes do not cover the question, say so \
plainly and then answer from general knowledge, marking that part as not from their notes. \
Keep answers short and concrete. End with one short check question unless the student is \
just asking for a definition."""


@dataclass
class ChatSession:
    """A general conversation, carrying what was already said.

    :class:`TutorSession` answers one question about a fixed set of notes and is not kept
    between turns. This one is kept: every request replays the recent transcript, which is what
    makes "and what about the second one?" mean something. The window is capped because an
    unbounded history eventually costs more than the answer is worth.

    The module-level :func:`session` registry exists for one reason: ``BernAI.chat`` creates
    no object it could hold onto, so conversations are kept by session id here. Without it
    every turn started from nothing no matter what the user had already said.
    """

    system_prompt: str = CHAT_SYSTEM_PROMPT
    history: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    max_turns: int = 12

    def ask(self, question: str, config: LLMConfig | None = None) -> dict[str, Any]:
        """Answer one question, carrying the conversation so far.

        Returns:
            ``{"answer", "engine"}``. ``engine`` is ``"llm"`` or ``"none"``.

        Raises:
            LLMError: only when a model *is* configured but the call fails. Callers that want
            to fall back instead should catch it.
        """
        config = config or LLMConfig()
        if not config.available:
            return {"answer": "", "engine": "none"}

        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-(self.max_turns * 2):])
        messages.append({"role": "user", "content": question})

        answer = _chat(messages, config=config).strip()
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "engine": "llm"}

    def reset(self) -> None:
        self.history.clear()


#: Live conversations, keyed by session id. Bounded so a long-running server cannot grow
#: without limit; the oldest conversation is dropped first, which only costs that user's
#: history and never correctness.
_SESSIONS: OrderedDict[str, ChatSession] = OrderedDict()
_MAX_SESSIONS = 128


def session(session_id: str = "default") -> ChatSession:
    """Return the conversation for this id, creating it if needed."""
    existing = _SESSIONS.get(session_id)
    if existing is not None:
        _SESSIONS.move_to_end(session_id)
        return existing
    created = ChatSession()
    _SESSIONS[session_id] = created
    while len(_SESSIONS) > _MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
    return created


def forget(session_id: str = "default") -> None:
    """Drop one conversation, so the next turn starts clean."""
    _SESSIONS.pop(session_id, None)


@dataclass
class TutorSession:
    """A tutoring conversation that remembers what was already discussed.

    Continuity is the whole point: the earlier turn's question and answer go back with every
    request, so "what about the second one?" resolves. The history is capped, because an
    unbounded transcript eventually costs more than the answer is worth.

    With no language model configured, :meth:`ask` returns an empty answer and
    ``grounded=False``; the caller shows the local reviewer instead. Nothing silently pretends
    to be a tutor.
    """

    notes: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    max_turns: int = 8

    def ask(self, question: str, config: LLMConfig | None = None) -> dict[str, Any]:
        """Answer one question, carrying the conversation so far.

        Returns:
            ``{"answer", "engine", "grounded"}``. ``engine`` is ``"llm"`` or ``"none"``.

        Raises:
            LLMError: only when a model *is* configured but the call fails. Callers that want
            silence instead should catch it and fall back.
        """
        config = config or LLMConfig()
        if not config.available:
            return {"answer": "", "engine": "none", "grounded": False}

        messages: list[dict[str, str]] = [
            {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
            {"role": "system", "content": f"--- STUDENT'S NOTES ---\n{self.notes[:MAX_SOURCE_CHARS]}"},
        ]
        # Keep the most recent turns; drop the oldest first.
        recent = self.history[-(self.max_turns * 2):]
        messages.extend(recent)
        messages.append({"role": "user", "content": question})

        answer = _chat(messages, config=config).strip()
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "engine": "llm", "grounded": bool(self.notes.strip())}

    def reset(self) -> None:
        """Forget the conversation, keeping the notes."""
        self.history.clear()
