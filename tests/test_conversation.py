"""Tests for conversation memory and the chat paths that depend on it.

The behaviour under test is the difference between a command box and a conversation: a
follow-up must reach the thing it refers to, a reference that cannot be resolved must not be
guessed at, and a language model that is missing or broken must not take the chat down.
"""

from __future__ import annotations

import pytest

from bern.conversation import Memory, wants_explanation


class TestMemoryRecording:
    def test_entities_accumulate_so_the_previous_item_stays_reachable(self):
        memory = Memory()
        memory.record("q1", "ph_price_lookup", "r1", entities=["rice"])
        memory.record("q2", "ph_price_lookup", "r2", entities=["eggs"])
        assert memory.entities == ["rice", "eggs"]

    def test_repeat_entity_moves_to_the_end_without_duplicating(self):
        memory = Memory()
        memory.record("q1", "ph_price_lookup", "r", entities=["rice", "eggs"])
        memory.record("q2", "ph_price_lookup", "r", entities=["rice"])
        assert memory.entities == ["eggs", "rice"]

    def test_transcript_is_capped(self):
        memory = Memory()
        for index in range(40):
            memory.record(f"q{index}", "greeting", "r")
        assert len(memory.turns) == 12

    def test_turn_without_entities_keeps_the_previous_subject(self):
        memory = Memory()
        memory.record("q1", "ph_price_lookup", "r", entities=["rice"])
        memory.record("q2", "greeting", "r")
        assert memory.last_entity == "rice"


class TestReferenceResolution:
    def test_other_one_picks_the_item_not_currently_discussed(self):
        memory = Memory()
        memory.record("q", "ph_price_lookup", "r", entities=["rice", "eggs"])
        rewritten, subject = memory.resolve("what about the other one?")
        assert subject == "rice"
        assert rewritten == "magkano ang rice"

    def test_other_one_declines_when_only_one_item_was_ever_discussed(self):
        # Inventing a subject here would silently answer about the same thing while
        # sounding like it moved on.
        memory = Memory()
        memory.record("q", "ph_price_lookup", "r", entities=["rice"])
        rewritten, subject = memory.resolve("what about the other one?")
        assert subject is None
        assert rewritten == "what about the other one?"

    def test_same_reference_reuses_the_last_subject(self):
        memory = Memory()
        memory.record("q", "ph_price_lookup", "r", entities=["eggs"])
        rewritten, subject = memory.resolve("iyon din")
        assert subject == "eggs"
        assert rewritten == "magkano ang eggs"

    def test_nutrition_reference_uses_the_nutrition_shape(self):
        memory = Memory()
        memory.record("q", "ph_nutrition_lookup", "r", entities=["adobo"])
        rewritten, subject = memory.resolve("same for that")
        assert subject == "adobo"
        assert rewritten == "calories in adobo"

    def test_repeated_follow_up_advances_instead_of_repeating(self):
        memory = Memory()
        memory.record("q", "ph_price_lookup", "r", entities=["rice", "eggs"])
        first, _ = memory.resolve("what about the other one?")
        second, _ = memory.resolve("and that one?")
        assert first != second

    def test_long_message_is_not_treated_as_a_follow_up(self):
        memory = Memory()
        memory.record("q", "ph_price_lookup", "r", entities=["rice"])
        text = "how much is rice in the market today and also what about eggs"
        rewritten, subject = memory.resolve(text)
        assert subject is None
        assert rewritten == text

    def test_empty_session_resolves_nothing(self):
        assert Memory().resolve("what about that?") == ("what about that?", None)

    def test_summary_reports_what_is_remembered(self):
        memory = Memory()
        memory.record("q", "ph_price_lookup", "r", entities=["rice"])
        summary = memory.summary()
        assert summary["turns"] == 1
        assert summary["subject"] == "rice"
        assert summary["discussed"] == ["rice"]


class TestExplanationCues:
    @pytest.mark.parametrize("text", [
        "bakit mahal ang bigas",
        "why are rice prices rising",
        "explain photosynthesis",
        "paano nangyari ito",
    ])
    def test_reasoning_requests_are_detected(self, text):
        assert wants_explanation(text) is True

    @pytest.mark.parametrize("text", ["magkano ang bigas", "calories in adobo", "hello"])
    def test_lookups_are_not_reasoning_requests(self, text):
        assert wants_explanation(text) is False


class TestChatUsesMemory:
    """End-to-end through the engine, with no language model configured."""

    @pytest.fixture(scope="class")
    def bern(self):
        from bern.engine import BernAI

        return BernAI()

    def test_follow_up_resolves_to_the_previous_subject(self, bern):
        session = "test-memory-1"
        bern.chat("magkano ang bigas", session_id=session, use_llm=False)
        bern.chat("magkano ang itlog", session_id=session, use_llm=False)
        result = bern.chat("what about the other one?", session_id=session, use_llm=False)

        assert result["resolved_as"] is not None
        assert "rice" in result["resolved_as"].lower()
        assert result["intent"].startswith("ph_price")

    def test_conversation_snapshot_is_reported(self, bern):
        result = bern.chat("magkano ang bigas", session_id="test-snapshot", use_llm=False)
        assert result["conversation"]["turns"] == 1
        assert result["conversation"]["subject"]

    def test_sessions_do_not_leak_into_each_other(self, bern):
        bern.chat("magkano ang bigas", session_id="session-a", use_llm=False)
        result = bern.chat("what about the other one?", session_id="session-b", use_llm=False)
        assert result["resolved_as"] is None

    def test_local_engine_is_reported(self, bern):
        result = bern.chat("magkano ang bigas", session_id="test-local", engine="local")
        assert result["engine"] == "local"

    def test_declined_question_does_not_become_the_next_subject(self, bern):
        session = "test-declined"
        bern.chat("what is the airspeed of an unladen swallow", session_id=session, use_llm=False)
        result = bern.chat("what about that?", session_id=session, use_llm=False)
        assert result["resolved_as"] is None


class TestLanguageModelThatIsMissingOrBroken:
    """The chat must degrade to the local answer, never to an error."""

    @pytest.fixture(scope="class")
    def bern(self):
        from bern.engine import BernAI

        return BernAI()

    def test_no_key_keeps_the_local_answer(self, bern, monkeypatch):
        monkeypatch.delenv("BERN_LLM_API_KEY", raising=False)
        result = bern.chat("magkano ang bigas", session_id="test-nokey")
        assert result["engine"] == "local"

    def test_model_failure_falls_back_to_local(self, bern, monkeypatch):
        from bern import llm

        monkeypatch.setenv("BERN_LLM_API_KEY", "sk-test")
        monkeypatch.setattr(llm, "is_available", lambda: True)

        class Broken:
            def __init__(self, *args, **kwargs):
                pass

            def ask(self, *args, **kwargs):
                raise llm.LLMError("model unreachable")

        monkeypatch.setattr(llm, "TutorSession", Broken)
        result = bern.chat("bakit mahal ang bigas", session_id="test-broken")
        assert result["engine"] == "local"
        assert result["reply"]

    def test_explanation_request_without_a_model_says_so(self, bern, monkeypatch):
        monkeypatch.delenv("BERN_LLM_API_KEY", raising=False)
        result = bern.chat("bakit mahal ang bigas ngayon?", session_id="test-why")
        assert "can't explain why" in result["reply"]


# ------------------------------------------------------------------ conversational replies

# These are the questions that made Bern answer a request for plain talk with an essay on
# TF-IDF, or reply "Goodbye!" to "talk to me normally". Each case names the intent its wording
# targets, so a regression shows up as a routing failure rather than a differing sentence.
@pytest.fixture(scope="module")
def engine():
    """One trained engine for the whole module: loading the model takes about two seconds."""
    from bern.engine import BernAI

    return BernAI()


CONVERSATIONAL_CASES = [
    ("can you understand my prompts?", "comprehension_check"),
    ("do you understand me", "comprehension_check"),
    ("naiintindihan mo ba ako", "comprehension_check"),
    ("you did not answer my question", "conversation_repair"),
    ("that is not what i asked", "conversation_repair"),
    ("hindi yun ang tanong ko", "conversation_repair"),
    ("talk to me normally", "conversation_state"),
    ("you sound like a robot", "conversation_state"),
    ("are you listening", "conversation_state"),
    ("was that correct", "answer_check"),
    ("are you sure", "answer_check"),
    ("sigurado ka ba", "answer_check"),
]


@pytest.mark.parametrize("message,expected", CONVERSATIONAL_CASES)
def test_conversational_questions_route_to_their_own_intent(engine, message, expected):
    result = engine.chat(message)
    assert result["intent"] == expected, f"{message!r} -> {result['intent']}"
    assert not result["needs_training"]
    assert result["reply"].strip()


def test_understanding_question_is_not_answered_with_architecture_essay(engine):
    """The original complaint: a request for plain talk was met with a lecture on TF-IDF."""
    reply = engine.chat("can you understand my prompts?")["reply"].lower()
    for jargon in ("tf-idf", "logisticregression", "joblib", "n-gram", "cosine"):
        assert jargon not in reply, f"reply still recites {jargon}"


def test_weak_match_is_refused_instead_of_answered(engine):
    """A 28% guess used to be served as fact; the similarity clause in the gate hid that."""
    result = engine.chat("what did i just say")
    # Either it recognises the phrase (it is now trained) or it refuses - never a wrong topic
    # delivered with a "confident" framing.
    assert result["intent"] in ("conversation_repair", "unknown")


@pytest.mark.parametrize("message", ["why", "are we good", "can you hear me now"])
def test_unanswerable_small_prompts_are_not_dressed_up_as_answers(engine, message):
    """Either refuse and say the score, or flag that no explanation is possible.

    A bare "why" is answered through the reason path, which prepends what Bern cannot do rather
    than pretending the topic was understood. Both outcomes are honest; serving a stored reply
    as if it were the answer is not, and that is what the score in the refusal guards against.
    """
    result = engine.chat(message)
    reply = result["reply"].lower()
    # A refusal carries the score it guessed at, unless the reason path replaced the body -
    # "why" gets "I can't explain why" prepended instead, which is the same honesty in words.
    explained = "can't explain why" in reply or "no way to reason" in reply
    scored = str(round(result["confidence"] * 100)) in result["reply"]
    if result["needs_training"]:
        assert explained or scored, f"refused without saying why or how sure: {result['reply'][:120]}"
    else:
        assert explained, "answered a bare 'why' as though the topic was understood"


def test_answering_normally_does_not_break_the_old_intents(engine):
    """Tightening the gate must not cost real answers."""
    assert engine.chat("10 + 10")["reply"].find("20") >= 0
    assert engine.chat("who developed you")["intent"] == "developer"
    assert engine.chat("magkano ang itlog")["intent"] in ("ph_price_lookup", "unknown")


def test_no_reply_is_an_essay(engine):
    """A question deserves an answer, not a leaflet.

    "what can you do" once returned 1,116 characters that opened "Here is what I can do right
    now: 1. ..." and buried the answer. Asked twice it returned the same wall, because the
    bundle stored a single variant while the dataset held three. Both are checked here: the
    ceiling on length, and that the variants actually reach the model.
    """
    for tag, responses in engine.bundle["intent_responses"].items():
        assert len(responses) >= 1, tag
        for reply in responses:
            assert len(reply) < 700, f"{tag} reply is {len(reply)} chars: {reply[:70]}"


def test_answers_to_the_same_question_vary(engine):
    """One stored variant means the identical wall every time. Three means it reads like talk."""
    stored = engine.bundle["intent_responses"]
    assert len(stored["capabilities"]) == 3
    assert len(stored["greeting"]) >= 2


def test_capabilities_answers_in_one_line(engine):
    reply = engine.chat("what can you do")["reply"]
    assert len(reply) < 400
    assert reply.count("\n") <= 2, "a capability list is not a paragraph"
    for noise in ("1. **", "2. **", "/api/", "React Native"):
        assert noise not in reply, f"capabilities still recites {noise!r}"


def test_asking_whether_a_file_arrived_reaches_the_file_intent(engine):
    """"can you understand files i sent" was answered with the "do you understand me" reply.

    The file intent was trained only on commands - "read this pdf", "extract text from this
    file" - and a question about an attachment shares none of those words except "file", while
    sharing "understand" with a whole class of comprehension phrases. The question about the
    file lost to the question about the model.
    """
    for message in ("can you understand files i sent", "did you get my file",
                    "can you see my pdf", "did you receive the file"):
        result = engine.chat(message)
        assert result["intent"] == "file_scan", f"{message!r} -> {result['intent']}"

    # And the genuine comprehension questions must still land where they did.
    for message in ("do you understand me", "can you understand my prompts"):
        assert engine.chat(message)["intent"] == "comprehension_check"
