"""Tests for the opt-in language-model layer.

Everything here runs offline. The point of these tests is the *fallback contract*: with no
key configured the layer must be inert, and a failed call must be reported rather than
silently swallowed into an empty study pack.

The live HTTP path is not exercised — there is no API key in the test environment, and a test
that needs one would fail on every machine without it.
"""

from __future__ import annotations

import pytest

from bern import llm
from bern.study import generate_study_pack

NOTES = (
    "Mitochondria are the powerhouse of the cell. They produce ATP through respiration. "
    "Photosynthesis occurs in the chloroplast. Plants use sunlight to make glucose."
)


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    """Guarantee the layer is disabled regardless of the machine's environment."""
    monkeypatch.delenv("BERN_LLM_API_KEY", raising=False)


class TestAvailability:
    def test_disabled_without_a_key(self):
        assert llm.is_available() is False

    def test_enabled_by_a_key(self, monkeypatch):
        monkeypatch.setenv("BERN_LLM_API_KEY", "sk-test")
        assert llm.is_available() is True

    def test_blank_key_does_not_enable(self, monkeypatch):
        monkeypatch.setenv("BERN_LLM_API_KEY", "   ")
        assert llm.is_available() is False

    def test_model_and_url_are_configurable(self, monkeypatch):
        monkeypatch.setenv("BERN_LLM_MODEL", "local-model")
        monkeypatch.setenv("BERN_LLM_BASE_URL", "http://localhost:11434/v1/chat/completions")
        config = llm.LLMConfig()
        assert config.model == "local-model"
        assert config.base_url.startswith("http://localhost")


class TestNoKeyBehaviour:
    def test_generate_raises_rather_than_returning_an_empty_pack(self):
        # An empty pack that looks successful is the worst outcome for a student.
        with pytest.raises(llm.LLMError):
            llm.generate_study_pack(NOTES)

    def test_tutor_reports_itself_unavailable(self):
        result = llm.TutorSession(notes=NOTES).ask("What is ATP?")
        assert result["engine"] == "none"
        assert result["answer"] == ""


class TestStudyPackEngine:
    def test_default_engine_is_local(self):
        pack = generate_study_pack(NOTES)
        assert pack["engine"] == "local"

    def test_auto_falls_back_to_local_without_a_key(self):
        pack = generate_study_pack(NOTES, engine="auto")
        assert pack["engine"] == "local"
        assert pack["quiz"], "the local generator must still produce items"

    def test_explicit_llm_engine_fails_loudly(self):
        # Asking for the model explicitly must not quietly hand back a local pack; the
        # caller asked for something that is not available.
        with pytest.raises(llm.LLMError):
            generate_study_pack(NOTES, engine="llm")

    def test_local_pack_keeps_its_shape(self):
        pack = generate_study_pack(NOTES)
        for key in ("reviewer", "flashcards", "quiz", "mock_exam", "source_stats", "engine"):
            assert key in pack


class TestReplyCleaning:
    """A model's reply is untrusted input and is cleaned before it reaches a student."""

    def test_quiz_item_whose_answer_is_not_an_option_is_dropped(self):
        cleaned = llm._clean_quiz([
            {"question": "q1", "options": ["a", "b"], "answer": "z", "explanation": ""},
            {"question": "q2", "options": ["a", "b"], "answer": "a", "explanation": ""},
        ])
        assert [item["question"] for item in cleaned] == ["q2"]

    def test_quiz_item_needs_a_question_and_options(self):
        assert llm._clean_quiz([{"question": "", "options": ["a"], "answer": "a"}]) == []
        assert llm._clean_quiz("not a list") == []

    def test_cards_need_both_sides(self):
        assert llm._clean_cards([{"front": "a", "back": ""}, {"front": "b", "back": "c"}]) == [
            {"front": "b", "back": "c"}
        ]

    def test_true_false_needs_a_real_boolean(self):
        cleaned = llm._clean_true_false(
            [{"statement": "s", "answer": "true"}, {"statement": "t", "answer": True}], 5
        )
        assert len(cleaned) == 1 and cleaned[0]["statement"] == "t"


class TestJsonExtraction:
    def test_plain_object(self):
        assert llm._parse_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_object(self):
        assert llm._parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_object_wrapped_in_prose(self):
        assert llm._parse_json_object('Sure! {"a": 1} Hope that helps.') == {"a": 1}

    def test_no_object_raises(self):
        with pytest.raises(llm.LLMError):
            llm._parse_json_object("I cannot help with that.")

    def test_invalid_json_raises(self):
        with pytest.raises(llm.LLMError):
            llm._parse_json_object("{not json}")


class TestMockExamAssembly:
    def test_answer_key_numbers_run_on_from_quiz_to_true_false(self):
        quiz = [{"question": "q", "options": ["a", "b"], "answer": "a", "explanation": ""}]
        true_false = [{"statement": "s", "answer": False, "explanation": ""}]
        exam = llm._mock_exam_from(quiz, true_false)
        numbers = [item["number"] for item in exam["answer_key"]]
        assert numbers == [1, 2]
        assert exam["answer_key"][1]["answer"] == "FALSE"

    def test_empty_parts_produce_no_sections(self):
        exam = llm._mock_exam_from([], [])
        assert exam["sections"] == []
        assert exam["answer_key"] == []


class TestTutorSession:
    def test_reset_keeps_the_notes(self):
        session = llm.TutorSession(notes=NOTES)
        session.history.append({"role": "user", "content": "earlier"})
        session.reset()
        assert session.history == []
        assert session.notes == NOTES


# --------------------------------------------------------------- general conversation

# A real model cannot be called from a test suite: it needs a key and a network, and a test
# that depends on either is a test that fails for reasons unrelated to the code. So these
# cases stand up a server that speaks the chat-completions protocol and record what it was
# sent. That is enough to prove the wiring, the conversation memory and the fallback - and it
# is honest about what has *not* been proven: the quality of a real model's answers.

class _MockModel:
    """A chat-completions endpoint that echoes what it was asked, and records every request."""

    def __init__(self):
        self.requests: list[dict] = []

    def __call__(self, messages):
        self.requests.append({"messages": [dict(m) for m in messages]})
        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        return f"answer to: {user_turns[-1]}"


@pytest.fixture
def mock_model(monkeypatch):
    """Point the LLM layer at an in-process fake instead of a network endpoint."""
    from bern import llm

    fake = _MockModel()

    def fake_chat(messages, config=None, force_json=False):
        return fake(messages)

    monkeypatch.setattr(llm, "_chat", fake_chat)
    monkeypatch.setenv("BERN_LLM_API_KEY", "test-key-not-a-real-one")
    llm.forget("test-session")
    llm.forget("other-session")
    yield fake
    llm.forget("test-session")
    llm.forget("other-session")


class TestChatSession:
    def test_without_a_key_it_answers_nothing_and_claims_nothing(self, monkeypatch):
        from bern import llm

        monkeypatch.delenv("BERN_LLM_API_KEY", raising=False)
        result = llm.ChatSession().ask("who wrote hamlet")
        assert result == {"answer": "", "engine": "none"}

    def test_the_conversation_is_sent_back_with_the_next_question(self, mock_model):
        from bern import llm

        chat = llm.ChatSession()
        chat.ask("my name is Ana")
        chat.ask("what is my name?")

        last = mock_model.requests[-1]["messages"]
        carried = [m["content"] for m in last]
        assert "my name is Ana" in carried, "earlier turn was not replayed"
        assert last[-1] == {"role": "user", "content": "what is my name?"}

    def test_history_is_capped_so_a_long_chat_cannot_grow_without_bound(self, mock_model):
        from bern import llm

        chat = llm.ChatSession(max_turns=2)
        for i in range(10):
            chat.ask(f"message {i}")

        last = mock_model.requests[-1]["messages"]
        # system prompt + at most 2 turns (4 entries) + the new question
        assert len(last) <= 1 + 4 + 1
        assert "message 0" not in [m["content"] for m in last]

    def test_the_session_survives_between_calls_by_id(self, mock_model):
        from bern import llm

        llm.session("test-session").ask("my name is Ana")
        llm.session("test-session").ask("what is my name?")

        last = mock_model.requests[-1]["messages"]
        assert "my name is Ana" in [m["content"] for m in last]

    def test_two_conversations_do_not_share_history(self, mock_model):
        from bern import llm

        llm.session("test-session").ask("secret for session one")
        llm.session("other-session").ask("hello from the other chat")

        last = mock_model.requests[-1]["messages"]
        assert "secret for session one" not in [m["content"] for m in last]

    def test_forget_starts_a_clean_conversation(self, mock_model):
        from bern import llm

        llm.session("test-session").ask("my name is Ana")
        llm.forget("test-session")
        llm.session("test-session").ask("what is my name?")

        last = mock_model.requests[-1]["messages"]
        assert "my name is Ana" not in [m["content"] for m in last]


class TestGeneralChatThroughTheEngine:
    def test_a_general_question_is_answered_when_a_model_is_configured(self, mock_model):
        from bern.engine import BernAI

        result = BernAI().chat("who wrote hamlet")
        assert result["engine"] == "llm"
        assert "hamlet" in result["reply"].lower()
        assert not result["needs_training"]

    def test_without_a_model_a_general_question_is_refused_and_says_why(self, monkeypatch):
        from bern import llm
        from bern.engine import BernAI

        monkeypatch.delenv("BERN_LLM_API_KEY", raising=False)
        llm.forget("default")
        result = BernAI().chat("who wrote hamlet")
        assert result["needs_training"]
        assert "language model" in result["reply"].lower(), "refusal must say what is missing"

    def test_a_broken_model_does_not_take_the_chat_down(self, monkeypatch):
        from bern import llm
        from bern.engine import BernAI

        def explode(*args, **kwargs):
            raise llm.LLMError("connection refused")

        monkeypatch.setattr(llm, "_chat", explode)
        monkeypatch.setenv("BERN_LLM_API_KEY", "test-key-not-a-real-one")
        result = BernAI().chat("who wrote hamlet")
        # Falls back to the local refusal rather than raising into the caller.
        assert result["intent"] in ("unknown", "out_of_scope")
        assert result["reply"].strip()

    def test_arithmetic_still_beats_the_model(self, mock_model):
        """A calculator is not a language-model problem; the local answer must win."""
        from bern.engine import BernAI

        result = BernAI().chat("24 * 15")
        assert "360" in result["reply"]
        assert result.get("engine") != "llm"
        assert not mock_model.requests, "the model was called for a sum it was not needed for"

    def test_a_conversation_keeps_its_context_across_turns(self, mock_model):
        """Two general questions in a row: the second must arrive with the first.

        The router answers plenty of turns without any model at all - "my name is Ana" is a
        trained intent - so the case that proves memory is two questions the classifier cannot
        place, where the model is what carries the thread.
        """
        from bern import llm
        from bern.engine import BernAI

        llm.forget("default")
        engine = BernAI()
        # Both must be questions the classifier genuinely cannot place, or the router answers
        # them locally - which is correct, and proves nothing about the model's memory.
        first = engine.chat("explain photosynthesis")
        second = engine.chat("and how does the light reaction part work?")

        assert first["engine"] == "llm" and second["engine"] == "llm"
        last = mock_model.requests[-1]["messages"]
        assert "explain photosynthesis" in [m["content"] for m in last]

    def test_a_trained_intent_does_not_call_the_model_unnecessarily(self, mock_model):
        """The classifier is cheaper and exact for what it knows; the model is for the rest."""
        from bern import llm
        from bern.engine import BernAI

        llm.forget("default")
        result = BernAI().chat("who developed you")
        assert result["intent"] == "developer"
        assert not mock_model.requests
