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
