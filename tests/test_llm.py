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
