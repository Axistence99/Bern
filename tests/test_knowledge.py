"""Tests for fact-checking a learned claim, and for what "learn from every prompt" may mean.

The dangerous version of this feature stores whatever arrives and calls it knowledge. These
tests exist to keep the safe version in place: a claim is checked against data Bern actually
holds, a contradiction is refused, and everything outside price and nutrition is stored as
explicitly unverified rather than quietly promoted to fact.
"""

from __future__ import annotations

import json

import pytest

from bern import knowledge, memory


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Never write into the real learned-claims file from a test."""
    store = tmp_path / "learned_claims.json"
    monkeypatch.setattr(memory, "STORE_PATH", store)
    yield store
    if store.exists():
        store.unlink()


class TestCheckClaim:
    def test_a_true_price_is_verified(self):
        verdict = knowledge.check_claim("rice is 45 per kilo")
        assert verdict.status == "verified"
        assert "38" in verdict.reference_value

    def test_a_wrong_price_is_disputed_and_carries_the_real_range(self):
        verdict = knowledge.check_claim("rice is 65 per kilo")
        assert verdict.status == "disputed"
        assert "38" in verdict.detail and "50" in verdict.detail

    def test_a_true_nutrition_figure_is_verified(self):
        assert knowledge.check_claim("adobo has 312 calories").status == "verified"

    def test_a_wrong_nutrition_figure_is_disputed(self):
        assert knowledge.check_claim("pork adobo has 450 kcal per 100 g").status == "disputed"

    @pytest.mark.parametrize("claim", ["the sky is green", "manila is the capital", "berned rocks"])
    def test_a_claim_bern_has_no_data_for_is_unverified_not_true(self, claim):
        """The important distinction: unverified is not a pass."""
        assert knowledge.check_claim(claim).status == "unverified"

    def test_a_calorie_claim_is_not_checked_against_a_price_table(self):
        """"chicken joy is 300 kcal" once matched raw-chicken price data on the word "chicken"."""
        verdict = knowledge.check_claim("chicken joy is 300 kcal")
        assert verdict.status == "unverified"
        assert "per 100 g" not in verdict.detail or "calorie" in verdict.detail.lower()

    def test_a_generic_name_with_its_own_unit_still_matches(self):
        assert knowledge.check_claim("bigas 42 per kg").status == "verified"


class TestLearn:
    def test_a_verified_claim_is_stored(self):
        result = memory.learn("rice is 45 per kilo")
        assert result["saved"]
        assert result["verdict"]["status"] == "verified"
        assert len(memory.stored_claims()) == 1

    def test_a_disputed_claim_is_refused_by_default(self):
        result = memory.learn("rice is 65 per kilo")
        assert not result["saved"]
        assert "Refused" in result["reason"]
        assert memory.stored_claims() == []

    def test_a_disputed_claim_can_be_kept_with_an_override_and_keeps_the_disagreement(self):
        result = memory.learn("rice is 65 per kilo", override=True)
        assert result["saved"]
        stored = memory.stored_claims()[0]
        assert stored["verdict"] == "disputed"
        assert stored["reference_value"] == result["verdict"]["reference_value"]

    def test_an_unverified_claim_is_stored_marked_not_promoted(self):
        result = memory.learn("adobo has 9999 calories")
        if result["saved"]:
            assert memory.stored_claims()[0]["verdict"] == "unverified"
        assert "unverified" in result["reason"].lower() or not result["saved"]

    @pytest.mark.parametrize("message", [
        "what is the price of rice?", "hello bern", "my name is Ana", "can you understand me",
    ])
    def test_a_question_or_a_greeting_is_not_a_fact_and_is_not_stored(self, message):
        result = memory.learn(message)
        assert not result["saved"]
        assert memory.stored_claims() == []

    def test_the_same_claim_is_not_stored_twice(self):
        memory.learn("rice is 45 per kilo")
        again = memory.learn("rice is 45 per kilo")
        assert not again["saved"]
        assert len(memory.stored_claims()) == 1

    def test_verdict_counts_are_reportable(self):
        memory.learn("rice is 45 per kilo")
        memory.learn("rice is 65 per kilo", override=True)
        counts = memory.count_by_verdict()
        assert counts.get("verified") == 1
        assert counts.get("disputed") == 1


class TestEngineFacade:
    def test_learning_goes_through_the_engine_and_is_off_by_default_in_chat(self, isolated_store):
        from bern.engine import BernAI

        engine = BernAI()
        before = len(memory.stored_claims())
        engine.chat("rice is 45 per kilo")          # a normal turn must store nothing
        assert len(memory.stored_claims()) == before
        engine.learn_claim("rice is 45 per kilo")   # explicit learning does
        assert len(memory.stored_claims()) == before + 1
