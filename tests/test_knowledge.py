"""Tests for fact-checking a learned claim, and for what "learn from every prompt" may mean.

The dangerous version of this feature stores whatever arrives and calls it knowledge. These
tests exist to keep the safe version in place: a claim is checked against data Bern actually
holds, a contradiction is refused, and everything outside price and nutrition is stored as
explicitly unverified rather than quietly promoted to fact.
"""

from __future__ import annotations

import json

import numpy as np

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


# ------------------------------------------------------------------ file formats

def _xlsx_bytes() -> bytes:
    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("xl/sharedStrings.xml",
                         '<sst xmlns="x"><si><t>Item</t></si><si><t>Rice</t></si></sst>')
        archive.writestr("xl/worksheets/sheet1.xml",
                         '<worksheet xmlns="x"><sheetData>'
                         '<row><c t="s"><v>0</v></c><c><v>45</v></c></row>'
                         '<row><c t="s"><v>1</v></c></row>'
                         '</sheetData></worksheet>')
    return buf.getvalue()


def _pptx_bytes() -> bytes:
    import io, zipfile
    def slide(text):
        return ('<p:sld xmlns:p="p" xmlns:a="a"><p:cSld><p:spTree><p:sp><p:txBody>'
                f'<a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp>'
                '</p:spTree></p:cSld></p:sld>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide("Photosynthesis"))
        archive.writestr("ppt/slides/slide2.xml", slide("Chlorophyll"))
        archive.writestr("ppt/slides/slide10.xml", slide("Last"))
    return buf.getvalue()


class TestSpreadsheetAndSlides:
    """.xlsx failed outright before this, which is the plainest gap in file support."""

    def test_spreadsheet_cells_are_read(self):
        from bern.text_extract import extract_text

        result = extract_text(_xlsx_bytes(), "notes.xlsx")
        assert "Item" in result["text"] and "45" in result["text"]
        assert "Rice" in result["text"]

    def test_slide_text_is_read_in_slide_order(self):
        from bern.text_extract import extract_text

        text = extract_text(_pptx_bytes(), "deck.pptx")["text"]
        assert "Photosynthesis" in text and "Chlorophyll" in text
        # "slide10" must not sort between slide1 and slide2.
        assert text.index("Chlorophyll") < text.index("Last")

    def test_a_broken_spreadsheet_says_what_went_wrong(self):
        from bern.text_extract import ExtractionError, extract_text

        with pytest.raises(ExtractionError):
            extract_text(b"not a zip at all", "broken.xlsx")


class TestPrintedGlyphCoverage:
    """The reader is pointed at printed text but was trained only on handwriting.

    These cases pin the two things that measurement exposed: the font split must withhold whole
    families (holding out one file left every other serif in training), and the reader must
    actually cope with a serif at all.
    """

    def test_the_font_split_withholds_whole_families(self):
        from bern import printed_glyphs as pg

        train, held = pg.split_fonts(holdout_families=2)
        train_families = {pg.family_of(f) for f in train}
        held_families = {pg.family_of(f) for f in held}
        assert not (train_families & held_families), "a family was held out but still trained on"
        assert held_families

    def test_hard_edged_glyphs_read_at_least_as_well_as_smooth_ones(self):
        """A photo arrives binarised, so training has to cover that, not just smooth renders.

        The page reader masks every page against a local threshold before reading, which means
        every glyph reaching the model has hard edges. Measured before this was trained in: 79.4%
        on hard-edged glyphs against 90.1% on smooth ones. Half the printed training samples are
        now binarised so the gap closes.
        """
        from bern import handwriting, printed_glyphs as pg

        _, held = pg.split_fonts(holdout_families=2)
        if not held:
            pytest.skip("no held-out families available on this machine")
        backend = handwriting.HandwritingReader()._resolve_backend()

        def read_rate(binarize: bool) -> float:
            correct = total = 0
            for ch in "ABCDEFGHKLMNOPRSTUWXYZ0123456789":
                glyph = pg.render(ch, held[0], 56, binarize=binarize)
                if glyph is None:
                    continue
                patch = np.asarray(glyph, dtype=np.uint8).reshape(28, 28)
                bgr = np.repeat(patch[:, :, None], 3, axis=2)
                total += 1
                if backend.recognize(bgr)[0] == ch:
                    correct += 1
            return correct / max(1, total)

        sharp = read_rate(True)
        smooth = read_rate(False)
        assert sharp >= smooth - 0.05, f"hard-edged {sharp:.2f} vs smooth {smooth:.2f}"

    def test_a_rendered_serif_is_read_as_the_right_character(self):
        from bern import handwriting, printed_glyphs as pg

        train, held = pg.split_fonts(holdout_families=2)
        if not held:
            pytest.skip("no held-out families available on this machine")
        # The recogniser lives on the numeric backend the reader resolves, not on the reader.
        backend = handwriting.HandwritingReader()._resolve_backend()
        correct = total = 0
        for ch in "ABCDEFGHIKLMNOPRSTUVWXYZ0123456789":
            glyph = pg.render(ch, held[0], 56)
            if glyph is None:
                continue
            # recognize() wants a BGR glyph the size the model was trained on.
            patch = np.asarray(glyph, dtype=np.uint8).reshape(28, 28)
            bgr = np.repeat(patch[:, :, None], 3, axis=2)
            total += 1
            if backend.recognize(bgr)[0] == ch:
                correct += 1
        assert total, "nothing rendered"
        # The threshold is deliberately below what the page reader achieves, because this test
        # feeds single held-out serif glyphs straight to `recognize()` with no line context and no
        # case fold - the hardest form of the task. Its job is to catch a model that has stopped
        # reading unseen printed type at all. Measured 61.8% here; the page pipeline reads 84% on
        # cleanly segmented lines.
        assert correct / total > 0.55, f"held-out font read {correct}/{total}"
