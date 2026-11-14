"""Tests for document extraction, the lexicon, and study-material generation."""

from __future__ import annotations

import io
import zipfile
import zlib

import pytest

from bern.ph_lexicon import Lexicon, decode_word, get_lexicon, normalize_word
from bern.study import (
    extract_definitions,
    generate_study_pack,
    key_terms,
    split_sentences,
)
from bern.text_extract import ExtractionError, extract_text

ENGLISH_NOTES = """# Photosynthesis

Photosynthesis is the process by which green plants convert light energy into chemical energy stored in glucose. Chlorophyll is the green pigment that absorbs light.

The Calvin cycle is a series of reactions that fixes carbon dioxide into sugar in the stroma. Stomata are small pores on the leaf surface that allow gas exchange.
"""

FILIPINO_NOTES = """# Aralin 1: Nutrisyon

Ang nutrisyon ay ang proseso ng pagkuha ng mga sustansya mula sa pagkain para sa enerhiya. Ang protina ay isang sustansya na tumutulong sa pagbuo ng kalamnan.

Ang bigas ay isang butil na sagana sa carbohydrates.
"""


# --------------------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------------------

def test_lexicon_contains_filipino_and_english():
    lexicon = get_lexicon()
    for word in ("piso", "bigas", "salamat", "ang", "reviewer", "student"):
        assert word in lexicon, f"{word} should be in the lexicon"


def test_lexicon_is_not_empty_and_covers_datasets():
    lexicon = get_lexicon()
    assert len(lexicon) > 2000
    # Vocabulary harvested from Bern's bundled PH datasets.
    assert "malunggay" in lexicon or "sibuyas" in lexicon


def test_normalize_word_keeps_enie():
    assert normalize_word("Niño") == "niño"
    assert normalize_word("KÁNIN") == "kanin"


def test_decode_word_repairs_digit_for_letter():
    """The whole point of the dictionary: 'b1gas' must come back as 'bigas'."""
    glyph_options = [
        [("B", 0.99), ("8", 0.01)],
        [("1", 0.93), ("l", 0.06), ("I", 0.01)],
        [("G", 0.96), ("6", 0.02)],
        [("A", 0.99), ("H", 0.01)],
        [("S", 0.82), ("5", 0.18)],
    ]
    word, confidence = decode_word(glyph_options, get_lexicon())
    assert word.lower() == "bigas"
    assert 0.0 < confidence <= 1.0


def test_decode_word_falls_back_when_no_word_matches():
    glyph_options = [[("X", 0.9), ("Z", 0.1)], [("Q", 0.8), ("K", 0.2)]]
    word, _ = decode_word(glyph_options, get_lexicon())
    assert word.lower() == "xq"


def test_decode_word_handles_empty_input():
    assert decode_word([], get_lexicon()) == ("", 0.0)


def test_lexicon_suggest():
    lexicon = Lexicon(["bigas", "bili", "bahay"])
    assert "bigas" in lexicon.suggest("bigaz")
    assert lexicon.suggest("bigas") == ["bigas"]


# --------------------------------------------------------------------------------------
# Sentence and definition extraction
# --------------------------------------------------------------------------------------

def test_split_sentences_english():
    sentences = split_sentences(ENGLISH_NOTES)
    assert len(sentences) >= 4
    assert all(s.strip() for s in sentences)


def test_split_sentences_drops_headings():
    """A heading glued onto the next sentence produced nonsense glossary terms."""
    sentences = split_sentences(ENGLISH_NOTES)
    assert not any(s.startswith("#") for s in sentences)


def test_split_sentences_does_not_split_abbreviations():
    text = "Dr. Reyes studied the data. Prof. Cruz agreed with the result."
    sentences = split_sentences(text)
    assert len(sentences) == 2


def test_split_sentences_handles_unpunctuated_ocr_text():
    """OCR output often has no full stops at all; lines become the unit instead."""
    text = "BIGAS 50 PISO\nGATAS 90 PISO\nTUBIG 20 PISO"
    sentences = split_sentences(text)
    assert len(sentences) == 3


def test_extract_definitions_english():
    entries = extract_definitions(split_sentences(ENGLISH_NOTES))
    terms = {e["term"].lower() for e in entries}
    assert "photosynthesis" in terms
    assert "chlorophyll" in terms


def test_extract_definitions_filipino():
    entries = extract_definitions(split_sentences(FILIPINO_NOTES))
    terms = {e["term"].lower() for e in entries}
    assert "nutrisyon" in terms
    assert "protina" in terms


def test_key_terms_excludes_stopwords():
    terms = key_terms(ENGLISH_NOTES, split_sentences(ENGLISH_NOTES))
    assert "the" not in terms
    assert "ang" not in terms
    assert any(t in terms for t in ("photosynthesis", "chlorophyll", "calvin"))


# --------------------------------------------------------------------------------------
# Study pack
# --------------------------------------------------------------------------------------

def test_study_pack_has_all_sections():
    pack = generate_study_pack(ENGLISH_NOTES, flashcards=6, quiz_items=4)
    for key in ("reviewer", "flashcards", "quiz", "mock_exam", "source_stats"):
        assert key in pack


def test_flashcards_have_front_and_back():
    pack = generate_study_pack(ENGLISH_NOTES, flashcards=5)
    assert pack["flashcards"]
    for card in pack["flashcards"]:
        assert card["front"].strip()
        assert card["back"].strip()


def test_quiz_options_contain_the_answer():
    pack = generate_study_pack(ENGLISH_NOTES, quiz_items=5)
    for question in pack["quiz"]:
        assert question["answer"] in question["options"]
        assert question["options"][question["answer_index"]] == question["answer"]
        assert len(question["options"]) == len(set(question["options"]))


def test_quiz_distractors_are_not_the_answer():
    pack = generate_study_pack(ENGLISH_NOTES, quiz_items=5)
    for question in pack["quiz"]:
        wrong = [o for o in question["options"] if o != question["answer"]]
        assert question["answer"] not in wrong


def test_true_false_answer_key_is_balanced():
    """Leaving the mix to chance produced 7-of-8 TRUE papers."""
    pack = generate_study_pack(ENGLISH_NOTES, true_false_items=8)
    items = pack["mock_exam"]["sections"][1]["items"]
    true_count = sum(1 for item in items if item["answer"])
    assert abs(true_count - (len(items) - true_count)) <= 1


def test_study_pack_is_deterministic_for_a_given_seed():
    first = generate_study_pack(ENGLISH_NOTES, quiz_items=4, seed=7)
    second = generate_study_pack(ENGLISH_NOTES, quiz_items=4, seed=7)
    assert [q["question"] for q in first["quiz"]] == [q["question"] for q in second["quiz"]]


def test_study_pack_works_on_filipino_notes():
    pack = generate_study_pack(FILIPINO_NOTES, flashcards=5, quiz_items=4)
    assert pack["reviewer"]["glossary"]
    terms = {g["term"].lower() for g in pack["reviewer"]["glossary"]}
    assert "nutrisyon" in terms


def test_study_pack_handles_empty_text():
    pack = generate_study_pack("", flashcards=3)
    assert pack["flashcards"] == []
    assert pack["source_stats"]["characters"] == 0


def test_reviewer_sections_match_their_own_content():
    """Points must come from the section they belong to, not a globally sliced list."""
    pack = generate_study_pack(ENGLISH_NOTES)
    headings = [s["heading"] for s in pack["reviewer"]["sections"]]
    assert "Photosynthesis" in headings


# --------------------------------------------------------------------------------------
# Document extraction
# --------------------------------------------------------------------------------------

def _make_pdf(lines: list[str]) -> bytes:
    """Build a minimal single-page PDF with a Flate-compressed content stream."""
    content = "BT /F1 12 Tf 40 750 Td\n"
    for line in lines:
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content += f"({escaped}) Tj 0 -20 Td\n"
    content += "ET"
    compressed = zlib.compress(content.encode("latin-1"))

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(compressed)).encode() + b" /Filter /FlateDecode >>\nstream\n"
        + compressed + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = b"%PDF-1.4\n"
    offsets = []
    for index, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


def _make_docx(paragraphs: list[str]) -> bytes:
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    xml = (f'<?xml version="1.0" encoding="UTF-8"?>'
           f'<w:document xmlns:w="{namespace}"><w:body>{body}</w:body></w:document>')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


def test_extract_pdf():
    result = extract_text(_make_pdf(["Photosynthesis is the process plants use.",
                                     "Ang bigas ay isang butil."]), "notes.pdf")
    assert result["source_type"] == "pdf"
    assert "Photosynthesis" in result["text"]
    assert "bigas" in result["text"]


def test_extract_docx():
    result = extract_text(_make_docx(["Cell Respiration",
                                      "Cellular respiration releases energy from glucose.",
                                      "Ang protina ay tumutulong sa kalamnan."]), "notes.docx")
    assert result["source_type"] == "docx"
    assert "Cellular respiration" in result["text"]
    assert "protina" in result["text"]


def test_extract_txt_handles_bom_and_bad_encoding():
    data = "Ang bigas ay butil.".encode("utf-8-sig")
    assert "bigas" in extract_text(data, "notes.txt")["text"]
    latin = "Señor José".encode("latin-1")
    assert "José" in extract_text(latin, "notes.txt")["text"]


def test_extract_rejects_unknown_extension():
    with pytest.raises(ExtractionError):
        extract_text(b"whatever", "notes.xlsx")


def test_extract_rejects_encrypted_pdf():
    with pytest.raises(ExtractionError):
        extract_text(b"%PDF-1.4 /Encrypt 9 0 R", "secret.pdf")


def test_extract_bad_docx_raises():
    with pytest.raises(ExtractionError):
        extract_text(b"not a zip at all", "bad.docx")


def test_extract_empty_pdf_is_empty_not_crashing():
    result = extract_text(b"%PDF-1.4\n%%EOF", "empty.pdf")
    assert result["text"] == ""
    assert result["characters"] == 0
