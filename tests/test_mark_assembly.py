"""Tests for how marks and dictionary decoding assemble into words.

These cover the defects that only show up on a whole page: a mark split into pieces by the
segmenter, a mark isolated as its own "word", and a beam search that prunes the correct word
before the vocabulary bonus can rescue it.
"""

from __future__ import annotations

from bern.handwriting import HandwritingReader
from bern.ph_lexicon import decode_word, get_lexicon


class TestTidyMarks:
    """One glyph split into pieces must not become two marks."""

    def test_doubled_mark_collapses(self):
        assert HandwritingReader._tidy_marks("SALH-AT,,") == "SALH-AT,"

    def test_semicolon_then_comma_is_one_comma(self):
        assert HandwritingReader._tidy_marks("KANIN;,") == "KANIN,"

    def test_ellipsis_survives(self):
        # Three dots are deliberate; only an exactly doubled mark is treated as a split.
        assert HandwritingReader._tidy_marks("and...") == "and..."

    def test_single_mark_untouched(self):
        assert HandwritingReader._tidy_marks("MANILA.") == "MANILA."

    def test_short_strings_are_safe(self):
        assert HandwritingReader._tidy_marks("") == ""
        assert HandwritingReader._tidy_marks(".") == "."


class TestUnionBox:
    def test_overlapping_boxes_merge(self):
        merged = HandwritingReader._union_box((10, 5, 10, 20), (12, 5, 10, 20))
        assert merged == (10, 5, 12, 20)

    def test_separate_boxes_do_not_merge(self):
        assert HandwritingReader._union_box((0, 0, 10, 10), (40, 0, 10, 10)) is None

    def test_touching_boxes_merge(self):
        # A mark split down the middle yields halves that only touch.
        assert HandwritingReader._union_box((0, 0, 5, 16), (5, 0, 5, 16)) == (0, 0, 10, 16)


class TestPrefixConstrainedDecode:
    """The beam must keep a real vocabulary prefix alive while it is still a prefix."""

    def test_equally_likely_letters_prefer_a_real_prefix(self):
        # With the glyph probabilities equal, only the vocabulary prior can decide — which
        # is exactly the mechanism under test.
        lexicon = get_lexicon()
        options = [
            [("A", 0.5), ("X", 0.5)],
            [("D", 0.5), ("Q", 0.5)],
            [("J", 0.5), ("O", 0.5)],
        ]
        decoded, _ = decode_word(options, lexicon)
        assert decoded.lower().startswith("ad")

    def test_unknown_word_falls_back_to_argmax(self):
        lexicon = get_lexicon()
        options = [[("Z", 0.99), ("Q", 0.01)], [("Z", 0.99), ("Q", 0.01)]]
        decoded, _ = decode_word(options, lexicon)
        assert decoded.lower() == "zz" or decoded == "ZZ"

    def test_empty_input_is_safe(self):
        decoded, _ = decode_word([], get_lexicon())
        assert decoded == ""
