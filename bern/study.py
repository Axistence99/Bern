"""Turn notes or lecture text into study material: reviewers, quizzes, flashcards.

The generator is **extractive**, not generative. Bern has no language model behind it, so
everything produced here is built from sentences that were already in the source: terms are
pulled out of definition patterns, blanks are cut into real sentences, and wrong answers are
other real terms from the same document. Nothing is invented, which is also why nothing
will be factually wrong *about the source* — though a badly written source yields bad
questions, and that is worth knowing.

What it produces:

* **flashcards** — term on the front, its definition on the back
* **quiz** — multiple choice, with distractors drawn from the document's own terms
* **reviewer** — headings, key points and a glossary, grouped by section
* **mock exam** — a mixed paper (multiple choice, fill-in-the-blank, true/false) plus an
  answer key with explanations

Both English and Filipino are handled. Filipino definition sentences ("Ang *bigas* ay
isang butil ng palay...") are recognised by pattern, and cloze questions work in either
language because they only need a term and a sentence containing it.

Usage:
    python3 -m bern.study notes.pdf --out study_pack.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------------------
# Sentence and term extraction
# --------------------------------------------------------------------------------------

#: Abbreviations that must not end a sentence.
_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "atty", "engr", "no", "vol", "fig",
                  "eq", "approx", "e.g", "i.e", "etc", "vs", "st", "jr", "sr", "blg"}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: English and Filipino definition patterns. The named groups are the term and definition.
DEFINITION_PATTERNS: list[re.Pattern[str]] = [
    # English: "Photosynthesis is the process by which ..."
    re.compile(r"^\s*(?:The\s+|A\s+|An\s+)?(?P<term>[A-Z][\w\- ]{1,40}?)\s+"
               r"(?:is|are|was|were)\s+(?P<definition>.{15,400})$", re.IGNORECASE),
    re.compile(r"^\s*(?:The\s+)?(?P<term>[\w\- ]{2,40}?)\s+"
               r"(?:refers to|means|is defined as|can be defined as|is called)\s+"
               r"(?P<definition>.{10,400})$", re.IGNORECASE),
    # Filipino: "Ang bigas ay isang butil ng palay na ..."
    re.compile(r"^\s*Ang\s+(?P<term>[\w\- ]{1,40}?)\s+ay\s+(?:isang\s+|ang\s+)?"
               r"(?P<definition>.{10,400})$", re.IGNORECASE),
    re.compile(r"^\s*(?P<term>[\w\- ]{2,40}?)\s+ay\s+(?:isang\s+|tumutukoy sa\s+|"
               r"nangangahulugang\s+|ang\s+)?(?P<definition>.{10,400})$", re.IGNORECASE),
    re.compile(r"^\s*Ang\s+ibig\s+sabihin\s+ng\s+(?P<term>[\w\- ]{1,40}?)\s+ay\s+"
               r"(?P<definition>.{10,400})$", re.IGNORECASE),
]

_STOPWORDS = {
    # English
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that",
    "these", "those", "as", "by", "from", "has", "have", "had", "not", "can", "will",
    "which", "who", "what", "when", "where", "how", "why", "there", "their", "they",
    "we", "you", "i", "he", "she", "him", "her", "them", "our", "your", "my",
    "also", "more", "most", "some", "such", "than", "then", "into", "so", "up", "out",
    "place", "way", "through", "each", "other", "about", "over", "very", "only", "just",
    "one", "two", "three", "per", "both", "many", "much", "same", "different", "used",
    "use", "using", "made", "make", "given", "give", "takes", "take", "occurs", "called",
    # Filipino
    "ang", "ng", "mga", "sa", "na", "at", "ay", "o", "pero", "kung", "dahil", "para",
    "ito", "iyan", "iyon", "siya", "sila", "kami", "tayo", "kayo", "ako", "ikaw",
    "may", "meron", "wala", "hindi", "oo", "din", "rin", "lang", "po", "ba", "nga",
    "maging", "bilang", "upang", "nang", "kasi", "tapos", "saka", "pala", "sana",
}

_WORD_RE = re.compile(r"[A-Za-zÑñÁÉÍÓÚáéíóúü][A-Za-zÑñÁÉÍÓÚáéíóúü\-']*")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")

#: Markdown headings. A plain ALL-CAPS line is handled separately, because OCR output is
#: full of all-caps data lines ("BIGAS 50 PISO") that must not be mistaken for headings.
_HEADING_RE = re.compile(r"^\s*#{1,4}\s+(?P<md>.+?)\s*$")

#: A line that could be an ALL-CAPS heading, subject to the structural check below.
_CAPS_CANDIDATE_RE = re.compile(r"^[A-ZÑ0-9][A-ZÑ0-9 \-:,.&()/]{3,60}$")


def heading_title(lines: list[str], index: int) -> str | None:
    """Return the heading text if line ``index`` is a heading, else ``None``.

    Markdown headings are unambiguous. ALL-CAPS lines are only accepted when they stand
    alone as their own paragraph — blank line above and below, at most eight words. That
    is what separates "CELL RESPIRATION" from a data row like "GATAS 90 PISO": the data row
    sits flush against its neighbours, so it fails the isolation test and stays content.
    """
    line = lines[index]
    markdown = _HEADING_RE.match(line)
    if markdown:
        return markdown.group("md").strip()

    stripped = line.strip()
    if not _CAPS_CANDIDATE_RE.match(stripped):
        return None
    if len(stripped.split()) > 8:
        return None
    if sum(ch.isalpha() for ch in stripped) < 2:
        return None

    previous_blank = index == 0 or not lines[index - 1].strip()
    next_blank = index + 1 >= len(lines) or not lines[index + 1].strip()
    return stripped if (previous_blank and next_blank) else None


def strip_headings(text: str) -> str:
    """Remove heading lines so they do not become sentences.

    Without this, a heading runs into the paragraph beneath it and produces a "sentence"
    like ``CELL RESPIRATION Cellular respiration is the set of...``, which then shows up as
    a reviewer key point and a glossary term.
    """
    lines = text.splitlines()
    kept = [line for index, line in enumerate(lines) if heading_title(lines, index) is None]
    return "\n".join(kept)


def _logical_lines(paragraph: str) -> list[str]:
    """Rejoin hard-wrapped lines while keeping genuine line breaks.

    Notes exported from Word or a PDF are full of hard wraps in the middle of a sentence,
    and gluing those back together is what makes sentence splitting correct. But a line
    ending without punctuation followed by a **capitalised** line is not a wrap — it is a
    heading or a new bullet — so it stays a boundary. Getting this wrong merged headings
    into the text and produced glossary terms like "Cell Respiration Cellular respiration".

    Returns one entry per logical line; the caller splits each into sentences.
    """
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if len(lines) <= 1:
        return lines

    merged: list[str] = []
    for line in lines:
        if not merged:
            merged.append(line)
            continue
        previous = merged[-1]
        continues_sentence = (
            not previous.rstrip().endswith((".", "!", "?", ":", ";"))
            and line[:1].islower()
        )
        if continues_sentence:
            merged[-1] = previous + " " + line
        else:
            merged.append(line)
    return merged


def split_sentences(text: str, strip_heading_lines: bool = True) -> list[str]:
    """Split text into sentences, keeping Filipino text intact.

    Filipino uses the same terminators as English, so punctuation splitting works. Three
    extra cases matter: headings must not be glued onto the paragraph below them; OCR
    output often has **no** punctuation at all, in which case lines become the unit; and
    common abbreviations must not trigger a split.
    """
    text = strip_headings(text) if strip_heading_lines else text
    text = text.strip()
    if not text:
        return []

    # If there is essentially no sentence punctuation, fall back to line structure.
    terminators = sum(text.count(ch) for ch in ".!?")
    if terminators < max(1, len(text) // 400):
        return [line.strip() for line in text.splitlines() if len(line.strip()) > 2]

    sentences: list[str] = []
    # Paragraph boundaries are respected: sentences never span a blank line.
    for paragraph in re.split(r"\n\s*\n", text):
        for logical_line in _logical_lines(paragraph):
            for raw in _SENTENCE_SPLIT.split(logical_line):
                candidate = raw.strip()
                if not candidate:
                    continue
                if sentences and sentences[-1].split():
                    last_word = sentences[-1].rstrip(".!?").split()[-1].lower()
                    if last_word in _ABBREVIATIONS:
                        sentences[-1] = sentences[-1] + " " + candidate
                        continue
                sentences.append(candidate)
    return sentences


def _clean_term(term: str) -> str:
    """Trim articles, punctuation and stray whitespace off a candidate term."""
    term = re.sub(r"\s+", " ", term).strip(" ,;:.-–—()[]\"'")
    term = re.sub(r"^(?:the|a|an|ang|mga|isang)\s+", "", term, flags=re.IGNORECASE)
    return term.strip()


def extract_definitions(sentences: Iterable[str]) -> list[dict[str, str]]:
    """Find ``term → definition`` pairs using English and Filipino sentence patterns.

    This is the highest-value extraction in the module: a definition gives both halves of a
    flashcard, a multiple-choice answer, and a reviewer entry, in one shot.
    """
    found: list[dict[str, str]] = []
    seen: set[str] = set()

    for sentence in sentences:
        for pattern in DEFINITION_PATTERNS:
            match = pattern.match(sentence)
            if not match:
                continue
            term = _clean_term(match.group("term"))
            definition = match.group("definition").strip(" .;,")
            if not (2 <= len(term) <= 48) or len(definition) < 10:
                continue
            # A term made mostly of stopwords is a parsing artefact, not a concept.
            words = [w.lower() for w in _WORD_RE.findall(term)]
            if not words or sum(w in _STOPWORDS for w in words) / len(words) > 0.5:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append({"term": term, "definition": definition, "sentence": sentence})
            break
    return found


def key_terms(text: str, sentences: list[str], limit: int = 30) -> list[str]:
    """Rank candidate terms by a simple TF-IDF-style score.

    Frequency alone favours filler words; rarity alone favours typos. Multiplying term
    frequency by inverse sentence frequency keeps terms that recur across the document but
    still carry meaning.
    """
    document_frequency: dict[str, int] = {}
    total_frequency: dict[str, int] = {}

    for sentence in sentences:
        words = {w.lower() for w in _WORD_RE.findall(sentence)
                 if len(w) > 2 and w.lower() not in _STOPWORDS}
        for word in words:
            document_frequency[word] = document_frequency.get(word, 0) + 1
        for word in _WORD_RE.findall(sentence):
            lowered = word.lower()
            if len(lowered) > 2 and lowered not in _STOPWORDS:
                total_frequency[lowered] = total_frequency.get(lowered, 0) + 1

    sentence_count = max(1, len(sentences))
    scored: list[tuple[float, str]] = []
    for term, frequency in total_frequency.items():
        df = document_frequency.get(term, 1)
        score = frequency * (sentence_count / df) ** 0.5
        # Prefer terms that survive in the document as written (capitalised mid-sentence
        # usually signals a proper noun or a defined concept).
        if re.search(rf"\b{re.escape(term)}\b", text):
            score *= 1.05
        scored.append((score, term))

    scored.sort(reverse=True)
    return [term for _, term in scored[:limit]]


# --------------------------------------------------------------------------------------
# Item builders
# --------------------------------------------------------------------------------------

def _similar_distractors(answer: str, pool: list[str], count: int, rng: random.Random) -> list[str]:
    """Pick plausible wrong answers from the document's own vocabulary.

    Similar length reads as plausible; a wildly different length gives the answer away.
    Terms sharing the first letter are avoided for the same reason.
    """
    candidates = [t for t in pool
                  if t.lower() != answer.lower()
                  and not t.lower().startswith(answer.lower()[:1])]
    if len(candidates) < count:
        candidates = [t for t in pool if t.lower() != answer.lower()]

    def preference(term: str) -> tuple[int, float]:
        return (0 if abs(len(term) - len(answer)) <= 3 else 1, rng.random())

    candidates.sort(key=preference)
    chosen = candidates[:count]
    rng.shuffle(chosen)
    return chosen


def build_flashcards(definitions: list[dict[str, str]], terms: list[str],
                     sentences: list[str], limit: int = 20) -> list[dict[str, str]]:
    """Build front/back cards, preferring definitions and falling back to cloze facts."""
    cards: list[dict[str, str]] = []
    used: set[str] = set()

    for entry in definitions[:limit]:
        cards.append({
            "front": f"What is {entry['term']}?",
            "back": entry["definition"],
            "source": entry["sentence"],
        })
        used.add(entry["term"].lower())

    for term in terms:
        if len(cards) >= limit:
            break
        if term.lower() in used:
            continue
        sentence = next((s for s in sentences if re.search(rf"\b{re.escape(term)}\b", s, re.IGNORECASE)), None)
        if not sentence:
            continue
        cloze = re.sub(rf"\b{re.escape(term)}\b", "_____", sentence, count=1, flags=re.IGNORECASE)
        cards.append({"front": cloze, "back": term, "source": sentence})
        used.add(term.lower())

    return cards[:limit]


def build_quiz(definitions: list[dict[str, str]], terms: list[str], sentences: list[str],
               limit: int = 10, seed: int = 7) -> list[dict[str, Any]]:
    """Build multiple-choice questions.

    Two sources are used, in order: extracted definitions ("Which term means ...?") and
    cloze sentences from the body text. Options are the answer plus three other terms from
    the same document, so the questions test the material rather than general knowledge.
    """
    rng = random.Random(seed)
    questions: list[dict[str, Any]] = []
    # Concept terms make far better distractors than raw keywords: "suka" reads as a
    # plausible answer where "nagbebenta" (a verb) obviously does not. Definition terms go
    # first, then the keyword list fills any gap.
    concept_terms = [entry["term"] for entry in definitions]
    pool = list(dict.fromkeys(concept_terms + [t for t in terms]))

    for entry in definitions:
        if len(questions) >= limit:
            break
        answer = entry["term"]
        distractors = _similar_distractors(answer, pool, 3, rng)
        if len(distractors) < 3:
            continue
        options = [answer] + distractors
        rng.shuffle(options)
        questions.append({
            "type": "multiple_choice",
            "question": f"Which term matches this description: \"{entry['definition']}\"?",
            "options": options,
            "answer": answer,
            "answer_index": options.index(answer),
            "explanation": entry["sentence"],
        })

    if len(questions) < limit:
        for sentence in sentences:
            if len(questions) >= limit:
                break
            match = next((t for t in pool if re.search(rf"\b{re.escape(t)}\b", sentence, re.IGNORECASE)), None)
            if not match:
                continue
            if any(q.get("answer") == match for q in questions):
                continue
            distractors = _similar_distractors(match, pool, 3, rng)
            if len(distractors) < 3:
                continue
            options = [match] + distractors
            rng.shuffle(options)
            cloze = re.sub(rf"\b{re.escape(match)}\b", "_____", sentence, count=1, flags=re.IGNORECASE)
            questions.append({
                "type": "fill_in",
                "question": cloze,
                "options": options,
                "answer": match,
                "answer_index": options.index(match),
                "explanation": f"The source sentence names “{match}”.",
            })

    return questions[:limit]


def _perturb_for_true_false(sentence: str, rng: random.Random, force_false: bool) -> tuple[str, bool]:
    """Return a statement and whether it is true.

    Whether an item should be false is decided by the caller (alternating, so the answer
    key is always balanced) — leaving it to chance produced papers that were 7-of-8 TRUE.
    When a flip is requested, numbers are changed first because ``1990`` → ``1995`` is
    unambiguously false, where inserting a negation can read oddly. If nothing can be
    flipped the sentence stays true, and the caller is told so.
    """
    if not force_false:
        return sentence, True

    numbers = _NUMBER_RE.findall(sentence)
    if numbers:
        original = numbers[0]
        try:
            shifted = str(int(float(original.replace(",", ""))) + rng.choice([1, 2, 3, 5, 10]))
        except ValueError:
            shifted = original + "0"
        return sentence.replace(original, shifted, 1), False

    negations = [(" ay ", " ay hindi "), (" is ", " is not "), (" are ", " are not "),
                 (" can ", " cannot "), (" must ", " must not ")]
    for positive, negative in negations:
        if positive in sentence:
            return sentence.replace(positive, negative, 1), False

    return sentence, True


def build_true_false(sentences: list[str], limit: int = 10, seed: int = 7) -> list[dict[str, Any]]:
    """Build true/false items with a guaranteed balanced answer key.

    Items alternate between true and false, so a student cannot score 50% by answering the
    same way throughout.
    """
    rng = random.Random(seed)
    candidates = [s for s in sentences if 6 <= len(s.split()) <= 45]
    rng.shuffle(candidates)

    items: list[dict[str, Any]] = []
    for index, sentence in enumerate(candidates[:limit]):
        want_false = index % 2 == 1
        statement, is_true = _perturb_for_true_false(sentence, rng, force_false=want_false)
        items.append({
            "type": "true_false",
            "statement": statement,
            "answer": is_true,
            "explanation": f"Original sentence: \u201c{sentence}\u201d",
        })
    return items


def build_reviewer(text: str, sentences: list[str], definitions: list[dict[str, str]],
                   terms: list[str], max_points_per_section: int = 5) -> dict[str, Any]:
    """Assemble a revision sheet: headings, key points, and a glossary.

    Sections come from markdown headings or ALL-CAPS lines when the source has them. Each
    section keeps **its own** points, picked from the paragraphs underneath that heading —
    slicing one ranked list across sections would attribute sentences to the wrong topic.
    """
    blocks: list[dict[str, Any]] = []
    current = {"heading": None, "body": []}

    lines = text.splitlines()
    for index, line in enumerate(lines):
        title = heading_title(lines, index)
        if title:
            if current["body"] or current["heading"]:
                blocks.append(current)
            current = {"heading": title, "body": []}
            continue
        if line.strip():
            current["body"].append(line.strip())
    if current["body"] or current["heading"]:
        blocks.append(current)

    blocks = [block for block in blocks if block["body"] or block["heading"]]
    if not blocks:
        blocks = [{"heading": "Overview", "body": [text]}]

    sections: list[dict[str, Any]] = []
    for block in blocks:
        body = "\n".join(block["body"])
        block_sentences = split_sentences(body)
        # Rank by information density so boilerplate lines do not become key points.
        def density(sentence: str) -> float:
            words = sentence.split()
            if not words:
                return 0.0
            content = [w for w in words if w.lower().strip(".,;:") not in _STOPWORDS]
            return len(content) / len(words)

        ranked = sorted((s for s in block_sentences if 5 <= len(s.split()) <= 45),
                        key=density, reverse=True)
        heading = block["heading"] or (_guess_title(body) if body else "Overview")
        sections.append({"heading": heading, "points": ranked[:max_points_per_section]})

    if not sections:
        sections = [{"heading": "Overview", "points": sentences[:max_points_per_section]}]

    glossary = [{"term": entry["term"], "definition": entry["definition"]}
                for entry in definitions[:25]]

    return {
        "title": sections[0]["heading"] if sections else _guess_title(text),
        "sections": sections,
        "glossary": glossary,
        "key_terms": terms[:25],
    }


def _guess_title(text: str) -> str:
    """Use the first heading, or a short leading phrase, as the document title.

    A bare first sentence makes a poor title ("Photosynthesis is the process by which w"),
    so sentence-looking lines are shortened at a word boundary and marked with an ellipsis.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        title = heading_title(lines, index)
        if title:
            return title[:80]
        stripped = line.strip()
        if not stripped:
            continue
        words = stripped.split()
        if len(words) > 8 and stripped.endswith((".", "!", "?")):
            return " ".join(words[:6]) + "…"
        return stripped[:80]
    return "Study material"


def generate_study_pack(
    text: str,
    flashcards: int = 20,
    quiz_items: int = 10,
    true_false_items: int = 8,
    seed: int = 7,
    engine: str = "local",
) -> dict[str, Any]:
    """Build every study artefact from one block of text.

    Args:
        text: Notes, lecture transcript, or OCR output.
        flashcards: Maximum number of flashcards.
        quiz_items: Maximum number of multiple-choice questions.
        true_false_items: Maximum number of true/false items.
        seed: Makes shuffling reproducible, so the same notes give the same paper.
        engine: Which generator to use.

            * ``"local"`` (default) — the extractive generator in this module. Runs offline,
              invents nothing, and is the only engine available without configuration.
            * ``"llm"`` — ask a language model for a generated pack. Requires
              ``BERN_LLM_API_KEY``; raises rather than silently degrading.
            * ``"auto"`` — use the language model when one is configured, otherwise the
              local generator. This is the only value that never fails.

    Returns:
        ``{"reviewer", "flashcards", "quiz", "mock_exam", "source_stats", "engine"}``. The
        mock exam is a mixed paper with its own answer key.
    """
    if engine in ("llm", "auto"):
        from bern import llm

        if engine == "llm" or llm.is_available():
            pack = llm.generate_study_pack(
                text,
                flashcards=flashcards,
                quiz_items=quiz_items,
                true_false_items=true_false_items,
            )
            pack.setdefault("seed", seed)
            return pack
        # engine == "auto" with no model configured: fall through to the local generator.

    sentences = split_sentences(text)
    definitions = extract_definitions(sentences)
    terms = key_terms(text, sentences, limit=40)

    cards = build_flashcards(definitions, terms, sentences, limit=flashcards)
    question_pool = build_quiz(definitions, terms, sentences, limit=quiz_items, seed=seed)
    true_false = build_true_false(sentences, limit=true_false_items, seed=seed)
    reviewer = build_reviewer(text, sentences, definitions, terms)

    mock_exam = {
        "title": f"Mock exam — {reviewer['title']}",
        "sections": [
            {"name": "Part I — Multiple Choice", "instructions": "Choose the best answer.",
             "items": question_pool},
            {"name": "Part II — True or False", "instructions": "Write TRUE or FALSE.",
             "items": true_false},
        ],
        "answer_key": [
            {"number": index + 1, "answer": q["answer"], "explanation": q["explanation"]}
            for index, q in enumerate(question_pool)
        ] + [
            {"number": len(question_pool) + index + 1,
             "answer": "TRUE" if item["answer"] else "FALSE",
             "explanation": item["explanation"]}
            for index, item in enumerate(true_false)
        ],
    }

    return {
        "reviewer": reviewer,
        "flashcards": cards,
        "quiz": question_pool,
        "mock_exam": mock_exam,
        "engine": "local",
        "source_stats": {
            "characters": len(text),
            "sentences": len(sentences),
            "definitions_found": len(definitions),
            "terms_found": len(terms),
        },
        "notes": [
            "Generated extractively from the supplied text — no content was invented.",
            "Questions are only as good as the source: notes without definitions or "
            "distinct terminology produce weaker items.",
        ],
    }


def main() -> None:
    """``python3 -m bern.study FILE`` — extract a document and build a study pack."""
    parser = argparse.ArgumentParser(description="Build study material from notes or a lecture file")
    parser.add_argument("file", type=Path, help="PDF, DOCX, image, or text file")
    parser.add_argument("--out", type=Path, help="Write JSON here instead of stdout")
    parser.add_argument("--flashcards", type=int, default=20)
    parser.add_argument("--quiz", type=int, default=10)
    args = parser.parse_args()

    from bern.text_extract import extract_path

    extracted = extract_path(args.file)
    if not extracted["text"].strip():
        raise SystemExit(
            "No text found. If this is a scanned PDF, export the pages as images and use "
            "the handwriting endpoint instead."
        )

    pack = generate_study_pack(extracted["text"], flashcards=args.flashcards, quiz_items=args.quiz)
    pack["source_stats"].update({"source_type": extracted["source_type"], "engine": extracted["engine"]})

    payload = json.dumps(pack, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.out} — {len(pack['flashcards'])} cards, "
              f"{len(pack['quiz'])} quiz items, {len(pack['reviewer']['sections'])} reviewer sections")
    else:
        print(payload)


if __name__ == "__main__":
    main()
