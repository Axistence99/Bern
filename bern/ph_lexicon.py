"""Filipino, Taglish and English word lists, plus the spelling corrector that uses them.

Why this module exists: Bern's handwriting backend reads characters one at a time and
EMNIST has no notion of words. A correct ``g`` can still come back as ``9``, a ``0`` as
``O``, a ``1`` as ``l``. Nothing at the character level can fix that — but a dictionary
can, because ``p1s0`` is not a word and ``piso`` is.

Two jobs are done here:

* **:class:`Lexicon`** — holds a vocabulary and answers "is this a word, and how common is
  it?" It blends a hand-written core of Filipino function words and English basics with
  the domain vocabulary Bern already ships (prices, nutrition, SKU names), so food- and
  price-related notes get much better correction than generic ones.

* **:func:`decode_word`** — beam search over the per-glyph probability matrix produced by
  the handwriting backend. It finds the highest-scoring letter sequence that is also a
  known word, which is far more robust than taking the top-1 character at each position.

Everything is dependency-free and deliberately small. The lists are not exhaustive; they
are chosen to cover classroom notes, lecture material and Filipino day-to-day text.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# --------------------------------------------------------------------------------------
# Core vocabulary
# --------------------------------------------------------------------------------------

#: High-frequency Filipino words: pronouns, particles, question words, verbs, everyday
#: nouns. Function words matter most for correction because they are short and ambiguous
#: ("ang", "ng", "sa" are one and two characters long).
FILIPINO_WORDS = """
ang ng mga sa na at ay o pero kung dahil kaya para upang nang pa lang din rin ba po opo
nga daw raw pala sana muna naman kasi tapos saka pati maging bilang bukod tanging
ako ikaw ka siya kami tayo kayo sila ko mo niya namin natin ninyo nila akin iyo kaniya
amin atin inyo kanila ko
ano sino saan kailan bakit paano magkano ilan alin gaano kumusta nasaan alinman
may meron wala hindi oo opo huwag wag pwede puede dapat gusto ayaw alam ayoko
isa dalawa tatlo apat lima anim pito walo siyam sampu labing-isa labindalawa dalawampu
tatlumpu apatnapu limampu animnapu pitumpu walumpu siyamnapu daan daang libo milyon
unang pangalawa pangatlo huli susunod
tao bata lalaki babae nanay tatay ina ama anak kapatid kaibigan guro maestro estudyante
mag-aaral paaralan eskwelahan klase silid silid-aralan libro papel lapis bolpen mesa
upuan pinto bintana daan lungsod bayan baryo nayon probinsya bundok ilog dagat langit
araw gabi umaga tanghali hapon oras minuto segundo linggo buwan taon panahon
pera piso presyo tinda palengke tindahan merkado sari-sari pagkain tubig kanin bigas
ulam gulay prutas karne isda manok baboy baka itlog gatas asukal asin mantika kape
tinapay sopas sabaw adobo sinigang kare-kare sisig pares bopis lomi pansit lumpia
bibingka puto kutsinta suman leche-plan turon halaya
bawang sibuyas kamatis patatas repolyo kalabasa talong okra ampalaya malunggay kangkong
pechay mustasa saging mangga papaya bayabas santol lansones rambutan durian pinya
niyog buko tubo mais gabi kamote
trabaho opisina trabahador amo sweldo sahod kita gastos budget tipid matipid
aral pag-aaral pagtatapos pagsusulit eksamen test proyekto takdang-aralin report
tanong sagot halimbawa kahulugan katuturan paliwanag dahilan resulta epekto sanhi
layunin metodo paraan proseso sistema datos impormasyon kaalaman kasanayan
mabuti masama maganda pangit malaki maliit mataas mababa mahaba maikli malawak makitid
mabilis mabagal mainit malamig masarap matamis maasim maalat mapait mahal mura bago
luma malinis marumi madami marami kaunti lahat iba pareho totoo mali tama
siguro tiyak tiyak malinaw malabo madali mahirap mahalaga simple kumplikado
salamat pasasalamat paumanhin pasensya kumusta ingat paalam maligayang bati tuloy
sulat sulatin letra salita pangungusap talata kwento tula awit pelikula larawan
kaisipan damdamin pag-asa takot galit lungkot tuwa saya pagmamahal
buhay kamatayan kalusugan sakit gamot doktor nars ospital
bayan bansa wika kultura kasaysayan heograpiya agham matematika sining musika
magluto lutuin kainin inumin tikman bilhin bili benta bayad magbayad sukli
basahin isulat tingnan pakinggan sabihin sagutin tanungin aralin pag-aralan
turuan matuto tandaan isipin hanapin gawin gumawa takbo lakad upo tayo bangon
""".split()

#: Words that are spelled the same in Filipino and English, or that Taglish writers mix in
#: constantly. Kept separate so the source of a match is clear when debugging.
TAGLISH_COMMON = """
ang mga ng sa na at ay pero kaya para dahil tapos siyempre siguro actually talaga
sobrang medyo parang ganoon ganito ganun ganyan doon dito diyan iyon iyan ito
magkano presyo sale libre check promo order budget sulit worth
okay ok sige cge thanks thank you sorry please hello hi bye
teacher student project group report quiz exam reviewer notes lecture module
activity assignment deadline schedule class school college university
cellphone load internet wifi laptop charger battery kuryente tubig bill
jeep tricycle bus mrt lrt grab taxi pamasahe traffic
""".split()

#: English function words and classroom vocabulary. Short, high-frequency items only —
#: a full English dictionary would fight the pattern-based correction rather than help it.
ENGLISH_WORDS = """
a an the and or but if then than that this these those there their them they
i you he she it we they me him her us my your his its our
is are was were be been being am do does did done have has had having
will would shall should can could may might must not no yes
of to in on at by for with from into onto over under about above below between
as so such very too also just only even still yet already again ever never always
what when where which who whom whose why how many much more most less least
all any both each every few some none other another same different
here there now later before after during while until since
one two three four five six seven eight nine ten first second third last next
time day week month year hour minute second today tomorrow yesterday
study learn teach lesson lecture class course subject topic chapter section unit
note notes quiz exam test question answer problem solution example exercise
review reviewer summary outline definition term concept idea theory method
data result value number figure table graph chart image picture diagram
system process function model variable sample size measure rate level
student teacher school college university professor instructor
science math mathematics history language english filipino computer program
read write speak listen explain describe compare contrast define identify
list name state give show find make take use work call need want know think
book page word sentence paragraph text line title heading list mark
energy force motion light heat sound water air earth plant animal cell
body health food nutrition price cost value market shop store money
""".split()

#: Technical words that show up in scanned lecture notes. Small on purpose: the goal is
#: to bias correction toward plausible classroom vocabulary, not to model English.
ACADEMIC_WORDS = """
analysis analyses approach assumption conclusion condition consequence context
criteria criterion dataset definition derivation description discussion
distribution element equation estimate evaluation evidence experiment
explanation factor formula framework hypothesis implementation implication
interpretation introduction limitation measure mechanism observation
objective outcome parameter percentage period principle probability
procedure proportion property relationship representation requirement
research response sample significance statement strategy structure
technique tendency theory variable verification summary abstract appendix
chapter citation conclusion reference bibliography source quote
algebra geometry calculus statistics physics chemistry biology anatomy
grammar syntax vocabulary comprehension composition literature
revolution colony treaty government constitution democracy economy
""".split()


# --------------------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zñáéíóúü]+")


def strip_accents(text: str) -> str:
    """Remove diacritics but keep ``ñ``, which is a distinct letter in Filipino."""
    if "ñ" in text and len(text) < 200:
        # Protect ñ while normalising everything else, then restore it.
        placeholder = "\x00"
        text = text.replace("ñ", placeholder).replace("Ñ", placeholder)
        normalised = unicodedata.normalize("NFKD", text)
        normalised = "".join(ch for ch in normalised if not unicodedata.combining(ch))
        return normalised.replace(placeholder, "ñ")
    normalised = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalised if not unicodedata.combining(ch))


def normalize_word(word: str) -> str:
    """Lowercase and strip accents, keeping ``ñ``, for dictionary lookup."""
    return strip_accents(word.lower()).strip()


class Lexicon:
    """A set of known words with frequency weights, built for spelling correction.

    Membership alone is enough to fix ``p1s0`` → ``piso``. Weights matter for the
    tie-breaks: ``ang`` appears constantly and should beat a rare word that happens to
    have the same edit distance.

    Load order is deliberate — the hard-coded core first, then Bern's own datasets, which
    add food and price vocabulary nobody wants to maintain by hand.
    """

    def __init__(self, words: Iterable[str] | None = None) -> None:
        self.weights: dict[str, float] = {}
        if words is not None:
            self.add_words(words)
        else:
            self.add_words(FILIPINO_WORDS, weight=3.0)
            self.add_words(TAGLISH_COMMON, weight=2.5)
            self.add_words(ENGLISH_WORDS, weight=2.0)
            self.add_words(ACADEMIC_WORDS, weight=2.0)
            self._add_dataset_words()

    # -- construction ------------------------------------------------------------------

    def add_words(self, words: Iterable[str], weight: float = 1.0) -> None:
        """Insert words, keeping the highest weight when a word appears twice."""
        for word in words:
            key = normalize_word(str(word))
            if not key or len(key) > 32:
                continue
            self.weights[key] = max(self.weights.get(key, 0.0), weight)

    def _add_dataset_words(self) -> None:
        """Harvest vocabulary from Bern's bundled PH datasets.

        These files are already in the repo for the price and nutrition experts, so the
        lexicon gains hundreds of real Filipino food and product names — "malunggay",
        "sibuyas", "kangkong" — without a hand-maintained list.
        """
        harvested: set[str] = set()
        for name in ("ph_prices.json", "ph_nutrition_curated.json", "ph_sku_catalog.json"):
            path = DATA_DIR / name
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            harvested.update(self._walk_strings(payload))
        self.add_words(harvested, weight=2.0)

    @classmethod
    def _walk_strings(cls, node: Any, depth: int = 0) -> set[str]:
        """Collect alphabetic words from an arbitrarily nested JSON structure."""
        found: set[str] = set()
        if depth > 6:
            return found
        if isinstance(node, str):
            found.update(t.lower() for t in _TOKEN_RE.findall(strip_accents(node.lower())))
        elif isinstance(node, dict):
            for value in node.values():
                found |= cls._walk_strings(value, depth + 1)
        elif isinstance(node, list):
            for item in node:
                found |= cls._walk_strings(item, depth + 1)
        return {w for w in found if 2 <= len(w) <= 24}

    # -- queries -----------------------------------------------------------------------

    def __contains__(self, word: str) -> bool:
        return normalize_word(word) in self.weights

    def __len__(self) -> int:
        return len(self.weights)

    def weight(self, word: str) -> float:
        return self.weights.get(normalize_word(word), 0.0)

    def correct_enie(self, word: str) -> str:
        """Snap a word that is one letter away from a ``ñ`` spelling onto the ``ñ`` form.

        EMNIST cannot produce ``ñ`` — it is not one of its 62 classes — so "niño" arrives
        as "nino". If the plain form is unknown but the ``ñ`` form is in the lexicon, the
        ``ñ`` form is almost certainly what was written.
        """
        candidate = normalize_word(word)
        if candidate in self.weights:
            return candidate
        if "n" in candidate:
            variant = candidate.replace("n", "ñ")
            if variant in self.weights:
                return variant
            for index in range(len(candidate)):
                if candidate[index] == "n":
                    variant = candidate[:index] + "ñ" + candidate[index + 1:]
                    if variant in self.weights:
                        return variant
        return candidate

    def suggest(self, word: str, max_distance: int = 1, limit: int = 5) -> list[str]:
        """Return lexicon words within ``max_distance`` edits, cheapest first.

        Only used by :func:`bern.study` for term cleanup; the main decoder does not need
        it because it searches the probability lattice instead of repairing a fixed string.
        """
        target = normalize_word(word)
        if not target:
            return []
        if target in self.weights:
            return [target]
        if max_distance <= 0:
            return []

        # Restrict the candidate pool by length so this stays cheap on big lexicons.
        pool = [w for w in self.weights if abs(len(w) - len(target)) <= max_distance]
        scored: list[tuple[int, float, str]] = []
        for candidate in pool:
            distance = _edit_distance(target, candidate, max_distance)
            if distance <= max_distance:
                scored.append((distance, -self.weights[candidate], candidate))
        scored.sort()
        return [word for _, _, word in scored[:limit]]


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Levenshtein distance with early exit once ``limit`` is exceeded."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        best_in_row = i
        for j, char_b in enumerate(b, 1):
            cost = 0 if char_a == char_b else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            best_in_row = min(best_in_row, value)
        if best_in_row > limit:
            return limit + 1
        previous = current
    return previous[-1]


@lru_cache(maxsize=1)
def get_lexicon() -> Lexicon:
    """The shared lexicon instance. Building it is fast but not free, so it is cached."""
    return Lexicon()


# --------------------------------------------------------------------------------------
# Word decoding
# --------------------------------------------------------------------------------------

def decode_word(
    glyph_probabilities: Sequence[Sequence[tuple[str, float]]],
    lexicon: Lexicon | None = None,
    beam_width: int = 24,
    lexicon_bonus: float = 6.0,
    length_penalty: float = 0.05,
) -> tuple[str, float]:
    """Find the most likely *word* behind a column of per-glyph character probabilities.

    Each glyph gives a ranked list of ``(character, probability)``. Taking the top
    character independently maximises per-character accuracy but produces strings that no
    language contains. Beam search instead optimises the whole word:

    ``score = sum(log p(char)) + lexicon_bonus · [word ∈ lexicon] − length_penalty · len``

    The lexicon bonus is what repairs ``p1s0`` → ``piso``: the letters are individually
    plausible, but only one resulting string is a real word. The length penalty stops the
    search preferring long improbable words purely for containing a known prefix.

    Args:
        glyph_probabilities: One ranked ``(char, prob)`` list per segmented glyph.
        lexicon: Vocabulary to score against. Defaults to :func:`get_lexicon`.
        beam_width: Hypotheses kept alive per position. 24 is plenty for words.
        lexicon_bonus: Log-space reward for landing on a known word.
        length_penalty: Per-character cost, which discourages padding.

    Returns:
        ``(word, mean_confidence)`` where confidence is the geometric mean probability of
        the chosen characters, in ``[0, 1]``.
    """
    if not glyph_probabilities:
        return "", 0.0

    lexicon = lexicon if lexicon is not None else get_lexicon()

    # (score, text, accumulated_log_prob, char_count)
    beams: list[tuple[float, str, float, int]] = [(0.0, "", 0.0, 0)]

    for options in glyph_probabilities:
        if not options:
            continue
        expanded: list[tuple[float, str, float, int]] = []
        for score, text, log_prob, count in beams:
            for char, prob in options:
                p = max(float(prob), 1e-6)
                expanded.append((score + math.log(p), text + char, log_prob + math.log(p), count + 1))
        expanded.sort(key=lambda item: item[0], reverse=True)
        beams = expanded[:beam_width]

    scored: list[tuple[float, str, float, int]] = []
    for score, text, log_prob, count in beams:
        final = score
        if normalize_word(text) in lexicon:
            final += lexicon_bonus
        final -= length_penalty * count
        scored.append((final, text, log_prob, count))
    scored.sort(key=lambda item: item[0], reverse=True)

    _, best_text, best_log_prob, best_count = scored[0]
    confidence = float(math.exp(best_log_prob / best_count)) if best_count else 0.0
    return best_text, confidence


def looks_like_word(token: str, lexicon: Lexicon | None = None) -> bool:
    """True when ``token`` is a known word after accent normalisation."""
    lexicon = lexicon if lexicon is not None else get_lexicon()
    return normalize_word(token) in lexicon
