# HANDOFF — Bern handwriting feature

Everything a new chat needs to resume this work. Written to survive a context reset:
**the code lives on GitHub, the decisions live in this file.**

---

## 1. Thirty-second version

| | |
|---|---|
| Repo | `github.com/Axistence99/Bern` (public) |
| Branch | **`main` only** (owner-set, reinforced twice: no branches, no PRs) |
| Commits | `9ec7c30` OCR, `dbf722e` handoff, `21a72c6` AGENTS.md, plus the multilingual + study work |
| Author | **`Axistence99 <272527416+Axistence99@users.noreply.github.com>`** |
| Tests | 52 passing (`python3 -m pytest tests/ -q`) |
| New files | `bern/handwriting.py`, `bern/ph_lexicon.py`, `bern/text_extract.py`, `bern/study.py` |
| Modified | `bern/engine.py`, `server.py`, `data/bern_dataset.json`, `README.md` |
| Intents | 24 total (added `handwriting_ocr`, `study_materials`, `file_scan`) |
| Model | `models/handwriting_glyph_model.joblib` (252 KB, committed) |
| Push target | `origin main` |

**Nothing is uncommitted.** `git status` is clean.

---

## 2. Repository access from a new chat

The sandbox is fresh per chat, so the SSH deploy key must be regenerated:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "arena-agent-bern" -f ~/.ssh/bern_deploy_ed25519 -N "" -q
cat ~/.ssh/bern_deploy_ed25519.pub      # paste into GitHub → repo → Settings → Deploy keys
```

The user adds that public key at:
`github.com/Axistence99/Bern` → **Settings** → **Deploy keys** → **Add deploy key** → ☑️ **Allow write access**

⚠️ **The previous deploy key (`arena-agent-bern-docs`, fingerprint
`SHA256:bje/maGyx4080WJtvp1Hr1izK7nya8QKK1PqThugMDU`) is dead** — its private half did not
survive. Delete it from the repo's Deploy keys page if it is still listed.

Then:

```bash
git clone https://github.com/Axistence99/Bern.git bern && cd bern
git remote set-url origin git@github.com:Axistence99/Bern.git
git config user.name  "Axistence99"
git config user.email "272527416+Axistence99@users.noreply.github.com"
```

**Commit identity and branch are owner-set rules (see `AGENTS.md` §6):** author must be
`Axistence99`, and all pushes go to **`main` only** — no feature branches, no pull requests,
never create a branch. `git config` does not
survive a snapshot restore, so re-run those two lines before committing.

**Gotcha:** `.git/config` is excluded from workspace snapshots. In the same chat the remote
may vanish — re-run `git remote add origin git@github.com:Axistence99/Bern.git`.
Read-only clone/pull over HTTPS always works without any key.

---

## 3. Standing instructions from the user (do not lose these)

> **Verbatim source:** [`AGENTS.md`](AGENTS.md) at the repo root holds the owner's exact
> wording. The summary below is a convenience index — **when the two disagree, `AGENTS.md`
> wins.** Read that file first.

The user sent four instruction blocks. They are permanent unless the user revises them.

### 3a. Universal vibe coding rules (§0)
- Inspect the existing project before changing anything. Do not blindly rewrite.
- Don't remove functionality unless asked. Reuse existing components/utilities/styles.
- Make the smallest reasonable change. Preserve architecture unless there's a clear reason.
- Don't add dependencies unless genuinely necessary. Don't duplicate existing functionality.
- No unnecessary abstraction or overengineering. Preserve existing UI/UX unless required.
- After changes: check syntax, imports, runtime errors, the requested feature, that existing
  functionality still works, responsive behaviour, accessibility, security, performance.
- **Never claim something was tested if it wasn't.**
- Priority order: correctness → security → maintainability → accessibility → performance →
  UX → visual polish.

### 3b. Documentation (§1)
Produce docs describing the **actual** implementation, not an assumed one: overview, features,
stack, structure, file responsibilities, install/setup, env vars, run, build, test, API
endpoints, DB structure, deployment, dev notes, known limitations.
**Do not invent features, commands, files, APIs or configuration that do not exist.**

### 3c. Human-understandable code (§2)
Improve names, file organisation, function structure, complex conditions, repeated logic.
Comments only where useful (complex algorithms, non-obvious business logic, security-sensitive
logic, workarounds, external API behaviour). No comments that restate the code.
**Behaviour must stay unchanged** unless fixing a discovered bug.
*(Note: §3d forbids refactoring. §3c was never started — see pending work.)*

### 3d. Line-by-line walkthrough spec
Full template saved at `docs/CODE-WALKTHROUGH-SPEC.md`. Key constraints: explain the *why*;
assume a beginner; use exact names; **do not invent behaviour**; do not modify/refactor outside
the "Potential Issues and Improvements" section; output only Markdown.

### 3e. Absolute constraint
**Never run a sandbox server.** No long-lived processes, no dev servers. One-shot commands only.
This means HTTP handlers can be statically checked and unit-tested but **not exercised over a
socket** unless the user lifts the rule.

---

## 4. What was built

### Pipeline
```
photo → grayscale → Otsu/adaptive binarise → deskew → lines → words → glyphs → classify → text
```

### Model
`SGDClassifier(loss="log_loss", alpha=1e-5, average=True)`, 5 epochs, all **697,932 EMNIST
`byclass`** characters (62 classes: `0-9`, `A-Z`, `a-z`), streamed via `partial_fit` so it fits
in ~1 GB RAM. Feature vector is 341-dim: 324 HOG + 16 zoning + 1 ink ratio.

### Measured numbers (do not round these up)
| Metric | Result |
|---|---|
| Held-out character accuracy, 30,000 unseen EMNIST glyphs | **82.55%** |
| End-to-end char accuracy, clean composed lines (20 words) | 75.3% |
| End-to-end char accuracy, ±9° rotation + blur + scale jitter | 68.0% |
| Word accuracy, same set | 4/20 clean, 5/20 jittered |

### Integration points
- `POST /api/handwriting` — multipart `image` **or** JSON `image_base64`; optional `backend`,
  `case_mode`. Mirrors `CountMoneyHandler`'s two input shapes.
- `BernAI.read_handwriting(image_bytes, backend, case_mode)` — lazy `HandwritingReader`.
- New chat intent `handwriting_ocr` (dataset now 22 intents, routes at 99.77%).
- `HandwritingReader(backend="auto"|"glyph"|"trocr"|"tesseract", case_mode="auto"|"upper"|"lower")`

---

## 5. The five bugs found while training (don't re-introduce them)

1. **`deskew()` rotated the mask but not the image.** Every crop box was applied to
   misaligned pixels — worth ~15 points. Fixed by rotating both with one matrix;
   `deskew(mask, parallel)` returns a tuple. Regression test:
   `test_deskew_rotates_mask_and_parallel_together`.
2. **2×2 morphological opening in `ink_mask()`** erased thin strokes. An `I` stem is 2 px wide
   at 28 px. Cost ~6 points. Removed; specks are dropped by the minimum-line-height filter.
3. **Centroid-shifting glyphs in `normalize_glyph()`** — looked tidy, cost 3.5 points. EMNIST
   already centred its glyphs. Removed.
4. **Scale mismatch** — EMNIST glyphs fill 86% of the frame. Normalising to 80% cost 1–5
   points. Swept 4×3×2 combos; 0.86 + linear + no centroid shift won.
5. **Case twins split confidence three ways** (`O`/`o`/`0`, `S`/`s`, `I`/`l`/`1`). Pooling the
   two letter cases at predict time is worth ~5 points, no retraining.

---

## 6. Environment facts worth knowing

- Python 3.13.18, numpy 2.3.5, scikit-learn 1.6.1, OpenCV 4.11.0, Tornado 6.6, pytest 9.0.3.
- **No torch / transformers / pytesseract installed.** The TrOCR and Tesseract backends are
  written, documented, and import-guarded, but **unverified**. Don't claim otherwise.
- RAM is ~1 GB, 2 CPU cores, 25 GB disk. This is why training streams instead of loading
  matrices.
- EMNIST is **not** committed (536 MB zipped, 190 MB extracted for `byclass` alone). `~/.cache`
  does not survive a new chat. Re-fetch (~10 s). **Do not use `/tmp`** — it is a 992 MB tmpfs and
  the archive will fail with "write error (disk full?)". Use a path on `/`, which has ~20 GB:
  ```bash
  mkdir -p ~/emnist_dl && cd ~/emnist_dl
  curl -sL -o gzip.zip https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip   # 536 MB
  unzip -o -q gzip.zip 'gzip/emnist-byclass-*'                               # extract only byclass
  mkdir -p byclass && mv gzip/emnist-byclass-* byclass/ && rm -rf gzip gzip.zip
  ```
  Then, from the repo root:
  ```bash
  python3 scripts/e2e_handwriting_eval.py --data-dir ~/emnist_dl   # 75.3% / 68.0%, 4/20, 5/20
  python3 -m bern.handwriting --evaluate --data-dir ~/emnist_dl
  ```
  Delete `~/emnist_dl` afterwards — 194 MB of it would otherwise count against the workspace
  snapshot budget.
  The committed model means **no retraining is needed** to use the feature.
- NIST returns **403 to urllib's default User-Agent** — the downloader sets one explicitly.

---

## 7. Verified vs not verified

**Actually tested:**
- All 22 unit tests pass.
- `BernAI.chat("read my handwriting")` → `handwriting_ocr` @ 0.9977; price and nutrition
  intents still route correctly.
- `BernAI.read_handwriting()` on a real composed image → `'B 1 GAS'` @ 0.842.
- End-to-end accuracy harness against real EMNIST glyphs (the table in §4).
- Synthesis checks (`ast.parse`) on all edited Python files.

**Not tested — do not claim otherwise:**
- `POST /api/handwriting` over HTTP. Never served (see §3e). Handler logic is the same shape as
  the working `CountMoneyHandler`, but the route itself is unexercised.
- TrOCR / Tesseract backends (dependencies absent).
- Real handwriting photographed with a phone: only EMNIST ink composed into lines was tested.
  A real photo adds perspective, shadows, ruled lines, and uneven spacing — all untested.
- Front-end integration: `web/` and `react-native/` **do not exist in the repo**, although the
  README describes them. There is no UI for this feature.

---

## 7b. Round two — Filipino/Taglish and study materials

### Filename map
| File | Responsibility |
|---|---|
| `bern/ph_lexicon.py` | ~2,900-word Filipino/English/Taglish dictionary; `decode_word()` beam search |
| `bern/text_extract.py` | PDF / DOCX / image / text → `{text, source_type, engine, ...}`; stdlib only |
| `bern/study.py` | reviewer, MCQ quiz, mock exam, flashcards from source text |

### Measured results (reproduce with `scripts/e2e_handwriting_eval.py`)
| Corpus | Raw model | With dictionary |
|---|---|---|
| English clean | 81.0% | 84.5% |
| Filipino clean | 77.2% | **91.3%** |
| Taglish clean | 76.6% | 88.3% |
| All, char accuracy | 72.5% | **83.6%** |
| All, exact words | 8/82 | **29/82** |

Tuned by measurement, not guesswork: `lexicon_bonus` swept 2/4/6/8/12 → 6.0 won;
`top_k` swept 5/8/12 → no difference past 5, so 5 is kept for cost.

### Bugs found this round (do not re-introduce)
1. **All-caps OCR lines eaten as headings.** `BIGAS 50 PISO` matched the caps-heading regex
   and vanished from the text. Heading detection is now *structural*: a caps line counts only
   if it is isolated by blank lines above and below and has ≤ 8 words.
2. **Headings glued onto sentences**, producing glossary terms like
   "Cell Respiration Cellular respiration". Fixed with `_logical_lines()`, which rejoins
   hard-wrapped lines but keeps a capitalised line after an unterminated one as a boundary.
3. **Reviewer points sliced globally.** Sentences were cut from one ranked list and handed out
   by index, so a Cell Respiration section received Photosynthesis points. Now each section
   ranks only its own body text.
4. **True/false was 7-of-8 TRUE.** Random flipping is not balance; items now alternate
   deterministically.
5. **Distractors were verbs.** Options came from raw keywords ("nagbebenta"); definition
   terms are now preferred as distractors.

### Environment changes
- `/tmp` is a **992 MB tmpfs** — EMNIST will not fit. Use `~/emnist_dl` on `/` (~20 GB).
- `pypdf` is **not** installed; `python-docx` was installed once but **does not persist**.
  That is why both extractors have stdlib fallbacks. Do not add them as hard dependencies.

### Verified this round
- 52 tests pass.
- PDF (built by hand), DOCX (built by hand), MD and TXT all extract and produce study packs.
- Filipino notes produce a correct glossary: nutrisyon, protina, carbohydrates, bigas, adobo.
- 8/8 intent routing correct including the two new intents; no regression in price/nutrition.
- Both end-to-end accuracy tables above, re-measured on this machine.

### NOT verified this round
- The new HTTP routes (`/api/extract`, `/api/study`) over a socket — still no server allowed.
- Real photographed handwriting; only composed EMNIST ink.
- `pypdf` path and the TrOCR/Tesseract backends (dependencies absent).

---

## 7c. Round three — Tier 1 to Tier 4, and what is actually measured

Scope: make Bern measurably better, with the honest acknowledgement that a 21k-parameter
model cannot read meaning. Local fixes were done first; anything needing a language model is
opt-in and keeps an offline fallback.

### What changed

| File | Change |
| --- | --- |
| `bern/symbols.py` | **New.** 28-class symbol model + rule-based `classify_punctuation`. |
| `bern/preprocess.py` | **New.** Perspective correction, ruled-line removal, column detection. |
| `bern/handwriting.py` | Symbol layer, photo mode, mark merging, prefix-constrained decode. |
| `bern/ph_lexicon.py` | Prefix index and prefix bonus in `decode_word`. |
| `models/symbol_model.joblib` | **New.** 28 classes, 117 KB. |
| `models/symbol_metrics.json` | **New.** Measured scores for the symbol model. |
| `scripts/bench_handwriting.py` | **New.** End-to-end benchmark including marks. |
| `tests/test_symbols.py`, `tests/test_mark_assembly.py` | **New.** 28 tests. |

### Measured results (reproduce with the commands given)

`PYTHONPATH=. python3 scripts/e2e_handwriting_eval.py --data-dir ~/emnist_dl`
(dictionary on, photo mode on):

| split | clean chars | clean exact words | jittered chars | jittered exact words |
| --- | --- | --- | --- | --- |
| english | 91.7% | 11/15 | 82.1% | 7/15 |
| filipino | 93.5% | 13/16 | 80.4% | 8/16 |
| taglish | 94.2% | 7/10 | 89.1% | 3/10 |

`PYTHONPATH=. python3 scripts/bench_handwriting.py --cases 40` (words plus `. , - ?`):

| condition | char accuracy | exact words |
| --- | --- | --- |
| clean | 88.1% | 36/68 |
| jittered | 82.1% | 22/59 |

Per-mark recall on 14 trials each (`?` and the marks are rendered with Noto, a font family
the symbol model never trained on):

| mark | exact | detected at all |
| --- | --- | --- |
| `.` | 10/14 | 14/14 |
| `,` | 8/14 | 11/14 |
| `-` | 10/14 | 14/14 |
| `?` | 7/14 | 10/14 |

Symbol model on 10 symbols across 3 DejaVu faces and 4 sizes it never saw: **100% (120/120)**;
on Noto CJK, also never trained on: **100% (42/42)**. The held-out number in
`models/symbol_metrics.json` (99.67%) was measured on a random split of the same renders, so
the cross-family figures above are the ones that mean anything.

### Bugs found this round (do not re-introduce)

1. **The symbol backend was silently dead.** It loaded lazily inside a path that only ran for
   single-glyph words, so `self._symbol_backend` stayed `None` for every real word. `read_image`
   now resolves it up front.
2. **`segment_words` split words containing narrow letters** (`BIGAS` became `B` + `IGAS`).
   Otsu on two to four gap samples is meaningless. The current rule uses a floor relative to
   the median gap plus a global line-height crossing check.
3. **A comma split into two components** and produced `,,`; a thin tail was also read as `;`.
   Marks are now merged by union box before classification, and `:`/`;` are no longer inferred
   from geometry at all — the learned model does them, and measurement showed the geometric
   guess was costing real commas.
4. **The baseline was dragged down by the mark being classified.** A comma is a descender by
   nature, so including it in the median made the comma stop looking like it hung below the
   baseline. Only letter-sized glyphs vote now.
5. **`cv2.threshold(..., THRESH_OTSU)` raises on float32.** Cast to `uint8` first.
6. **Mark polarity.** `compose` already inverts; a harness that inverts first paints marks as
   white gaps. Two hours of debugging came from this.
7. **A held-out score measured on the training distribution is not a generalisation claim.**
   The symbol model's 99.67% is real but only says "separates its own renders". The 100%
   cross-family numbers are the honest ones.

### Known limits, stated plainly

- `,` and `?` are the weakest marks. `,` depends on how far the tail crosses the baseline.
- Punctuation is only inferred for `. , -` from geometry. `: ; ' "` come from the symbol
  model and only where the reader already isolates them.
- Column detection requires three or more lines; it will not split a two-line page.
- Perspective correction needs a visible page outline. A photo cropped inside the page is
  passed through untouched.
- No real handwriting photograph has been tested. All photo-mode numbers come from synthetic
  pages. The code says so where it matters.

---

## 7d. Round three, part two — Tier 3 and Tier 4 (opt-in)

| File | Change |
| --- | --- |
| `bern/llm.py` | **New.** Opt-in language-model layer: study packs, tutoring sessions. |
| `bern/study.py` | `generate_study_pack(..., engine="local" \| "llm" \| "auto")`. Default unchanged. |
| `server.py` | `engine` accepted by `/api/study`; new `/api/tutor` with in-memory sessions. |
| `tests/test_llm.py` | **New.** 22 tests, all offline. |

### The contract

- Disabled unless `BERN_LLM_API_KEY` is set. `llm.is_available()` is the single check.
- `engine="local"` (the default) is byte-for-byte the old behaviour.
- `engine="auto"` uses the model when configured, else the local generator. Never raises.
- `engine="llm"` raises `LLMError` when no model is configured. Deliberate: a caller that
  asked for an explanation should not silently receive flashcards.
- `/api/tutor` returns 503 without a key rather than a scripted answer.
- Stdlib `urllib.request` only. No SDK, no new dependency.

### Reply handling

Model output is untrusted input. A quiz item whose `answer` is not among its own `options` is
dropped; flashcards need both sides; true/false needs a real boolean; an answer key is rebuilt
from the questions rather than trusted from the model. `_parse_json_object` accepts a bare
object, a fenced block, or an object wrapped in prose.

### Verified / not verified

Verified: everything in `tests/test_llm.py` — availability gating, the fallback contract, reply
cleaning, JSON extraction, mock-exam assembly, session reset. 102 tests pass.

**Not verified: the live HTTP call.** No API key exists here. The request body follows the
OpenAI chat-completions schema, but that has not been exercised against the service. Treat the
first real call as untested code.

---

## 8. Pending work, in priority order

1. ~~**Fix the README's phantom structure.**~~ **Partly fixed this round.** The worst case was
   `scripts/train_bern.py`, referenced four times and **never committed to this repo** — anyone
   following the README got `No such file or directory`. Full training now runs as
   `python3 -m bern.trainer` (real `argparse` CLI added; before this, `--help` silently started
   a training run). `requirements.txt` and `web/` genuinely exist; `react-native/` only partly.
   Remaining: re-audit the project-structure tree against the actual file listing.
2. **No `requirements.txt`** anywhere. Deps are unpinned and undiscoverable. Straightforward win.
3. **§2 readability pass** — never started, and it conflicts with §3d's "do not refactor".
   Ask the user which wins before touching code.
4. **§3d line-by-line walkthrough** — spec ready, never applied. `bern/handwriting.py` is the
   natural first target (983 lines, well-documented, fresh).
5. **Word-accuracy improvements still open:**
   - ~~lexicon/beam pass over glyph probabilities~~ — **done** (+11.1 points)
   - ~~punctuation~~ — **done for `. , - ?`**, see the per-mark table above
   - per-language case model: score the whole word in upper and lower, pick the better fit
   - **lexicon coverage is the binding limit.** Measured: a correct word outside the 2,936-word
     list cannot be recovered by any prior, because the score needed to win is never available.
     Expanding the Filipino/English list is worth more than any further model tuning.
   - finer vertical zoning / stroke-width features; retrain with `C`/`alpha` neighbours
6. **Real-photo robustness:** perspective, ruled lines and columns are implemented and unit
   tested, but **no real photograph has been run through them.** Next step is a handful of
   real phone photos; expect the thresholds in `bern/preprocess.py` to need adjusting.
7. **Cursive support** requires TrOCR (or similar CTC model). Character segmentation cannot
   solve it, by construction.
8. **Tier 3 and 4 are built, but only offline paths are verified.** `bern/llm.py` holds the
   opt-in layer and `/api/tutor` adds sessions. See §7d. The live HTTP call has never been
   made — no API key exists in this environment.

---

## 9. How to resume in one message

Paste this into the new chat:

> Continue the Bern project. Clone `github.com/Axistence99/Bern`, branch `main`. Read
> `AGENTS.md` and `HANDOFF.md` at the repo root first — they have the standing rules, the full
> state, verified-vs-unverified status, and pending work. Do not run a sandbox server. Then
> [your next request].

---

## 10. Files that exist outside the repo

These were workspace-only and will **not** survive into a new chat:
- `sample_handwriting.png` — demo image used in the final report
- `docs/CODE-WALKTHROUGH-SPEC.md` — the walkthrough template (§3d)
- `/tmp/harness.py` — the original throwaway accuracy harness. **Superseded** by
  `scripts/e2e_handwriting_eval.py`, which is committed and reproduces the same numbers.
- Extracted EMNIST under `~/emnist_dl/` — re-download with the recipe in §6.
