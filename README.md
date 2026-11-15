# Bern AI

**Bern** is a custom, trainable Python Machine Learning assistant built with `scikit-learn` and OpenCV, persisted via `joblib`, served over a CORS-enabled Tornado REST API, and ready to integrate into React Native / Expo apps.

It combines:
1. **Trainable Intent ML Engine (`bern/engine.py`, `bern/trainer.py`)** — Hybrid Word (1–2 n-gram) + Char-WB (3–5 n-gram) TF-IDF + calibrated `LogisticRegression`, OOD vocabulary gating, AST math evaluator, `Asia/Manila` clock, and live teach/retrain endpoints (`22` intents, `1,481` hand-written English/Taglish phrases + `398` generated ones, tuned by cross-validation — see *Training Bern*).
2. **Philippine Common Goods Price Expert (`bern/ph_prices.py`)** — Curated 2026 PH catalog (`64` common items across `9` categories), `1,836` searchable PH retail SKUs (`150` canned foods), `17`-region UN WFP food price medians (`244,835` rows through `2026-09-15`), DA daily/NCR weekly Bantay Presyo (`30,600` rows through `2026-11-10`), official DTI May 2026 BNPC SRP bulletin, and Taglish budget/basket calculator.
3. **Philippine Food & Nutrition Expert (`bern/ph_nutrition.py`)** — **19,594 foods** with calories, macros, sodium and more: **1,542 DOST-FNRI PhilFCT foods** (garlic, onion, milk, canned corned beef/sardines/luncheon meat ...), **13,577 USDA FoodData Central foods** (ground beef, fast-food chicken, McDonald's USA ...), **4,378 Philippine packaged products** from Open Food Facts (191 canned goods, 117 milk/dairy ...), **28 Filipino dishes + 3 meals** (adobo, pares, bopis, sisig, sinigang, kare-kare ...) and **66 McDonald's PH / Jollibee items**. Understands Taglish (`bawang`, `sibuyas`, `gatas`, `giniling na baka`, `kanin`), scales servings, totals whole meals and tells you how trustworthy each number is (see *Data quality* below).
4. **Philippine Peso Money Counter (`bern/money_vision.py`)** — Dual-mode OpenCV + `scikit-learn` (`VotingClassifier` over 93-dim centered ROI + color/texture/contour features) trained on `1,100` real Philippine banknote & coin photos (`RichMiguel/Banknotes-Coins`, MIT) plus synthetic tabletop lighting scenes (**95.45%** end-to-end accuracy on `220` unseen real photos).
5. **Handwriting-to-Text OCR (`bern/handwriting.py`)** — OpenCV layout analysis (Otsu/adaptive binarisation, deskew, projection-profile line/word/glyph segmentation) driving a scikit-learn **SGD glyph classifier trained on all 697,932 EMNIST `byclass` handwritten characters**: **82.55% held-out character accuracy** on 30,000 unseen characters. Optional TrOCR backend for cursive (`pip install transformers torch`). See *Handwriting* below for what it can and cannot read.
6. **Study Material Generator (`bern/study.py`, `bern/text_extract.py`)** — point Bern at notes, a lecture PDF, a Word document, or a photo of handwriting, and it builds a **reviewer, multiple-choice quiz, mock exam and flashcards** from that material. Fully extractive: every question is cut from a real sentence in the source, and wrong answers are other real terms from the same document. Understands English and Filipino definition patterns ("X is ...", "Ang X ay ...").
7. **Filipino / Taglish / English handwriting (`bern/ph_lexicon.py`)** — a 2,900-word dictionary (Filipino function words + day-to-day nouns, English basics, academic vocabulary, plus food and price terms harvested from Bern's own datasets) drives dictionary-constrained beam search over the glyph model. Raises end-to-end character accuracy from **72.5% to 83.6%** and exact-word hits from **8/82 to 29/82** across the three languages.

## Project Structure

```text
bern-ai/
├── bern/
│   ├── engine.py                    # BernAI inference engine, calibrated 'I don't know' gating, AST math, session memory, skill routing
│   ├── trainer.py                   # Word+Char TF-IDF + LogisticRegression training, 80/20 split + 5-fold CV
│   ├── augment.py                   # template phrases generated from the catalogs (training only)
│   ├── benchmark.py                 # held-out routing benchmark harness (python3 -m bern.benchmark)
│   ├── ph_prices.py                 # PHPriceExpert: curated prices, 17-region WFP/DA, 1,836 SKUs
│   ├── ph_nutrition.py              # PHNutritionExpert: PhilFCT + USDA + Open Food Facts + dishes + McDo/Jollibee
│   ├── money_vision.py              # PhilippineMoneyCounter (real-photo ROI + tabletop OpenCV)
│   ├── handwriting.py               # HandwritingReader: segment page -> classify glyphs (EMNIST)
│   ├── ph_lexicon.py                # Filipino/English/Taglish dictionary + beam-search word decoder
│   ├── text_extract.py              # PDF / DOCX / image / text -> plain text (stdlib only)
│   └── study.py                     # reviewer, quiz, mock exam, flashcards from source text
├── data/
│   ├── bern_dataset.json            # 22 intents, 1,481 English/Taglish phrases + the tuned training settings
│   ├── augmented_examples.json      # 398 catalog-generated phrases (training only, never evaluated on)
│   ├── eval/bern_benchmark.json     # 380 held-out messages: dev / test / test2
│   ├── ph_prices.json               # 64 curated PH basic goods (DTI SRP, DA, Sari-Sari 2026)
│   ├── ph_regional_prices.json      # 59 WFP commodities across 17 PH regions + 50 DA Nov 2026 items
│   ├── ph_sku_catalog.json          # 1,836 PH retail SKUs (incl. 150 canned goods SKUs)
│   ├── ph_nutrition_curated.json    # 28 Filipino dishes + 3 meals + 66 McDonald's/Jollibee items (built by script)
│   └── external/                    # Downloaded open datasets (see DATASET_MANIFEST.md)
│       ├── philfct_fnri_foods.json.gz          # 1,542 FNRI PhilFCT foods (CC0 extraction)
│       ├── usda_fdc_core.json.gz               # 13,577 USDA FoodData Central foods (CC0)
│       ├── openfoodfacts_ph_nutrition.json.gz  # 4,425 PH packaged products with nutrition facts (ODbL)
│       └── ...                                 # prices, PDFs, peso photos
├── models/
│   ├── bern_model.joblib            # Trained NLP intent classifier
│   ├── money_vision_model.joblib    # Trained PH banknote/coin classifier
│   ├── metrics.json                 # Held-out split accuracy, CV score, per-intent F1, benchmark summary
│   └── training_report.json         # Baseline vs trained benchmark, model search, rejection thresholds, dev log
├── react-native/
│   └── bernApi.ts                   # Full API client (chat, prices, nutrition, money, handwriting, study)
├── examples/
│   ├── quickstart.sh                # curl every endpoint in one go
│   └── python_client.py             # stdlib-only Python client
├── scripts/
│   ├── build_ph_catalog.py          # ETL: ph_prices / regional / sku JSON catalogs
│   ├── build_philfct_compact.py     # FNRI PhilFCT (CC0 dump) -> compact gz
│   ├── build_usda_core.py           # USDA FDC bulk CSV -> compact gz
│   ├── download_off_ph_nutrition.py # Open Food Facts PH products (bulk CSV export, resumable)
│   ├── build_ph_nutrition.py        # Filipino dish recipes + McDonald's / Jollibee -> ph_nutrition_curated.json
│   └── (intent training lives in bern/trainer.py: python3 -m bern.trainer)
│   └── download_kaggle_datasets.py  # Optional authenticated Kaggle downloader
├── tests/
│   └── test_bern.py                 # NLP, benchmark thresholds, prices, nutrition, money counter & API tests (23)
├── web/
│   └── index.html                   # Browser test console for every endpoint (served at "/")
└── server.py                        # Tornado HTTP + REST API server (0.0.0.0:8000)
```

## Quick Commands

```bash
python3 -m bern.trainer                     # FULL training: model search + calibration (~20 s)
python3 -m bern.trainer --fast              # refit only, what POST /api/train does (~10 s)
python3 -m bern.trainer                      # quick retrain with the stored settings
python3 -m bern.benchmark                    # score Bern on the held-out benchmark (prints dev failures only)
python3 -m bern.money_vision                 # train the peso vision model
python3 -m bern.handwriting                  # train the handwriting glyph model (downloads EMNIST, ~536 MB)
python3 -m bern.handwriting --image note.jpg # transcribe one photo of handwriting
python3 -m bern.study notes.pdf              # build a study pack from any document
python3 scripts/e2e_handwriting_eval.py --data-dir ~/emnist_dl   # EN/FIL/Taglish accuracy
python3 server.py                            # API + web studio on http://0.0.0.0:8000
python3 -m pytest -v                         # 23 tests

# rebuild the data (all optional - the built files are included)
python3 scripts/build_ph_catalog.py          # price catalogs
python3 scripts/build_philfct_compact.py     # FNRI PhilFCT (downloads ~0.5 MB)
python3 scripts/build_usda_core.py           # USDA FDC (downloads ~13 MB)
python3 scripts/download_off_ph_nutrition.py # Open Food Facts PH (streams a ~1.5 GB export, ~5 min)
python3 scripts/build_ph_nutrition.py --check  # recompute dishes / restaurants and print a review table
```

## What you can ask Bern (chat / `POST /api/chat`)

```text
calories in garlic                         -> FNRI PhilFCT: 129 kcal/100 g, 1 clove = 4 kcal
ilang calories ang kanin                   -> boiled white rice, 1 cup = 206 kcal
protein in egg / nutrition facts of milk   -> macro highlighted, similar foods listed
ground beef nutrition facts                -> USDA raw 80/20 etc.
corned beef / century tuna / canned foods  -> FNRI lab values + branded PH cans
calories ng bopis / beef pares calories    -> recipe estimate with range + reported values
nutrition facts of pork adobo              -> 312 kcal/100 g (range 181-428), FNRI canned adobo 277
mcdo big mac calories / chicken mcdo       -> McDonald's PH flyer + US macro reference
jollibee chickenjoy nutrition              -> 220 (drumstick) - 380 (thigh) kcal, flagged unverified
calories of 2 pc chickenjoy and 1 cup rice -> meal total with per-item breakdown
bopis vs adobo                             -> side-by-side comparison
magkano ang itlog                          -> (price expert, unchanged)
```

### Conversation memory

`POST /api/chat` keeps a per-session transcript, so a follow-up works the way a person expects
it to. Measured in one session:

```text
magkano ang bigas                -> ph_price_lookup 0.93   —
magkano ang itlog                -> ph_price_lookup 0.99   —
what about the other one?        -> ph_price_lookup 0.92   understood as "magkano ang Regular Milled Rice"
and that one?                    -> ph_price_lookup 0.88   understood as "magkano ang Chicken Egg (Medium…)"
```

This is rule-based, not a language model. Bern records the items each answer was actually
about, then rewrites a short follow-up into the question it stands for. It **declines rather
than guesses**: with only one item ever discussed, "what about the other one?" is left alone
and answered as written, because inventing a subject answers the wrong question while sounding
confident.

The response reports what was understood and what is remembered:

```json
{"reply": "...", "intent": "ph_price_lookup", "resolved_as": "magkano ang Regular Milled Rice",
 "conversation": {"turns": 3, "subject": "Regular Milled Rice", "discussed": ["Chicken Egg…", "Regular Milled Rice"]}}
```

`DELETE /api/tutor` clears a tutoring session; chat sessions are in-memory and reset with the
process. Pass a different `session_id` to start a fresh conversation.

### Explanations and unseen questions

Two things a bag-of-words model cannot do, handled honestly:

| Ask | Without a key | With `BERN_LLM_API_KEY` |
| --- | --- | --- |
| `bakit mahal ang bigas?` | Says plainly it cannot explain causes, then gives the price data it does have | Explained |
| A question outside the 24 intents | Declines and offers *Teach Bern This* | Answered |

`engine` controls this: `"local"` never calls a model, `"llm"` and `"auto"` use one when a key
is set and fall back to the local answer when it is not — or when the call fails. The web
console shows which mode is active under the message box.

`use_llm=False` on `BernAI.chat` forces the offline path; the frozen benchmark uses it so
scoring stays reproducible. Routing is unchanged by any of this: 96.9% on `test`, 97.4% on
`test2`.

## REST API Endpoints

- `GET /api/health` — Server status, model readiness, and catalog counts
- `POST /api/chat` — Body: `{"message": "calories ng bopis", "session_id": "app-1"}`
- `POST /api/predict` — Raw intent classification & confidence breakdown
- `POST /api/train` — Retrain Bern or teach new intents/phrases live
- `GET /api/metrics` — Model accuracy, F1, per-intent scores and the benchmark summary
- `GET /api/training-report` — Baseline vs trained benchmark, model search, rejection thresholds, development log
- `GET /api/dataset` — Full NLP dataset JSON
- `GET /api/datasets` — Summary of integrated price, SKU, nutrition and peso-photo datasets
- `GET /api/ph-prices` — Query curated & regional prices (`?q=bigas&region=calabarzon`, `?category=...`)
- `GET /api/ph-prices/skus` — Search `1,836` PH retail & canned-food SKUs
- `GET /api/nutrition?q=garlic` — Search foods (`&source=philfct|usda|off|dish|restaurant`, `&type=`, `&category=`, `&limit=`)
- `GET /api/nutrition/item?id=philfct:D084&grams=200` — One food; with `grams` returns the nutrients for that amount
- `POST /api/nutrition/meal` — `{"text": "2 pc chickenjoy and 1 cup rice"}` or `{"items": [{"id": "dish:pork-adobo", "qty": 1}, {"name": "rice", "qty": 1, "unit": "cup"}]}`
- `GET /api/nutrition/summary` — Counts per source / type / category
- `POST /api/count-money` — Multipart (`image`) or JSON (`{"image_base64": "..."}`) Philippine Peso counter
- `GET /api/sample-money-photo` — Synthetic tabletop peso photo for testing
- `DELETE /api/intents/<tag>` — Remove an intent and retrain

Example:
```bash
curl "http://localhost:8000/api/nutrition?q=bopis&limit=2"
curl -X POST http://localhost:8000/api/nutrition/meal -H "Content-Type: application/json" \
     -d '{"text":"pares at kanin"}'
```

Each food record has `per_100g` (generic foods, products, dishes) or `per_serving` (restaurant items, composite meals), a default household `serving` (e.g. `1 clove (3 g)`), `quality`, `source_ref`, and — where relevant — `range`, `reported`, `ingredients` and a US macro `reference`.

## Training Bern

Bern's intent model was trained in a measured, repeatable way (`python3 -m bern.trainer`):

1. **Frozen benchmark first.** `data/eval/bern_benchmark.json` holds 380 realistic English/Taglish messages (typos, short chat, off-topic questions) written *independently* of the training phrases and checked for zero overlap (a test enforces it). `dev` (135) is used for error analysis; `test` (130) and `test2` (115, written after the first training run) are held-out.
2. **Data.** 1,101 new hand-written phrases (Tagalog/Taglish conversation, API / React Native / training questions, price, budget, nutrition wording), a new **`out_of_scope`** intent (239 off-topic examples so Bern knows what it does *not* know), 398 catalog-generated phrases where the same food appears with price, nutrition and "how to cook" wording, and refreshed answers about Bern's own API, files and pipeline.
3. **Rule fixes** the data could not solve: calculator words (`sum of`, `times`, `divided by`, `squared`), English + Tagalog name extraction (`ako si Ana`, `i am called Rina`), Tagalog baskets (`3 itlog at 1 kilo bigas`), Tagalog price categories (`gulay`, `karne`), SKU-only products (`tuyo`) answered honestly, `high in sodium` / `compare X and Y` parsing, and a calculator shortcut for "how much is 12 x 12".
4. **Model selection** by 5-fold cross-validated macro-F1 on the hand-written phrases (generated phrases are train-only): n-gram ranges → word/char weights → `C` × out-of-scope weight. Chosen: word 1–2 + char_wb 2–5 TF-IDF (char weight 1.0 / word 0.8), `LogisticRegression(C=16)`, generated phrases on (CV macro-F1 **0.876**, vs 0.864 LinearSVC and 0.820 ComplementNB on the same features).
5. **Calibrated "I don't know" thresholds** (cross-validated predictions + dev split) replaced the hand-set ones — false rejections of valid messages fell from 5.1% to 0.9%.

| Held-out score | Before training | After (test) | After (fresh test2) |
|---|---|---|---|
| Message routed to the right skill | 74.6% | **96.9%** | **97.4%** |
| Raw ML classifier (in-scope) | 79.7% | 97.5% | 98.9% |
| Valid messages wrongly declined as "unknown" | 24.6% | 0.9% | 0.0% |
| Off-topic questions correctly declined | 91.7% | 91.7% | 90.5% |
| Answer checks (e.g. `4410`, `kcal`, `₱`) | 97.4% | 100% | 100% |

How to read this honestly: the benchmark was written by one author (me), so real-world accuracy will differ; `dev` scores (100%) are optimistic because errors on it were used for learning; `test` was scored twice (before and after the threshold calibration) and `test2` once for routing. Three expert-module bugs (generic words matching a vegetable in SKU search, unit words like "cup" treated as food names, Tagalog "listahan ng presyo") were fixed after they surfaced in held-out failures — only answer checks were affected, and both numbers are in `models/training_report.json`. Known weak spots: off-topic questions phrased like price or budget questions ("magkano ang bagong kotse", "convert 100 dollars to pesos") can still slip through, and ambiguous pairs (`api_usage` vs `deployment_tips`, `greeting` vs `small_talk` for "kumusta") are sometimes swapped. Prices for products that only exist as SKUs (e.g. *tuyo*) are reported as "no verified price".

**Teach Bern more:** use the **Train Bern** tab or `POST /api/train`. Live retraining takes ~6 s, uses the tuned settings stored in the dataset, and moves an identical phrase out of `out_of_scope` if you teach Bern something it used to decline. Re-run `python3 -m bern.trainer` after bigger changes and compare `models/metrics.json`.

## Data quality — read this before trusting a number

| `quality` | Meaning |
|---|---|
| `lab_analysis` | FNRI PhilFCT laboratory-analysed / calculated values (energy matches 4P+4C+9F for 100 % of foods). |
| `usda_reference` | USDA FoodData Central reference values. |
| `manufacturer_label` | Printed label typed in by Open Food Facts volunteers; implausible entries were filtered out, but typos can remain. |
| `recipe_estimate` | **Estimate, not a measurement.** Filipino dishes (adobo, pares, bopis ...) have no lab data, so Bern sums FNRI/USDA ingredient values for a documented home-style recipe (ingredients are listed in every record). The cut of meat and oil change calories a lot, so each carries a `range`. |
| `official_published` | McDonald's PH flyer (**data as of March 2010**; recipes/portions may have changed). |
| `reported_secondary` | Third-party menu sites (Jollibee does not publish nutrition facts on its PH website); they disagree with each other, so a range is shown. |

Philippine chains' live nutrition pages (Jollibee app, McDelivery) cannot be downloaded, and FNRI's own web app is blocked for scripts — see `data/external/DATASET_MANIFEST.md` for sources, licences (PhilFCT CC0 extraction, USDA CC0, Open Food Facts **ODbL**) and dead ends.

## React Native / Expo

A dependency-free TypeScript client lives in `react-native/bernApi.ts`: handwriting, chat with
session memory, study packs, prices, nutrition and peso counting, fully typed.

```bash
cp react-native/bernApi.ts your-app/src/bern/bernApi.ts
```

```ts
import { BernClient } from "./bern/bernApi";
const bern = new BernClient("http://192.168.1.20:8000");   // your server, not localhost
const result = await bern.handwriting(photoUri);            // photo URI from the image picker
```

**[react-native/README.md](react-native/README.md)** covers the parts that actually cost time:
the iOS and Android settings that block plain HTTP, why `localhost` never works on a phone,
camera-to-text in a working component, and how to degrade when the server is unreachable.

## React Native / Expo

Copy `react-native/` into your app, set `BERN_API_URL` in `bernApi.ts`, then:

```tsx
import { searchNutrition, estimateMeal, askBern } from "./bernApi";
import BernNutritionScreen from "./BernNutritionScreen";

const { items } = await searchNutrition("bawang");           // Garlic bulb 129 kcal/100 g
const meal = await estimateMeal({ text: "pares at kanin" });  // totals + per-item breakdown
```

## Handwriting

`bern/handwriting.py` turns a photo of handwriting into text. It is two stages:
OpenCV finds and isolates each character, then a classifier names it.

```bash
python3 -m bern.handwriting                    # train (one-off, downloads EMNIST ~536 MB)
python3 -m bern.handwriting --evaluate         # re-score the saved model
python3 -m bern.handwriting --image note.jpg   # transcribe a file
```

```python
from bern.handwriting import HandwritingReader

result = HandwritingReader().read_image(open("note.jpg", "rb").read())
print(result["text"], result["confidence"], result["backend"])
```

```bash
curl -X POST http://localhost:8000/api/handwriting -F image=@note.jpg
```

Chat: `POST /api/chat` with `"read my handwriting"` routes to the `handwriting_ocr` intent
and returns upload instructions — the chat endpoint is text-only, so the image itself goes
to `/api/handwriting`.

### What it reads, and what it does not

| | |
|---|---|
| **Supported** | Hand-*printed* text: separated block letters, clean paper or a phone photo |
| **Supported** | `. , - ?` and standalone symbols — `+ = % ° ñ Ñ ( ) / @ # & $ _ * : ;` — see below |
| **Partly supported** | Phone photos: perspective, ruled paper and two-column layout are corrected automatically |
| **Not supported** | Cursive or connected script — the segmenter cannot split a word written without lifting the pen |

Measured performance of the default `bern-glyph` backend:

| Metric | Result |
|---|---|
| Character accuracy (EMNIST `byclass` test split, 30,000 unseen glyphs) | **82.55%** |
| End-to-end, dictionary and symbols on, clean pages | **91.7%** English, **93.5%** Filipino, **94.2%** Taglish |
| End-to-end with ±9° rotation, blur, scale and baseline jitter | **82.1%** / **80.4%** / **89.1%** |
| Pages with words and `. , - ?` composed in | 88.1% clean, 82.1% jittered |

The single biggest accuracy lever is the **dictionary**, not the classifier: the same model
scoring 81.0% raw reaches 91.7% once `bern/ph_lexicon.py` constrains the decode to real words.
That also sets the limit — a correct word outside the 2,936-word list cannot be recovered,
because the score needed to win is never on the table.

Cursive needs the TrOCR backend:

```bash
pip install transformers torch pillow
python3 -m bern.handwriting --image note.jpg   # auto-selects TrOCR when installed
```

`HandwritingReader(backend="glyph" | "trocr" | "tesseract" | "auto")` overrides the choice.
`case_mode="upper"` or `"lower"` forces output case (EMNIST scores `O` and `o` as different
classes, and folding the two case forms together is worth about 5 points).

### Retraining

`python3 -m bern.handwriting` writes `models/handwriting_glyph_model.joblib` (252 KB) and
`models/handwriting_metrics.json`. Training streams the EMNIST file in batches, so it needs
under 1 GB of RAM; on 2 CPU cores it takes about 8 minutes for 5 epochs.

### Punctuation, symbols and accents

EMNIST contains only `0-9A-Za-z`, so the glyph model structurally cannot emit `.` or `ñ`.
Two additions sit on top of it:

- **`bern/symbols.py`** — a 28-class classifier (`.,?!-:;()/\+=%°ñÑ×÷√@#&$_*'"`) trained on
  rendered glyphs, 117 KB. Cross-family accuracy is 100% on fonts it never saw. It runs only
  where the glyph model cannot be trusted: standalone single-glyph words, and the last glyph of
  a word, gated at 0.95 confidence.
- **`classify_punctuation`** — pure geometry, no model. A mark begins low in the letter band;
  every letter reaches the top. That single positional test separates `. , -` from letters far
  more reliably than any size threshold.

Measured, 14 trials each:

| mark | read exactly | detected at all |
|---|---|---|
| `.` | 10/14 | 14/14 |
| `,` | 8/14 | 11/14 |
| `-` | 10/14 | 14/14 |
| `?` | 7/14 | 10/14 |

Colons, semicolons and quotes are deliberately *not* inferred from geometry: the test that
looked plausible also fired on the thin tail of a comma and cost real commas.

### Phone photos

`HandwritingReader(photo_mode=True)` is on by default and applies three repairs, each of which
only helps a photograph and does nothing to a clean scan:

- **Perspective** — finds the page outline and warps it square, so glyphs near one corner are
  not systematically taller than those near another.
- **Ruled lines** — deletes horizontal runs that span most of the page. Guarded: if that would
  remove more than 35% of the ink it is abandoned as handwriting rather than rules.
- **Columns** — reads each column top to bottom. Requires three or more lines and a gutter
  wider than 2.5× the page's median word gap, so a single line of text is never split.

```python
reader = HandwritingReader(photo_mode=False)   # compare against the raw pipeline
```

The result dict reports what happened:

```python
{"text": "...", "repairs": {"perspective": True, "rules_removed": 2, "columns": 2}}
```

**None of this has been run against a real photograph.** It is unit tested on synthetic pages,
and the thresholds in `bern/preprocess.py` should be expected to need adjustment once real
photos exist.

## Study materials from notes and lectures

Point Bern at study material and it produces a reviewer, quiz, mock exam and flashcards.

```bash
python3 -m bern.study lecture.pdf --out pack.json --flashcards 20 --quiz 10
```

```python
from bern.text_extract import extract_path
from bern.study import generate_study_pack

text = extract_path("lecture.pdf")["text"]
pack = generate_study_pack(text, flashcards=20, quiz_items=10, true_false_items=8)
print(pack["reviewer"]["sections"][0]["heading"])
print(pack["quiz"][0]["question"], pack["quiz"][0]["options"])
```

```bash
curl -X POST localhost:8000/api/study -F file=@lecture.pdf
curl -X POST localhost:8000/api/extract -F file=@notes.docx
```

### Supported inputs

| Format | Engine | Notes |
|---|---|---|
| PDF (text layer) | `pypdf` if installed, else built-in | Built-in reader inflates FlateDecode streams with `zlib` and reads text operators |
| DOCX | `zipfile` + XML | Stdlib only |
| Images (PNG/JPG/WebP/BMP/TIFF) | handwriting pipeline | Scanned or photographed notes |
| TXT / MD / CSV | direct decode | UTF-8, cp1252 and latin-1 handled |

**No new dependencies.** Every format has a working stdlib path, so the feature does not
disappear when a virtualenv does.

### What it generates

- **flashcards** — term on the front, definition on the back
- **quiz** — multiple choice; wrong answers are other real terms from the same document
- **reviewer** — headings, key points and a glossary, section by section
- **mock exam** — multiple choice + true/false with a balanced answer key and explanations

### Honest limits

- **Extractive, not generative.** Bern has no language model. Every question is built from a
  sentence that exists in the source; nothing is invented, and nothing is summarised the way
  an LLM would summarise it.
- Quality tracks the source. Notes with definitions and distinct terminology produce strong
  items; bullet fragments produce weak ones.
- Distractors are chosen by length similarity, not by part of speech, so an occasional verb
  lands where a noun belongs.
- **Scanned PDFs with no text layer extract to nothing.** There is no PDF rasteriser bundled;
  export the pages as images and send them through `/api/handwriting` instead.
- Encrypted PDFs are refused explicitly rather than silently returning garbage.

---

## Multilingual handwriting

`bern/handwriting.py` reads hand-*printed* text in English, Filipino and Taglish. The
character model is unchanged — EMNIST is script-agnostic Latin — but a dictionary now repairs
what the character model cannot.

```python
from bern.handwriting import HandwritingReader
result = HandwritingReader().read_image(open("notes.jpg", "rb").read())
print(result["text"])
```

Word decoding runs beam search over the per-glyph probabilities (top-5 per glyph) and scores
each hypothesis as `sum(log p) + 6.0 · [word ∈ lexicon]`. That is what turns `B1GAS` into
`bigas` and `6A7AS` into `gatas`: the letters are individually plausible, but only one string
is a Filipino word. `HandwritingReader(use_lexicon=False)` disables it for inspection.

The dictionary is `bern/ph_lexicon.py`: ~2,900 words — Filipino function words and everyday
nouns, English basics, academic vocabulary, and food/price terms harvested automatically from
`data/ph_prices.json`, `data/ph_nutrition_curated.json` and `data/ph_sku_catalog.json`.

### Measured effect

Same corpus, same seeds, same jitter — only dictionary correction differs:

| Corpus | Raw model | With dictionary | |
|---|---|---|---|
| English, clean | 81.0% | **84.5%** | +3.5 |
| Filipino, clean | 77.2% | **91.3%** | +14.1 |
| Taglish, clean | 76.6% | **88.3%** | +11.7 |
| English, jittered | 66.7% | **79.8%** | +13.1 |
| Filipino, jittered | 60.9% | **69.6%** | +8.7 |
| Taglish, jittered | 71.5% | **84.7%** | +13.2 |
| **All, character accuracy** | **72.5%** | **83.6%** | **+11.1** |
| **All, exact words** | **8/82** | **29/82** | **3.6×** |

Filipino gains the most because its vocabulary is short and highly patterned — exactly the
case where a dictionary beats a character model. Reproduce with:

```bash
python3 scripts/e2e_handwriting_eval.py --data-dir ~/emnist_dl              # with dictionary
python3 scripts/e2e_handwriting_eval.py --data-dir ~/emnist_dl --no-lexicon # without
```

Still out of scope: **cursive** (segmentation cannot split a word written without lifting the
pen — use the TrOCR backend) and **punctuation** (EMNIST has no punctuation classes).

## How training actually works

Bern is four small models and one word list, each trained or built by a different process.
"Retraining" means one of these, not editing code — but several real improvements *were* code
fixes rather than training, and that distinction matters.

| What | How it is trained | Command | Measured |
|---|---|---|---|
| **Intent model** (`models/bern_model.joblib`, 5.6 MB) | TF-IDF word 1–2-grams + char 2–5-grams → `LogisticRegression` | `python3 -m bern.trainer` | 17.7 s, CV macro-F1 0.8634, CV accuracy 0.8719 |
| **Handwriting glyphs** (`models/handwriting_glyph_model.joblib`, 252 KB) | SGD (log-loss, averaged) over 341-d HOG, 5 epochs over EMNIST `byclass` | `python3 -m bern.handwriting` | ~8 min on 2 cores, 82.55% held-out |
| **Symbols** (`models/symbol_model.joblib`, 117 KB) | SGD over 19,600 synthetic renders of 28 classes | `python3 -m bern.symbols` | 100% on unseen font families |
| **Dictionary** (`bern/ph_lexicon.py`, 2,936 words) | **Not trained.** Hand-curated. | add words by hand | coverage is the binding limit |

Two ways to retrain the intent model, and they are not the same job:

```bash
python3 -m bern.trainer          # full: model search + threshold calibration (~18 s)
python3 -m bern.trainer --fast   # refit only, no CV (~10 s)
```

`POST /api/train` runs the **fast** path. It reports no `cv_macro_f1` — there is no
cross-validation in that mode — so do not compare a fast score against a full one.

### What training does *not* fix

Worth knowing, because it is where most of the recent accuracy came from:

| Symptom | Training helps? | What actually helps |
|---|---|---|
| Model confuses `0` and `O` | A little | Dictionary constraint on decoding |
| Word splits in the wrong place | **No** | Segmentation code |
| `.` and `,` never appear | **No** — not in EMNIST | Separate symbol model |
| A correct word outside the dictionary | **No** | Adding the word to `ph_lexicon.py` |
| Cursive | No | Different model class entirely (CTC) |

Measured on this repo: the same 82.55% glyph model scores 91.7% end-to-end once decoding is
constrained, and punctuation went from impossible to readable without retraining the glyph
model at all. Retraining is the right tool when the *model* is wrong; it does nothing when the
*pipeline around it* is wrong.

---

## Optional: a language model for Tier 3 and Tier 4

Everything above runs offline on scikit-learn, OpenCV and the 2,936-word dictionary. Two
things genuinely need a language model, because a 21k-parameter classifier cannot do them:

| | Local (default) | With `BERN_LLM_API_KEY` |
|---|---|---|
| **Chat memory** | Works — rule-based reference resolution over the session transcript | Unchanged (memory is local either way) |
| **Explanations** | Declines openly, gives the data instead | Explains |
| **Unseen questions** | Declines, offers to be taught | Answers |
| **Study packs** | Extractive: finds real definitions in your notes and quizzes you on them. Invents nothing. | Generative: explains ideas in its own words, writes plausible distractors, follows the notes' language. |
| **Tutoring** | Not available. | Multi-turn: remembers what you asked and answers follow-ups, grounded in your notes. |

The local path is **never removed**. With no key, `/api/study` behaves exactly as before and
`/api/tutor` returns 503 with an explanation rather than a scripted reply that ignores you.

```bash
export BERN_LLM_API_KEY=sk-...          # required to enable
export BERN_LLM_MODEL=gpt-4o-mini       # default
export BERN_LLM_BASE_URL=...            # any OpenAI-compatible endpoint
```

`BERN_LLM_BASE_URL` matters more than it looks: point it at Ollama or llama.cpp and the notes
never leave the machine even with a model configured.

```bash
# auto: uses the model only if one is configured, otherwise the local generator
curl -X POST http://localhost:8000/api/study -H "Content-Type: application/json" \
     -d '{"text":"Mitochondria produce ATP...","engine":"auto"}'

# tutoring keeps the conversation, keyed by session_id
curl -X POST http://localhost:8000/api/tutor -H "Content-Type: application/json" \
     -d '{"session_id":"bio","notes":"Mitochondria produce ATP.","question":"What makes ATP?"}'
```

```python
from bern import llm

session = llm.TutorSession(notes=open("chapter3.txt").read())
print(session.ask("What is the powerhouse of the cell?")["answer"])
print(session.ask("And what does it produce?")["answer"])   # follow-up, same thread
```

Unreachable model, HTTP errors and malformed replies all raise `bern.llm.LLMError`. The server
turns that into a 400 with the reason; it does not silently substitute a local pack, because a
student who asked for an explanation should not receive ten flashcards and be told it worked.

Model replies are treated as untrusted: a quiz item whose answer is not one of its own options
is dropped, flashcards need both sides, and the mock exam's answer key is rebuilt from the
questions rather than taken from the model.

**Not verified:** the live HTTP call. There is no API key in this environment, so every test in
`tests/test_llm.py` covers the no-key and reply-cleaning paths only. The request/response shape
follows the OpenAI chat-completions API, which is unverified against the live service here.

---

## Sharing it with other people

See **[SHARE.md](SHARE.md)** for the three ways to get Bern in front of someone else, in order
of effort: same network with no install, Docker plus a tunnel, or a hosted deploy.

```bash
python3 -m pip install -r requirements.txt
python3 server.py                 # binds 0.0.0.0, honours $PORT
```

Then hand over `http://<your-lan-ip>:8000`. The Handwriting tab has a **"Load a sample"**
button that loads `examples/sample_handwriting.png` — real handwriting that reads back exactly
— so a first-time visitor has something to try without hunting for a photo.

> **Bern has no authentication.** `POST /api/train` rewrites the model and `DELETE
> /api/intents/<tag>` deletes a skill, both open to anyone who has the URL. Fine on a
> classroom LAN; not fine on the public internet without a login in front of it. `SHARE.md`
> has the details.

## Testing it

Three ways, in order of speed.

### 1. Browser console

```bash
python3 server.py          # serves http://0.0.0.0:8000
```

Open `http://localhost:8000` — `web/index.html` is a console with a tab per feature: chat,
handwriting, study pack, extract, money. The health badge turns green when the server answers.

### 2. curl

```bash
./examples/quickstart.sh                            # every endpoint, localhost
BERN=http://192.168.1.20:8000 ./examples/quickstart.sh   # another host
NOTE_FILE=note.jpg PDF_FILE=lecture.pdf ./examples/quickstart.sh   # include uploads
```

### 3. Python, stdlib only

```bash
python3 examples/python_client.py http://localhost:8000
```

`examples/python_client.py` has no `requests` dependency and doubles as copy-paste material.

### Automated tests

```bash
python3 -m pytest -v        # 129 tests: NLP, prices, nutrition, handwriting, study, conversation,
                            # extraction, punctuation geometry, symbols, photo preprocessing
```

The end-to-end handwriting benchmark needs EMNIST locally (see `HANDOFF.md` §6 for the
download); the unit tests do not.

```bash
PYTHONPATH=. python3 scripts/e2e_handwriting_eval.py --data-dir ~/emnist_dl
PYTHONPATH=. python3 scripts/bench_handwriting.py --cases 40
```

---

## Integrating it into your app

The server sets `Access-Control-Allow-Origin: *`, so a browser or Expo web build can call it
directly. In React Native / Expo:

```ts
import { BernClient } from "./react-native/bernApi";

const bern = new BernClient("http://192.168.1.20:8000");

await bern.chat("magkano ang bigas");                      // { reply, intent, confidence, ... }
await bern.handwriting(photoUri);                          // { text, lines, confidence, ... }
await bern.studyFile(pdfUri, "lecture.pdf");               // { reviewer, quiz, flashcards, mock_exam }
await bern.countMoney(photoUri);                           // { total_php, items }
```

`react-native/bernApi.ts` is dependency-free (uses `fetch` + `FormData`), fully typed, and
covers all 22 routes.

### The one thing that trips people up

**`localhost` on a phone means the phone.** When testing on a physical device, use your
machine's LAN address — `http://192.168.x.x:8000` — and make sure the device is on the same
Wi-Fi. If the app hangs instead of erroring, it is almost always this.

### Deployment notes

- The server binds `0.0.0.0:8000`, port overridable with `PORT`.
- It loads the intent model, the nutrition corpus (~19k foods) and the money model at import
  time, so cold start takes several seconds before the first request is answered.
- There is **no authentication**. Anyone who can reach the port can call every endpoint,
  including `POST /api/train` and `DELETE /api/intents/{tag}`. Put it behind a reverse proxy
  with auth, and do not expose it directly to the internet.
