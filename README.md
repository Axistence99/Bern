# Bern AI

**Bern** is a custom, trainable Python Machine Learning assistant built with `scikit-learn` and OpenCV, persisted via `joblib`, served over a CORS-enabled Tornado REST API, and ready to integrate into React Native / Expo apps.

It combines:
1. **Trainable Intent ML Engine (`bern/engine.py`, `bern/trainer.py`)** — Hybrid Word (1–2 n-gram) + Char-WB (3–5 n-gram) TF-IDF + calibrated `LogisticRegression`, OOD vocabulary gating, AST math evaluator, `Asia/Manila` clock, and live teach/retrain endpoints (`21` intents, `1,481` hand-written English/Taglish phrases + `398` generated ones, tuned by cross-validation — see *Training Bern*).
2. **Philippine Common Goods Price Expert (`bern/ph_prices.py`)** — Curated 2026 PH catalog (`64` common items across `9` categories), `1,836` searchable PH retail SKUs (`150` canned foods), `17`-region UN WFP food price medians (`244,835` rows through `2026-09-15`), DA daily/NCR weekly Bantay Presyo (`30,600` rows through `2026-11-10`), official DTI May 2026 BNPC SRP bulletin, and Taglish budget/basket calculator.
3. **Philippine Food & Nutrition Expert (`bern/ph_nutrition.py`)** — **19,594 foods** with calories, macros, sodium and more: **1,542 DOST-FNRI PhilFCT foods** (garlic, onion, milk, canned corned beef/sardines/luncheon meat ...), **13,577 USDA FoodData Central foods** (ground beef, fast-food chicken, McDonald's USA ...), **4,378 Philippine packaged products** from Open Food Facts (191 canned goods, 117 milk/dairy ...), **28 Filipino dishes + 3 meals** (adobo, pares, bopis, sisig, sinigang, kare-kare ...) and **66 McDonald's PH / Jollibee items**. Understands Taglish (`bawang`, `sibuyas`, `gatas`, `giniling na baka`, `kanin`), scales servings, totals whole meals and tells you how trustworthy each number is (see *Data quality* below).
4. **Philippine Peso Money Counter (`bern/money_vision.py`)** — Dual-mode OpenCV + `scikit-learn` (`VotingClassifier` over 93-dim centered ROI + color/texture/contour features) trained on `1,100` real Philippine banknote & coin photos (`RichMiguel/Banknotes-Coins`, MIT) plus synthetic tabletop lighting scenes (**95.45%** end-to-end accuracy on `220` unseen real photos).

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
│   └── money_vision.py              # PhilippineMoneyCounter (real-photo ROI + tabletop OpenCV)
├── data/
│   ├── bern_dataset.json            # 21 intents, 1,481 English/Taglish phrases + the tuned training settings
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
│   ├── bernApi.ts                   # API client: chat, prices, nutrition search / item / meal, money counter
│   ├── BernChatScreen.tsx           # Drop-in Taglish chat & price screen
│   ├── BernNutritionScreen.tsx      # Drop-in nutrition search + meal calorie basket
│   └── BernMoneyScannerScreen.tsx   # Drop-in Expo Camera / ImagePicker Philippine Peso counter
├── scripts/
│   ├── build_ph_catalog.py          # ETL: ph_prices / regional / sku JSON catalogs
│   ├── build_philfct_compact.py     # FNRI PhilFCT (CC0 dump) -> compact gz
│   ├── build_usda_core.py           # USDA FDC bulk CSV -> compact gz
│   ├── download_off_ph_nutrition.py # Open Food Facts PH products (bulk CSV export, resumable)
│   ├── build_ph_nutrition.py        # Filipino dish recipes + McDonald's / Jollibee -> ph_nutrition_curated.json
│   ├── train_bern.py                # full training: model search -> threshold calibration -> final model -> benchmark report
│   └── download_kaggle_datasets.py  # Optional authenticated Kaggle downloader
├── tests/
│   └── test_bern.py                 # NLP, benchmark thresholds, prices, nutrition, money counter & API tests (23)
├── web/
│   └── index.html                   # Bern AI Studio: chat, prices, nutrition, money counter, trainer, RN code
└── server.py                        # Tornado HTTP + REST API server (0.0.0.0:8000)
```

## Quick Commands

```bash
python3 scripts/train_bern.py                # FULL training: model search + calibration + benchmark report (~2 min)
python3 -m bern.trainer                      # quick retrain with the stored settings
python3 -m bern.benchmark                    # score Bern on the held-out benchmark (prints dev failures only)
python3 -m bern.money_vision                 # train the peso vision model
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

Bern's intent model was trained in a measured, repeatable way (`python3 scripts/train_bern.py`):

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

**Teach Bern more:** use the **Train Bern** tab or `POST /api/train`. Live retraining takes ~6 s, uses the tuned settings stored in the dataset, and moves an identical phrase out of `out_of_scope` if you teach Bern something it used to decline. Re-run `python3 scripts/train_bern.py` after bigger changes and compare `models/training_report.json`.

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

Copy `react-native/` into your app, set `BERN_API_URL` in `bernApi.ts`, then:

```tsx
import { searchNutrition, estimateMeal, askBern } from "./bernApi";
import BernNutritionScreen from "./BernNutritionScreen";

const { items } = await searchNutrition("bawang");           // Garlic bulb 129 kcal/100 g
const meal = await estimateMeal({ text: "pares at kanin" });  // totals + per-item breakdown
```
