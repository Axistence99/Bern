# External Open-Source Datasets Downloaded for Bern AI

Generated on 2026-11-11 inside `/home/user/bern-ai/data/external/`.

## 1. Philippine Cash & Coins (Computer Vision)
- **`ph_banknotes_coins/`** (1,100 real labeled images + `metadata.csv`, 256×256 JPEG)
  - **Upstream source:** [`RichMiguel/Banknotes-Coins`](https://huggingface.co/datasets/RichMiguel/Banknotes-Coins) (Hugging Face)
  - **License:** MIT License
  - **Denominations covered (22 front/back/series classes):**
    - `₱1,000` banknotes (`1000_New_Back` polymer, `1000_Old_Front`, `1000_Old_Back`) — 150 images
    - `₱200` banknotes (`200_Front`, `200_Back`) — 100 images
    - `₱50` banknotes (`50_Front`, `50_Back`) — 100 images
    - `₱20` banknotes (`20_Old_Front`, `20_Old_Back`) & `₱20` bimetallic coins (`20_New_Front`, `20_New_Back`) — 200 images
    - `₱10` coins (`10_Old_Front`, `10_Old_Back`, `10_New_Front`, `10_New_Back`) — 200 images
    - `₱5` coins (`5_Old_Back`, `5_New_Front`, `5_New_Back`) — 150 images
    - `₱1` coins (`1_Old_Front`, `1_Old_Back`, `1_New_Front`, `1_New_Back`) — 200 images
  - **Note on Kaggle:** Kaggle has no public Philippine peso banknote/coin image dataset and blocks unauthenticated API downloads (`HTTP 404` without `~/.kaggle/kaggle.json`). `RichMiguel/Banknotes-Coins` provides an open, MIT-licensed real-photo Philippine currency dataset directly accessible without login.

## 2. Philippine Common Goods, Food Prices & Canned Foods
- **`wfp_food_prices_phl.csv` & `wfp_markets_phl.csv`** (244,835 rows, Jan 2000 – Sep 15, 2026)
  - **Upstream source:** [HDX — World Food Programme Philippines Food Prices](https://data.humdata.org/dataset/wfp-food-prices-for-philippines) (PSA / BAS / DA-BAS)
  - **License:** CC BY-IGO
  - **Note on Kaggle:** This is the official upstream dataset mirrored by Kaggle's `usmanlovescode/philippines-food-prices-dataset` (2023) and `roseamolato/foodprices-regionx-philippines` (2023), except HDX includes data through September 2026 and downloads without login.
- **`da_amas_cleaned.csv`** (30,600 rows, Mar 1, 2025 – Nov 10, 2026)
  - **Upstream source:** Philippine Department of Agriculture Daily Price Index (`da.gov.ph/price-monitoring/`), compiled by [`LeinnarF/Philippine-Commodity-Price-Tracker`](https://github.com/LeinnarF/Philippine-Commodity-Price-Tracker)
- **`da_weekly_prices_ncr.csv`** (13,387 rows, Aug 25, 2023 – Nov 1, 2026)
  - **Upstream source:** DA Bantay Presyo NCR Weekly Price Monitoring, compiled by [`benhrnndz/metro-manila-food-price-eda`](https://github.com/benhrnndz/metro-manila-food-price-eda)
- **`dti_bnpc_srp_may2026.pdf`** (Official May 11, 2026 DTI BNPC Suggested Retail Prices Bulletin)
  - **Upstream source:** [DTI e-Presyo Official S3 Bulletin](https://dtiwebfiles.s3.ap-southeast-1.amazonaws.com/e-Presyo/SRP+Basic+Necessities+and+Prime+Commodities/2026/BNPC+SRP+BULLETIN.11+MAY+2026.0010.PSU+%5BCLEAN+A3-final%5D.pdf)
  - **Coverage:** Official SRPs for canned sardines, corned beef, beef loaf, meat loaf, luncheon meat, condensed/evaporated/powdered milk, coffee 3-in-1, bread, instant mami noodles, salt, condiments, soap, candles, and bottled water.
- **`sari_sari_2026_prices.pdf`** (293 Philippine Sari-Sari Store products across 22 categories, 2026 wholesale + SRP)
  - **Upstream source:** [PinoyNegosyo.net 2026 Sari-Sari Store Price List](https://pinoynegosyo.net/wp-content/uploads/2026/06/PinoyNegosyo.net-2026-Sari-Sari-Store-Price-List.pdf)
- **`retail_products_ph.json`** (1,706 Philippine sari-sari & supermarket SKUs incl. `canned_goods`, `canned fish`, `instant_noodles`, `dairy`, `rice`, `condiments`)
  - **Upstream source:** [`kierth/retail-products-philippines`](https://huggingface.co/datasets/kierth/retail-products-philippines)
- **`openfoodfacts_ph_canned.json`** (88 Philippine canned food SKUs with brands, weights, and barcodes)
  - **Upstream source:** [Open Food Facts Philippines Canned Foods API v2](https://world.openfoodfacts.org/country/philippines/category/canned-foods)
  - **License:** Open Database License (ODbL)
- **`ph_streetfoods_prices.csv`** (Philippine street foods & average PHP prices)
  - **Upstream source:** [`jomarie04/philippine-streetfoods-prices`](https://huggingface.co/datasets/jomarie04/philippine-streetfoods-prices)

## 3. Philippine Food Nutrition Facts (added 2026-11-11)

Everything below feeds `bern/ph_nutrition.py` (`/api/nutrition*`, the **PH Nutrition Facts** tab and the chat intent `ph_nutrition_lookup`).
Each record keeps a `quality` tag so Bern tells you *how trustworthy* a number is.

| File | Rows | Source & licence | Quality tag | Rebuild |
|---|---|---|---|---|
| `philfct_fnri_foods.json.gz` (112 KB) | **1,542 foods**, 17 food groups, 24 nutrients, per 100 g edible portion | **DOST-FNRI Philippine Food Composition Tables (PhilFCT)** via [`nedpals/fct-fnri-api`](https://github.com/nedpals/fct-fnri-api) (CC0-1.0 JSON extraction of `i.fnri.dost.gov.ph/fct/library`). Cite: Food and Nutrition Research Institute, DOST. | `lab_analysis` | `python3 scripts/build_philfct_compact.py` |
| `usda_fdc_core.json.gz` (770 KB) | **13,577 foods** (SR Legacy 7,793 + Foundation 353 + FNDDS 5,431) with household portions | **USDA FoodData Central**, public domain (CC0 1.0). | `usda_reference` | `python3 scripts/build_usda_core.py` |
| `openfoodfacts_ph_nutrition.json.gz` (250 KB) | **4,425 Philippine packaged products** with nutrition facts (4,378 unique after de-duplication; of 9,458 PH products listed - implausible labels removed) | **Open Food Facts**, ODbL 1.0 / DbCL 1.0 - attribution + share-alike. Streamed from the official CSV export (`countries_tags` contains `en:philippines`). | `manufacturer_label` | `python3 scripts/download_off_ph_nutrition.py` |
| `../ph_nutrition_curated.json` | **28 Filipino dishes + 3 meals + 66 McDonald's / Jollibee items** | Built by `scripts/build_ph_nutrition.py` (see below). | `recipe_estimate`, `official_published`, `reported_secondary` | `python3 scripts/build_ph_nutrition.py` |

### What covers the foods you asked for
| Food | Where it comes from |
|---|---|
| **Garlic, onion** (Bombay / shallot / spring), **milk** (cow, carabao, evaporated, condensed, powdered, goat) | FNRI PhilFCT (lab values, Filipino names such as *bawang ulo*, *sibuyas*, *gatas*) |
| **Canned foods** - sardines in tomato sauce, corned beef, luncheon meat, meat loaf, Vienna sausage, canned bangus, canned adobo / caldereta / kare-kare | FNRI PhilFCT (97 canned items, lab) + **191 branded canned products** (Century Tuna, Ligo, 555, Argentina, Purefoods ...) from Open Food Facts PH |
| **Ground beef** (70/80/85/90/95 % lean, raw/cooked) | USDA FoodData Central (PhilFCT has no ground beef) |
| **Adobo, pares, bopis, sisig, sinigang, kare-kare, lechon kawali ...** | FNRI lab values for *canned* dishes where they exist + Bern **recipe estimates** (below) |
| **McDonald's PH** | The chain's own nutrition flyer (kcal + sodium, March 2010) + USDA's McDonald's USA items as a macro reference |
| **Jollibee** | Jollibee USA nutrition guide values and Philippine menu-blog values (unverified, shown as ranges) |

### How to read the quality tags
- `lab_analysis` / `usda_reference` - laboratory or reference-database values.
- `manufacturer_label` - printed label typed in by Open Food Facts volunteers (98.9 % pass an energy-vs-macros sanity check; obvious typos were filtered out).
- `recipe_estimate` - **an estimate, not a measurement.** Each dish is a documented home-style recipe (ingredient grams are in the record and in `scripts/build_ph_nutrition.py`) whose nutrients are summed from FNRI/USDA ingredient values and divided by the cooked yield. Cuts of meat, oil and portion size change calories a lot, so dishes with a big spread carry a `range` (e.g. pork adobo 181-428 kcal/100 g) and, where sites publish numbers, a `reported` range for comparison (e.g. bopis 180-559 kcal per serving on recipe sites).
- `official_published` - McDonald's PH flyer (as of **March 2010**; recipes and portions may have changed).
- `reported_secondary` - numbers on third-party menu sites; they disagree with each other (Chickenjoy is listed anywhere from 200 to 380 kcal), so Bern shows a range and says so.

### Dead ends (do not retry)
- `i.fnri.dost.gov.ph/fct/library` (the FNRI web app) is **blocked by a WAF** for scripts - the CC0 JSON dump above is the usable copy.
- Jollibee PH (`jollibee.com.ph`) publishes **no nutrition facts** on its website (the app only); McDonald's PH menu / McDelivery pages are JavaScript apps (no static nutrition). The Wayback Machine was offline.
- `jrcbrr/hubog` (Filipino food + restaurant nutrition tracker, ~90 dishes) and `yuraka032/JolliNutrifacts` have **no licence** (`license: None`) - not redistributed.
- The Open Food Facts **search API** returned frequent HTTP 503s and is rate-limited (10 req/min); the bulk CSV export (~1.5 GB, cached under `~/.cache/bern_off`, not in the snapshot) is the reliable route.
- Kaggle nutrition datasets need credentials (`scripts/download_kaggle_datasets.py` if you have a key); none of the ones found were Philippines-specific, and PhilFCT/USDA/OFF cover the same ground with clear licences.

### Attribution / licences to keep when redistributing
- PhilFCT: CC0-1.0 extraction; credit **DOST-FNRI**. USDA FDC: CC0. Open Food Facts: **ODbL** (attribute "Open Food Facts contributors" and share derived databases alike).
- Restaurant figures are facts quoted with their source URL in each record (`source_ref`); Bern does not claim they are current.

## 4. Optional Kaggle Direct Downloader
If you have a Kaggle account (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` + `KAGGLE_KEY`), run:
```bash
python3 scripts/download_kaggle_datasets.py
```
