# Browser demo — let people try Bern

`index.html` is one self-contained file. Open it, double-click it, drop it in a chat, or serve
it from anywhere: it needs no server, no network and no build step to *run*.

It is not a mock. The intent classifier is the real one — `models/ondevice/intent.json`,
24 classes, the same weights the phone and the server use — loaded into the page and evaluated
in JavaScript. The replies are the ones stored with the model, so nothing is invented on the
page to make it look better than it is.

```
web/demo/
├── app.jsx          the UI, React
├── styles.css       the stylesheet, hand-written, no framework
├── index.html       generated, everything inlined
├── assets/          Bern the mascot, cut out of a white background
└── README.md        this file
```

## The mascot

`assets/` holds the same drawing at the sizes the page actually uses:

The mascot is the **app logo in the top-right corner and nothing else**. The conversation owns
the centre of the screen, so the empty state is a prompt rather than a picture, and replies have
no avatar beside them.

| File | Size | Used for |
| --- | --- | --- |
| `bern-dog-face-inline.png` | 80×80 | the logo in the header, inlined at 16 KB |
| `bern-dog-face.png` | 128×128 | the same crop at full resolution |
| `bern-dog-avatar.png` | 256×256 | head and ears, if you ever want more context |
| `bern-dog.png` | 1353×1885 | full figure, for a share card or a README |
| `bern-dog-cutout.png` | 1353×1885 | the transparency itself, if you want to re-crop |

Only the 80×80 logo is inlined into `index.html`; the rest are in the repo for reuse elsewhere.

Cutting it out was not a threshold. The fur is pale enough that a distance-from-white test
eats the dog, and a generous feather turns the nose pale so the white background shows through
it. The version here floods from the frame border only, so white *inside* the drawing — the eye
highlight, the gap between the legs — stays opaque, then softens just the boundary. It was
composited on both `#020617` and `#ffffff` and checked by eye before it went in.

## Rebuilding

```bash
python3 scripts/build_web_demo.py
```

Needs `node`. On first run it downloads React and three OFL typefaces into `.cache/`
(gitignored). None of it is needed to view the page.

What the build does:

1. Compiles `react-native/bernFeatures.ts` + `bernOnDevice.ts` to ES2020 and flattens the two
   modules into one IIFE. The runtime is compiled **without** the DOM lib, which is the check
   that it touches no browser API — the same code runs on a phone.
2. Compiles `app.jsx` to `React.createElement` calls with `tsc`. Inlining Babel would have added
   three megabytes to a page whose whole point is to load with nothing.
3. Inlines Instrument Serif and IBM Plex Mono as woff2, then concatenates `styles.css`.
4. Inlines React, ReactDOM, the runtime, the model and the CSS into one HTML file.

Nothing is fetched at runtime. The only `https://` strings in the output are the licence
comment inside React's minified build and example URLs inside Bern's own answers.

## Design

`DESIGN.md` holds the audit and the direction: **the confidence number is the interface.**
Three structural consequences, so a later change can be checked against them:

- Every reply is a two-column record — words on the left, a fixed-width readout on the right —
  not a chat bubble. The readout's numeral is the largest thing on the screen.
- Colour means one of two things: green *known*, burnt orange *guessing or needs the server*.
  There is no third use of colour anywhere on the page.
- Structure comes from rules and spacing. There are no shadows, no gradients, and one radius
  (0) with two 2px exceptions on the composer rule and the aside borders.

Type: **Instrument Serif** for display only (wordmark, the opening statement, the confidence
numerals), **IBM Plex Mono** for everything else — the instrument's labels, the model's output,
and its own identifiers. The mono is the point: `out_of_scope` and `ph_price_lookup` are shown
as themselves, not prettified into marketing copy.

## Rebuilding

```bash
python3 scripts/build_web_demo.py
```

Needs `node`. On first run it downloads React and the typefaces into `.cache/` (gitignored).
None of it is needed to view the page.

What the build does:

1. Compiles `react-native/bernFeatures.ts` + `bernOnDevice.ts` to ES2020 and flattens the two
   modules into one IIFE. The runtime is compiled **without** the DOM lib, which is the check
   that it touches no browser API — the same code runs on a phone.
2. Compiles `app.jsx` to `React.createElement` calls with `tsc`. Inlining Babel would have added
   three megabytes to a page whose whole point is to load with nothing.
3. Inlines Instrument Serif and IBM Plex Mono as woff2, then concatenates `styles.css`.
4. Inlines React, ReactDOM, the runtime, the model and the CSS into one HTML file.

Nothing is fetched at runtime. The only `https://` strings in the output are the licence comment
inside React's build and example URLs inside Bern's own answers.

## What it shows, and what it deliberately does not

| Shown | Not shown |
| --- | --- |
| Intent, confidence, runner-up, latency | Prices, nutrition figures, transcriptions |
| The stored answer, with markdown | Teaching (the weights are read-only here) |
| A note when the answer needs the server | Anything the model was not trained on |

The confidence score is the point of the demo. Ask something outside the training data and the
number drops — that is a model saying it does not know, which is more interesting to look at
than a chatbot that always sounds certain.

Two notes appear under some answers and both are honest rather than decorative:

- **"That line is only Bern's opening"** for the eleven intents whose stored reply announces an
  answer the server produces from data the page does not have. `server_backed` in
  `models/ondevice/intent.json`.
- **"The reply mentions a Teach Bern This button"** for `out_of_scope`, because that button is in
  the desktop app and rewriting model weights in a browser tab would give you a model you could
  not keep.

## Verified

Run headless in Chrome, not assumed:

- mounts with no console errors, no page errors, no failed requests
- 8 prompts in sequence, all 8 rendered with intent, confidence and reply
- 375 px viewport: no horizontal scroll, controls stay on one row
- light and dark themes both render correctly

What is *not* verified: Safari, Firefox, and any real phone. The code uses nothing exotic —
`Uint8Array`, `Map`, `Math.exp` — but that is an argument, not a test.
