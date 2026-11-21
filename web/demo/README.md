# Browser demo — let people try Bern

`index.html` is one self-contained file. Open it, double-click it, drop it in a chat, or serve
it from anywhere: it needs no server, no network and no build step to *run*.

It is not a mock. The intent classifier is the real one — `models/ondevice/intent.json`,
24 classes, the same weights the phone and the server use — loaded into the page and evaluated
in JavaScript. The replies are the ones stored with the model, so nothing is invented on the
page to make it look better than it is.

```
web/demo/
├── app.jsx          the UI, React + Tailwind classes
├── index.html       generated, everything inlined
├── assets/          Bern the mascot, cut out of a white background
└── README.md        this file
```

## The mascot

`assets/` holds the same drawing at the sizes the page actually uses:

| File | Size | Used for |
| --- | --- | --- |
| `bern-dog.png` | 1353×1885 | full resolution, for other uses (share cards, README) |
| `bern-dog-inline.png` | 366×510 | the hero on the empty screen, inlined at 2x |
| `bern-dog-face.png` | 128×128 | the header mark and the avatar beside each reply |
| `bern-dog-face-inline.png` | 80×80 | the inlined copy, 16 KB |
| `bern-dog-avatar.png` | 256×256 | head and ears, if you want more context |
| `bern-dog-cutout.png` | 1353×1885 | the transparency itself, if you want to re-crop |

Cutting it out was not a threshold. The fur is pale enough that a distance-from-white test
eats the dog, and a generous feather turns the nose pale so the white background shows through
it. The version here floods from the frame border only, so white *inside* the drawing — the eye
highlight, the gap between the legs — stays opaque, then softens just the boundary. It was
composited on both `#020617` and `#ffffff` and checked by eye before it went in.

## Rebuilding

```bash
python3 scripts/build_web_demo.py
```

Needs `node` and, the first time, downloads the Tailwind standalone binary and React's UMD
builds into `.cache/` (gitignored). Neither is needed to view the page.

What the build does:

1. Compiles `react-native/bernFeatures.ts` + `bernOnDevice.ts` to ES2020 and flattens the two
   modules into one IIFE. The runtime is compiled **without** the DOM lib, which is the check
   that it touches no browser API — the same code runs on a phone.
2. Compiles `app.jsx` to `React.createElement` calls with `tsc`. Inlining Babel would have added
   three megabytes to a page whose whole point is to load with nothing.
3. Compiles Tailwind against `app.jsx`, dark mode on a class rather than the media query.
4. Inlines React, ReactDOM, the runtime, the model and the CSS into one HTML file.

Nothing is fetched at runtime. The only `https://` strings in the output are the Tailwind
licence comment and example URLs inside Bern's own answers.

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
