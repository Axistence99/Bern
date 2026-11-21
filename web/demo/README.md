# Browser demo — let people try Bern

`index.html` is one self-contained file. Open it, double-click it, drop it in a chat, or serve
it from anywhere: it needs no server, no network and no build step to *run*.

It is not a mock. The intent classifier is the real one — `models/ondevice/intent.json`,
24 classes, the same weights the phone and the server use — loaded into the page and evaluated
in JavaScript. The replies are the ones stored with the model, so nothing is invented on the
page to make it look better than it is.

```
web/demo/
├── app.jsx          the UI, React + Tailwind classes, ~450 lines
├── index.html       generated: 2.8 MB, everything inlined
└── README.md        this file
```

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
