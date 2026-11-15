# Putting Bern in a mobile app

`bernApi.ts` is a complete, dependency-free client for Bern. Copy it into your project and you
have handwriting, chat, study packs, prices, nutrition and peso counting in about ten lines.

```bash
cp react-native/bernApi.ts your-app/src/bern/bernApi.ts
```

There is no npm package to install. The file uses only `fetch` and `FormData`, which React
Native and Expo both provide.

---

## Two ways to run Bern, and you can use both

| | Runs where | Needs a server | Answers from |
| --- | --- | --- | --- |
| `bernApi.ts` | Python server | yes | everything, including prices and file reading |
| `bernOnDevice.ts` | the phone | no | its 24 intents, stored answers, single characters |

**On the phone — `bernOnDevice.ts`.** The intent model, the handwriting model and the symbol
model are 2.84 MB of weights, exported to `models/ondevice/*.json` and evaluated in TypeScript.
No Python runtime, no OpenCV, no scikit-learn, no network. A message sent to a phone in
airplane mode still classifies and still gets an answer.

**On a server — `bernApi.ts`.** Everything the on-device models cannot do: reading a
photographed page (splitting lines, words and characters), looking up a price, correcting a
spelling against the dictionary, explaining an out-of-scope question, building a study pack.

The honest split is that the phone recognises *what you asked* and the server answers *with
data*. Shipping only the on-device path gives a Bern that knows 24 things; shipping only the
server path gives a Bern that stops working when the Wi-Fi does.

> An earlier version of this file said embedding Bern in the app was not recommended. That was
> about embedding the *Python runtime*, which is still a bad idea — hundreds of megabytes, slow
> cold start. Porting the *models* is a different proposition, and it is what `bernOnDevice.ts`
> does.

---

## Quick start

```ts
import { BernClient } from "./bern/bernApi";

const bern = new BernClient("http://192.168.1.20:8000");

// Handwriting from a photo URI
const result = await bern.handwriting(photoUri);
console.log(result.text);

// A conversation — same sessionId keeps the thread
await bern.chat("magkano ang bigas", "session-1");
await bern.chat("magkano ang itlog", "session-1");
const followUp = await bern.chat("what about the other one?", "session-1");
console.log(followUp.resolved_as);   // "magkano ang Regular Milled Rice"
```

| Method | Endpoint | Notes |
| --- | --- | --- |
| `health()` | `GET /api/health` | Ping first; `language_model` says if explanations are on |
| `chat(msg, sessionId, engine)` | `POST /api/chat` | Memory is server-side, keyed by `sessionId` |
| `handwriting(uri)` | `POST /api/handwriting` | Multipart field `image` |
| `countMoney(uri)` | `POST /api/count-money` | Multipart field `image` |
| `extractFile(uri, name)` / `extractText(t)` | `POST /api/extract` | Multipart field `file` |
| `studyFile` / `studyText` | `POST /api/study` | Multipart field `file` |
| `prices` `skus` `budget` `nutrition` `meal` | `/api/ph-prices*`, `/api/nutrition*` | Text only |
| `tutor(question, {notes, sessionId})` | `POST /api/tutor` | **Needs a server-side API key** |

---

## Running it with no server at all

The bundles live in `models/ondevice/` and are committed, so there is nothing to generate
before first run. Regenerate them with `scripts/export_ondevice.py` only when a model changes.

```ts
import { BernOnDevice } from "./bern/bernOnDevice";
import intent from "../models/ondevice/intent.json";
import glyph from "../models/ondevice/glyph.json";
import symbol from "../models/ondevice/symbol.json";

const leaf = await BernOnDevice.load(intent, glyph, symbol);

// What did they ask? Works offline, ~1 ms.
const { intent: tag, confidence } = leaf.classify("magkano ang bigas?");
// tag: "ph_price_lookup", confidence: 0.97

// Bern's own answer, stored with the model. No lookup, so no price in it.
leaf.reply("salamat po");        // { intent: "gratitude", text: "You're welcome! Happy to help anytime." }

// One character, straight from a 28x28 ink-bright crop.
const crop = leaf.prepare(cameraGray, width, height);   // normalise any size to 28x28
leaf.readGlyph(crop);            // { char: "a", confidence: 0.93, candidates: [...] }
leaf.readSymbol(crop);           // ".", ",", "?", "+", "ñ", ...
```

| Method | Cost | What it needs |
| --- | --- | --- |
| `classify(text)` | ~1 ms | nothing |
| `reply(text)` | ~1 ms | nothing; `needsServer` is set for `out_of_scope` |
| `readGlyph(28x28)` | ~2 ms | a **cropped** character — see below |
| `readSymbol(28x28)` | ~2 ms | a cropped symbol |
| `prepare(gray, w, h)` | ~1 ms | any grey crop, ink bright |

**`readGlyph` takes one character, not a page.** Splitting a photo into lines, words and glyphs
is done with OpenCV projection profiles and morphological thresholds on the server
(`bern/handwriting.py:segment_lines` and friends). That code was not ported. If your app
photographs whole pages, use `bernApi.ts` — or crop to a single character before calling
`readGlyph`. Saying this plainly beats shipping a reader that returns one letter for a
sentence.

---

## The two settings that break mobile builds

Both are the platform forbidding plain HTTP. The server runs on HTTP by default, and the app
works in a browser while failing on the device — which is a confusing way to lose an hour.

### iOS — App Transport Security

Since iOS 9, cleartext HTTP is blocked. Add to `ios/<App>/Info.plist`:

```xml
<key>NSAppTransportSecurity</key>
<dict>
  <!-- Development only. Remove this and use HTTPS for a released app. -->
  <key>NSAllowsArbitraryLoads</key>
  <true/>
</dict>
```

### Android — cleartext traffic

Since API 28, cleartext is off by default. In `android/app/src/main/AndroidManifest.xml`:

```xml
<application
    android:usesCleartextTraffic="true"
    ... >
```

Both are development conveniences. For anything you ship, put Bern behind HTTPS (a tunnel or a
real host) and delete these — otherwise you have disabled a transport-security protection for
every request the app makes, not just Bern's.

### `localhost` on a phone is the phone

`http://localhost:8000` points at the handset itself and will always fail. Use the machine's
LAN address. Find it on the machine running the server:

```bash
hostname -I | awk '{print $1}'      # Linux/macOS
ipconfig | findstr IPv4             # Windows
```

Never hardcode it. Read it from config so the same build works against a laptop and a host:

```ts
// Expo
const BASE_URL = process.env.EXPO_PUBLIC_BERN_URL ?? "http://192.168.1.20:8000";

// React Native (react-native-config)
import Config from "react-native-config";
const BASE_URL = Config.BERN_URL ?? "http://192.168.1.20:8000";
```

---

## Camera to text, end to end

`handwriting()` takes a local file URI, which is exactly what an image picker returns.

```bash
npx expo install expo-image-picker
```

```tsx
import * as ImagePicker from "expo-image-picker";
import { useState } from "react";
import { ActivityIndicator, Button, Text, View } from "react-native";
import { BernClient } from "./bern/bernApi";

const bern = new BernClient(process.env.EXPO_PUBLIC_BERN_URL ?? "http://192.168.1.20:8000");

export default function HandwritingScreen() {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);

  async function captureAndRead() {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) return;

    const shot = await ImagePicker.launchCameraAsync({ quality: 0.8 });
    if (shot.canceled || !shot.assets?.[0]?.uri) return;

    setBusy(true);
    try {
      const result = await bern.handwriting(shot.assets[0].uri);
      setText(result.text);
    } catch (error) {
      // The message is the server's own explanation, which is more useful than "Network error".
      setText(error instanceof Error ? error.message : "Could not read the photo.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <View style={{ padding: 20, gap: 12 }}>
      <Button title="Photograph handwriting" onPress={captureAndRead} />
      {busy ? <ActivityIndicator /> : <Text selectable>{text}</Text>}
    </View>
  );
}
```

Send the photo at `quality: 0.8` or lower. The server downscales anything past 1600 px anyway,
so a full-resolution 12 MP capture is bytes you pay for twice — once on the mobile data and
once in the upload.

---

## Chat that remembers

Memory lives on the server and is keyed by `sessionId`, so you must keep it stable across the
conversation and change it to start over.

```ts
const [sessionId, setSessionId] = useState(() => `chat-${Date.now()}`);

await bern.chat("magkano ang bigas", sessionId);
await bern.chat("what about the other one?", sessionId);   // resolves correctly
setSessionId(`chat-${Date.now()}`);                        // "New chat"
```

Show `resolved_as` when you want the interpretation visible — it is how a user learns that
"the other one" was understood as rice.

`engine: "local"` forces the offline path. Leave it at `"auto"`: without a server-side key both
behave identically, and with one you get explanations for free.

---

## Handle a missing server

Phones lose Wi-Fi constantly. Bern is usually on a laptop that may be asleep.

```ts
const error: BernError = ...;
if (error.status === undefined) {
  // No status means the request never reached the server: wrong URL, different network,
  // firewall, or the server is not running.
} else if (error.status === 503) {
  // Tutoring without a configured language model.
}
```

`BernClient` already applies a 30-second timeout, so a dead host fails rather than hangs. Keep
`health()` on screen-start to show a clear "Bern is offline" state instead of letting every
button fail on its own.

Worth knowing before you ship: **the handwriting reader is the only feature that needs the
network for real work.** Prices, nutrition, chat and study packs are also server-side, but
they are cheap. If a large part of your audience has unreliable data, that is an argument for
the hosted deploy in `../SHARE.md` over a laptop on the same Wi-Fi.

---

## What actually works today

Measured on this repo — do not oversell it in the app:

| Feature | State |
| --- | --- |
| Chat, no server | `bernOnDevice.ts`: 24 intents. Returns the intent and the stored answer, not a price |
| One handwritten character, no server | 80.9% per character, measured on 20,000 held-out EMNIST glyphs |
| Parity with the Python path | Features and probabilities agree to float32 rounding; 52/52 glyphs read identically |
| Handwriting across a page | **Server only** — the phone has no line/word/glyph segmenter |
| Block-letter handwriting | Good on clean paper. `. , - ?` now recovered; `ñ` supported |
| Phone photos | Perspective and ruled-line handling exist, **never tested on a real photo** |
| Chat follow-ups | Works, no key needed |
| Explanations / unseen questions | Only with a server-side API key |
| Study packs | Works, but extractive — it quotes your notes rather than explaining them |
| Cursive | **Not supported** — needs the TrOCR backend |
| Non-Latin scripts | **Not supported** |

If someone photographs cursive, they will get nonsense. Say so in the UI rather than letting
them conclude the app is broken.
