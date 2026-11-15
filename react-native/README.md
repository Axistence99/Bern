# Putting Bern in a mobile app

`bernApi.ts` is a complete, dependency-free client for Bern. Copy it into your project and you
have handwriting, chat, study packs, prices, nutrition and peso counting in about ten lines.

```bash
cp react-native/bernApi.ts your-app/src/bern/bernApi.ts
```

There is no npm package to install. The file uses only `fetch` and `FormData`, which React
Native and Expo both provide.

---

## Decide the architecture first

**Recommended: the phone talks to a server running Bern.**

```
[ phone app ]  --HTTP/HTTPS-->  [ Bern server: Python + OpenCV + scikit-learn ]
```

The phone sends a photo, the server runs the model, the phone gets text back. This is what
`bernApi.ts` assumes, and it is the only practical option today.

**Not recommended: running Bern on the device.** It is technically possible (BeeWare,
Chaquopy, pyodide) but you would be shipping a Python runtime, OpenCV and scikit-learn inside
the app — hundreds of MB, slow start, and a build process most teams regret. The models are
only 5.8 MB; the *runtime* is the problem.

The practical consequence: **the app needs a reachable server.** Either one on the same Wi-Fi
(a laptop), or a hosted one. See `../SHARE.md` for how to run it.

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
| Block-letter handwriting | Good on clean paper. `. , - ?` now recovered; `ñ` supported |
| Phone photos | Perspective and ruled-line handling exist, **never tested on a real photo** |
| Chat follow-ups | Works, no key needed |
| Explanations / unseen questions | Only with a server-side API key |
| Study packs | Works, but extractive — it quotes your notes rather than explaining them |
| Cursive | **Not supported** — needs the TrOCR backend |
| Non-Latin scripts | **Not supported** |

If someone photographs cursive, they will get nonsense. Say so in the UI rather than letting
them conclude the app is broken.
