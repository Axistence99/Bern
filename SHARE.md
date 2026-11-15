# Letting your friends try Bern

Three ways to do this, from easiest to most involved. Most people want the first one.

---

## 1. Same network, no install (best for classmates in one room)

Nothing to send but your IP address. They open a browser.

```bash
python3 -m pip install -r requirements.txt
python3 server.py
```

The server binds `0.0.0.0` and prints a port (8000 by default). Ask the phone or laptop that
is running it for its local address:

```bash
# Linux / macOS
hostname -I | awk '{print $1}'

# Windows
ipconfig | findstr IPv4
```

You will get something like `192.168.1.20`. Your friend opens **`http://192.168.1.20:8000`** on
the same Wi-Fi. That is the whole setup.

Notes:

- Both devices must be on the same network. Phone hotspot works too — connect both to it.
- Windows will ask to allow Python through the firewall; allow it on private networks.
- Nothing is exposed to the internet this way. It only works on that network.

**A demo is built in.** Open the Handwriting tab and click **"Load a sample"**. It loads a real
handwriting image that Bern reads back exactly, so nobody has to find a photo of their own to
see it work.

---

## 2. Docker (best if your friend is technical, or you want a permanent link)

```bash
docker build -t bern .
docker run -p 8000:8000 bern
```

Same address as above. To share it beyond your network, put it behind a tunnel:

```bash
# any of these, whichever you have
cloudflared tunnel --url http://localhost:8000
ngrok http 8000
```

Either gives you a public `https://...` URL you can send in a chat. It stays up as long as the
tunnel and the container run; close the terminal and the link dies.

> **Before you share a public link, read `SECURITY` below.**

---

## 3. A hosted deploy (best if this should just exist)

Bern binds `0.0.0.0` and reads `PORT` from the environment, so it runs unchanged on any
container host — Render, Railway, Fly.io, a VPS. Point the platform at the `Dockerfile`.

Two things to expect:

- **Cold start.** `models/bern_model.joblib` is not committed (it is rebuilt from
  `data/bern_dataset.json`). The image trains it at build time, so this is handled; deploying
  from source without building the image costs ~10 seconds on first request.
- **44 MB clone.** Most of that is `data/external/wfp_food_prices_phl.csv` (32 MB). If deploy
  size matters, remove it — Bern's own prices come from `data/ph_prices.json`.

---

## What your friends can actually do with it

Worth knowing before you send a link, so nobody is disappointed:

| Works well | Works, with a caveat | Does not work |
| --- | --- | --- |
| Ask about PH food prices | Handwriting — block letters, clean paper | Cursive |
| Nutrition of Filipino dishes | Study packs from notes (extractive, no explanations) | Reading *meaning* — it cannot explain why |
| Build flashcards/quiz from notes | Photos — perspective and ruled lines handled, but never tested on a real photo | Tutoring (needs an API key) |
| Read block-letter handwriting | | Non-Latin scripts |

The honest framing: **Bern reads and looks up. It does not reason.** It will tell you a kilo of
rice costs a certain amount and quiz you on your notes. It will not explain why prices rose.

---

## SECURITY — read before exposing it publicly

Bern has **no authentication**. Every endpoint is open to anyone who has the URL:

- `POST /api/train` **rewrites the model** and persists to `data/bern_dataset.json`
- `DELETE /api/intents/<tag>` deletes skills
- `/api/handwriting` and `/api/study` accept file uploads with **no size limit configured** —
  Tornado's default maximum request body applies, and nothing in this repo sets it

Verified by reading the handler: `TrainHandler.post` checks nothing before calling
`bern_ai.teach`, and no handler authenticates. `parse_json_body` decodes whatever arrives.

Someone who finds a public URL can teach Bern nonsense or delete its intents. This is fine for
a demo in a classroom, and **not** fine for a permanent public deployment without a reverse
proxy in front of it that requires a login, or at minimum blocking `/api/train` and the
`DELETE` routes.

There is also a cost line if you enable the language model: `BERN_LLM_API_KEY` spends real
money per call, and an open endpoint means strangers spending it.

---

## The one-line version to send a friend

> Bern — a small AI that reads Filipino/English handwriting and turns notes into quizzes.
> Runs entirely on our machine, no internet needed. Open http://192.168.1.20:8000 and click
> the Handwriting tab → "Load a sample".

Adjust the address. That's it.
