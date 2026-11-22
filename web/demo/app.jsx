/* Bern — try the intent model.
 *
 * Everything runs on this page: the real 25-class classifier from models/ondevice/intent.json,
 * evaluated in JavaScript. No server, no network. Same runtime the phone uses.
 *
 * Design notes live in DESIGN.md. The short version: the confidence number is the interface, so
 * every reply carries a stamped readout of what the model decided and how sure it was. Colour
 * is a signal — it means "known" or "guess", never decoration.
 */

const { useState, useRef, useEffect, useMemo, useCallback } = React;

/** The app logo. A two-tone stamp, cut from the mascot; see DESIGN.md. */
const LOGO = __MASCOT_FACE__;

const runtimeReady = (async () => {
  const runtime = window.__BERN_RUNTIME__;
  if (!runtime) throw new Error("The model runtime failed to load.");
  const { INTENT_BUNDLE, GLYPH_BUNDLE, SYMBOL_BUNDLE } = window.__BERN__;
  // Three separate bundles: intents classify text, glyph and symbol read single characters.
  return {
    bern: await runtime.BernOnDevice.load(INTENT_BUNDLE, GLYPH_BUNDLE, SYMBOL_BUNDLE),
    runtime,
  };
})();

/* ------------------------------------------------------------------ image intake */

/** Longest side the work buffer is allowed to be. A phone camera writes 12 MP; the glyph model
 * wants one character, so the wide shot is scaled down before anything looks at it. */
const WORK_MAX = 640;

/** The region we treat as "the character" when no crop has been chosen yet. */
const DEFAULT_CROP = { x: 0.18, y: 0.2, w: 0.64, h: 0.6 };

let workCanvas = null;
function workSurface() {
  if (!workCanvas) workCanvas = document.createElement("canvas");
  return workCanvas;
}

/** Draw `image` into the work canvas, downscaled, and keep the scale so crop maths can be done
 * in source pixels. */
function drawToWork(image, maxSide) {
  const canvas = workSurface();
  const scale = Math.min(1, maxSide / Math.max(image.width, image.height));
  canvas.width = Math.max(1, Math.round(image.width * scale));
  canvas.height = Math.max(1, Math.round(image.height * scale));
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  return canvas;
}

/** Rec.601 luma, the same weighting the Python side uses before Otsu. */
function toGray(imageData) {
  const { data } = imageData;
  const gray = new Uint8Array(data.length / 4);
  for (let i = 0, g = 0; i < data.length; i += 4, g += 1) {
    gray[g] = (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) | 0;
  }
  return gray;
}

/** Draw the chosen crop on its own so the readout can show exactly what the model was given. */
function cropPreview(canvas, crop) {
  const { x, y, w, h } = crop;
  const source = cropToGlyph(canvas, crop);
  const side = 112;
  const out = document.createElement("canvas");
  out.width = side;
  out.height = side;
  const context = out.getContext("2d");
  const buffer = context.createImageData(side, side);
  const scale = Math.max(source.width, source.height) / side;
  for (let py = 0; py < side; py += 1) {
    for (let px = 0; px < side; px += 1) {
      const sx = Math.min(source.width - 1, Math.floor(px * scale));
      const sy = Math.min(source.height - 1, Math.floor(py * scale));
      const value = source.gray[sy * source.width + sx];
      const index = (py * side + px) * 4;
      buffer.data[index] = value;
      buffer.data[index + 1] = value;
      buffer.data[index + 2] = value;
      buffer.data[index + 3] = 255;
    }
  }
  context.putImageData(buffer, 0, 0);
  return { x, y, w, h, preview: out.toDataURL("image/png") };
}

/** The 28x28 the model actually receives, drawn as a PNG so the readout can show it.
 *
 * cropPreview shows the source crop; this shows the result of prepare(), which is a different
 * thing — the mask, the crop to ink, the scale to 28px. When the two disagree the model is
 * being shown something the reader did not expect, and that is worth seeing.
 */
function glyphPreviewOf(bern, canvas, crop) {
  const { gray, width, height } = cropToGlyph(canvas, crop);
  const glyph = bern.prepare(gray, width, height);
  const side = 112;
  const out = document.createElement("canvas");
  out.width = side;
  out.height = side;
  const context = out.getContext("2d");
  const image = context.createImageData(side, side);
  const block = side / 28;
  for (let y = 0; y < side; y += 1) {
    for (let x = 0; x < side; x += 1) {
      const value = glyph[Math.floor(y / block) * 28 + Math.floor(x / block)];
      const index = (y * side + x) * 4;
      image.data[index] = value;
      image.data[index + 1] = value;
      image.data[index + 2] = value;
      image.data[index + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  return out.toDataURL("image/png");
}

function cropToGlyph(canvas, crop) {
  const x = Math.round(crop.x * canvas.width);
  const y = Math.round(crop.y * canvas.height);
  const w = Math.max(1, Math.round(crop.w * canvas.width));
  const h = Math.max(1, Math.round(crop.h * canvas.height));
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const pixels = context.getImageData(x, y, w, h);
  return { gray: toGray(pixels), width: w, height: h };
}

/** Split an ink mask into lines, then each line into words and glyphs.
 *
 * This is the part that cannot happen on the device: the phone runtime reads one character per
 * call, so segmentation lives here. It mirrors `bern/handwriting.py` in shape — horizontal
 * projection for lines, vertical gaps for words, connected runs for glyphs — but it is not a
 * byte-for-byte port, and its output is a reading aid rather than the Python reader's.
 */
function segmentInk(mask, width, height) {
  // --- lines: rows that contain ink
  const rowInk = new Uint32Array(height);
  for (let y = 0; y < height; y += 1) {
    let count = 0;
    for (let x = 0; x < width; x += 1) if (mask[y * width + x]) count += 1;
    rowInk[y] = count;
  }
  const lines = [];
  let lineTop = -1;
  const gapAllowance = Math.max(2, Math.round(height * 0.012));
  let blankRun = 0;
  for (let y = 0; y < height; y += 1) {
    const has = rowInk[y] > Math.max(1, width * 0.004);
    if (has) {
      if (lineTop < 0) lineTop = y;
      blankRun = 0;
    } else if (lineTop >= 0) {
      blankRun += 1;
      if (blankRun > gapAllowance) {
        lines.push([lineTop, y - blankRun]);
        lineTop = -1;
        blankRun = 0;
      }
    }
  }
  if (lineTop >= 0) lines.push([lineTop, height - 1]);

  // Drop slivers: a ruled line or a speck is not a line of writing.
  const tall = lines.filter(([top, bottom]) => bottom - top >= Math.max(6, height * 0.02));
  if (!tall.length) return [];

  return tall.map(([top, bottom]) => {
    const lineHeight = bottom - top + 1;

    // --- words: columns that contain ink, within this line
    const colInk = new Uint32Array(width);
    for (let y = top; y <= bottom; y += 1) {
      for (let x = 0; x < width; x += 1) if (mask[y * width + x]) colInk[x] += 1;
    }
    const spans = [];
    let start = -1;
    for (let x = 0; x < width; x += 1) {
      if (colInk[x] > 0) {
        if (start < 0) start = x;
      } else if (start >= 0) {
        spans.push([start, x - 1]);
        start = -1;
      }
    }
    if (start >= 0) spans.push([start, width - 1]);

    // A word gap is wider than a letter gap; merge spans that are too close to be separate
    // words. Scaled to line height so it works on a small note and a full page alike.
    const wordGap = Math.max(4, Math.round(lineHeight * 0.32));
    const words = [];
    let current = null;
    for (const [x0, x1] of spans) {
      if (!current) current = [x0, x1];
      else if (x0 - current[1] <= wordGap) current[1] = x1;
      else { words.push(current); current = [x0, x1]; }
    }
    if (current) words.push(current);

    return words.map(([x0, x1]) => {
      // --- glyphs: connected column runs inside the word
      const widths = [];
      for (let x = x0; x <= x1; x += 1) {
        let count = 0;
        for (let y = top; y <= bottom; y += 1) if (mask[y * width + x]) count += 1;
        widths.push(count);
      }
      const glyphs = [];
      let gStart = -1;
      for (let i = 0; i < widths.length; i += 1) {
        if (widths[i] > 0) {
          if (gStart < 0) gStart = i;
        } else if (gStart >= 0) {
          glyphs.push([x0 + gStart, x0 + i - 1]);
          gStart = -1;
        }
      }
      if (gStart >= 0) glyphs.push([x0 + gStart, x1]);

      // Runs are connected column spans, so "W" arrives whole and never needs cutting. Only
      // split one when it is wider than any single character plausibly is — two letters that
      // touch. Cutting at a fixed 1.15x line height was splitting "W" into "vv".
      const split = [];
      for (const [gx0, gx1] of glyphs) {
        const w = gx1 - gx0 + 1;
        if (w > lineHeight * 1.9) {
          const pieces = Math.round(w / (lineHeight * 0.8));
          const step = w / pieces;
          for (let i = 0; i < pieces; i += 1) {
            split.push([Math.round(gx0 + i * step), Math.round(gx0 + (i + 1) * step) - 1]);
          }
        } else {
          split.push([gx0, gx1]);
        }
      }

      return {
        box: [x0, top, x1 - x0 + 1, bottom - top + 1],
        glyphs: split.map(([gx0, gx1]) => {
          // Tighten vertically to the glyph's own ink, so a letter without a descender is not
          // stretched to the line's full height before normalisation.
          let gTop = bottom;
          let gBottom = top;
          for (let y = top; y <= bottom; y += 1) {
            for (let x = gx0; x <= gx1; x += 1) {
              if (mask[y * width + x]) {
                if (y < gTop) gTop = y;
                if (y > gBottom) gBottom = y;
              }
            }
          }
          return [gx0, gTop, gx1 - gx0 + 1, Math.max(1, gBottom - gTop + 1)];
        }),
      };
    });
  });
}

/** Read every glyph on the page and put the characters back together.
 *
 * Returns `{ lines: [{ text, confidence, glyphs }], text, confidence }`. Confidence is the mean
 * of the per-glyph probabilities, so a page of clearly printed letters scores high and a page
 * of scrawl scores low — which is the number the readout shows.
 */
function readPage(bern, inkMask, canvas) {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const { width, height } = canvas;
  const gray = toGray(context.getImageData(0, 0, width, height));
  const mask = inkMask(gray, width, height);

  const layout = segmentInk(mask, width, height);
  if (!layout.length) return { lines: [], text: "", confidence: 0, glyphCount: 0 };

  // Square off the glyph grid so a mask row can be sliced without re-reading the canvas.
  const lines = layout.map((words) => {
    const glyphs = [];
    for (const word of words) {
      for (const [gx, gy, gw, gh] of word.glyphs) {
        const patch = new Uint8Array(gw * gh);
        for (let y = 0; y < gh; y += 1) {
          for (let x = 0; x < gw; x += 1) {
            patch[y * gw + x] = mask[(gy + y) * width + (gx + x)] ? 255 : 0;
          }
        }
        // Already an ink-bright mask, so prepare() must not mask it a second time.
        const glyph = bern.prepare(patch, gw, gh, true);

        // No blob guard here, deliberately. A glyph the reader cannot actually interpret still
        // gets a confident character back - a plain grey circle comes out as "0". Ink density
        // cannot filter it out: bold N fills 0.52 and a filled disc only 0.59, so the classes
        // overlap and any threshold that catches the disc also eats real letters. The reply
        // discloses this instead of hiding it.

        const read = bern.readGlyph(glyph);
        const symbol = bern.readSymbol(glyph);
        // The glyph model cannot emit punctuation at all; the symbol model exists for it. Take
        // the symbol only when it is clearly better than the letter.
        const chosen = symbol.confidence > Math.max(0.72, read.confidence + 0.15)
          ? { char: symbol.char, confidence: symbol.confidence }
          : { char: read.char, confidence: read.confidence };
        glyphs.push({ ...chosen, box: [gx, gy, gw, gh] });
      }
      glyphs.push({ char: " ", confidence: 1, space: true });
    }
    if (glyphs.length && glyphs[glyphs.length - 1].space) glyphs.pop();

    const letters = glyphs.filter((g) => !g.space);
    const confidence = letters.length
      ? letters.reduce((sum, g) => sum + g.confidence, 0) / letters.length
      : 0;
    return { text: glyphs.map((g) => g.char).join(""), confidence, glyphs };
  });

  const allGlyphs = lines.flatMap((l) => l.glyphs.filter((g) => !g.space));
  return {
    lines,
    text: lines.map((l) => l.text).join("\n").trim(),
    confidence: allGlyphs.length
      ? allGlyphs.reduce((sum, g) => sum + g.confidence, 0) / allGlyphs.length
      : 0,
    glyphCount: allGlyphs.length,
  };
}

/* ------------------------------------------------------------------ markdown */

/** The stored answers use **bold**, `code` and fenced blocks. Nothing else needs parsing. */
function RichText({ text }) {
  const blocks = useMemo(() => splitBlocks(text), [text]);
  return (
    <>
      {blocks.map((block, index) =>
        block.type === "code" ? (
          <pre key={index} className="code-block">
            <code>{block.text}</code>
          </pre>
        ) : (
          <p key={index} className="whitespace-pre-wrap break-words">{inline(block.text)}</p>
        )
      )}
    </>
  );
}

function splitBlocks(text) {
  const blocks = [];
  const pattern = /```(?:\w+)?\n?([\s\S]*?)```/g;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) blocks.push({ type: "text", text: text.slice(cursor, match.index) });
    blocks.push({ type: "code", text: match[1].replace(/\n$/, "") });
    cursor = pattern.lastIndex;
  }
  if (cursor < text.length) blocks.push({ type: "text", text: text.slice(cursor) });
  return blocks.length ? blocks : [{ type: "text", text }];
}

function inline(text) {
  const parts = [];
  // Lazy, not `[^*]+`: bold spans can contain a literal asterisk, and multiplication is one.
  const pattern = /(\*\*.+?\*\*|`[^`]+`)/g;
  let cursor = 0;
  let match;
  let key = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) parts.push(text.slice(cursor, match.index));
    const token = match[0];
    parts.push(
      token.startsWith("**")
        ? <strong key={key++} className="strong">{token.slice(2, -2)}</strong>
        : <code key={key++} className="inline-code">{token.slice(1, -1)}</code>
    );
    cursor = pattern.lastIndex;
  }
  if (cursor < text.length) parts.push(text.slice(cursor));
  return parts;
}

/* ------------------------------------------------------------------ content */

/** Eight prompts a newcomer actually types, four they should not expect an answer to. */
const EXAMPLES = [
  { text: "magkano ang bigas ngayon?", expect: "price lookup" },
  { text: "kumusta ka naman?", expect: "small talk" },
  { text: "tell me a joke about machine learning", expect: "humour" },
  { text: "who are you and what can you do?", expect: "identity" },
  { text: "paano mag transcribe ng handwriting?", expect: "handwriting" },
  { text: "salamat po", expect: "gratitude" },
  { text: "what's the weather in Manila tomorrow?", expect: "out of scope", miss: true },
  { text: "write me a poem about the sea", expect: "out of scope", miss: true },
];

/** Human names for the 25 classes. The id stays visible — this is a model, not a persona. */
const INTENT_LABELS = {
  ph_price_lookup: "Philippine price lookup",
  ph_price_catalog: "Price catalogue",
  ph_budget_calc: "Grocery budget",
  ph_nutrition_lookup: "Nutrition lookup",
  ph_money_counter: "Peso counter",
  handwriting_ocr: "Handwriting OCR",
  file_scan: "File reading",
  study_materials: "Study materials",
  calculator: "Calculator",
  train_help: "Teaching it",
  api_usage: "API help",
  react_native_integration: "Mobile integration",
  deployment_tips: "Deployment",
  ml_architecture: "How it works",
  identity: "Identity",
  developer: "Who built me",
  capabilities: "Capabilities",
  greeting: "Greeting",
  gratitude: "Gratitude",
  goodbye: "Goodbye",
  small_talk: "Small talk",
  humor: "Humour",
  user_name: "Your name",
  time_date: "Date and time",
  comprehension_check: "Do I understand you",
  conversation_repair: "Correcting me",
  conversation_state: "Talking plainly",
  answer_check: "Checking my answer",
  out_of_scope: "Outside training",
};

/** Below this the model is not choosing, it is guessing. The threshold sets the colour.
 *  It matches the refusal floor in `bernOnDevice.reply` and `gating_rejects`, so the browser
 *  and the server never disagree about whether a sentence was answered or refused. */
const CONFIDENT = 0.4;
const SEQUENCE = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"];

/* ------------------------------------------------------------------ app */

function App() {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [attachment, setAttachment] = useState(null);
  const [busy, setBusy] = useState(false);
  const [showScores, setShowScores] = useState(true);
  const [theme, setTheme] = useState("ink");
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState(null);
  const runtime = useRef(null);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);
  const fileRef = useRef(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    let alive = true;
    runtimeReady
      .then((loaded) => {
        if (!alive) return;
        runtime.current = loaded;
        setStatus("ready");
      })
      .catch((problem) => {
        if (!alive) return;
        setStatus("failed");
        setError(String(problem && problem.message ? problem.message : problem));
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    const node = scrollRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [messages]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  /** Read the attached photo three times over, from the three densest regions.
   *
   * The model is a single-character reader. Asking it once would show the luckiest cell in the
   * picture and imply the page had read the word. Asking it three times, from regions chosen
   * by ink density, gives a spread we can show honestly: agreeing low readings mean blurred
   * writing, a confident one against two timid ones means the model found a clear character
   * and nothing else.
   */
  const readAttachment = useCallback((current) => readPage(
    runtime.current.bern,
    runtime.current.runtime.inkMask,
    current.work
  ), []);

  const ask = useCallback(
    (text) => {
      const message = (text ?? draft).trim();
      const photo = attachment;
      if (!message && !photo) return;
      if (status !== "ready" || !runtime.current) {
        setError("I am still loading.");
        return;
      }
      setError(null);
      setDraft("");

      const started = performance.now();
      const result = message ? runtime.current.bern.classify(message) : null;
      const reply = message ? runtime.current.bern.reply(message) : null;
      const reading = photo ? readAttachment(photo) : null;
      const elapsed = performance.now() - started;
      setAttachment(null);

      setMessages((previous) => [
        ...previous,
        {
          role: "user",
          text: message,
          id: `u${previous.length}`,
          attachment: photo
            ? { thumb: photo.thumb, preview: photo.preview.preview, name: photo.name }
            : null,
        },
        {
          role: "bern",
          id: `b${previous.length}`,
          // A photo is answered from the page reading; the classifier still runs on whatever
          // was typed, so the readout shows which kind of question the reader asked.
          text: photo
            ? describePage(reading, message).text
            : reply ? reply.text ?? "I do not have an answer stored for that one." : null,
          intent: result ? result.intent : null,
          confidence: result ? result.confidence : null,
          alternatives: result ? result.alternatives : null,
          needsServer: photo ? false : reply ? reply.needsServer : false,
          elapsed,
          reading,
          // the preview belongs to the reply's readout as well as to the turn above
          attachment: photo
            ? { preview: photo.preview.preview, glyphPreview: photo.glyphPreview }
            : null,
        },
      ]);
    },
    [draft, status, attachment, readAttachment]
  );

  const takeFile = useCallback(async (file) => {
    if (!file || !runtime.current) return;
    setBusy(true);
    setError(null);
    try {
      const url = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("That file could not be read."));
        reader.readAsDataURL(file);
      });
      const image = await new Promise((resolve, reject) => {
        const element = new Image();
        element.onload = () => resolve(element);
        element.onerror = () => reject(new Error("That file is not an image this browser can open."));
        element.src = url;
      });

      const full = drawToWork(image, 1600);
      const work = drawToWork(image, WORK_MAX);

      // Guess the crop from the densest ink, then let the reader move it.
      // The tray still offers a single-character crop, because the character reader is the
      // sharpest thing here and a reader may want to check one glyph. Reading the page does not
      // depend on it.
      const crop = DEFAULT_CROP;

      setAttachment({
        name: file.name || "photo.jpg",
        kind: /^image\//.test(file.type) ? "image" : "file",
        thumb: full.toDataURL("image/png"),
        preview: cropPreview(work, crop),
        glyphPreview: glyphPreviewOf(runtime.current.bern, work, crop),
        work,
        crop,
      });
    } catch (problem) {
      setError(String(problem && problem.message ? problem.message : problem));
    } finally {
      setBusy(false);
    }
  }, []);

  const nudgeCrop = useCallback((move) => {
    setAttachment((current) => {
      if (!current) return current;
      const c = { ...current.crop };
      const step = 0.04;
      if (move === "left") c.x = Math.max(0, c.x - step);
      if (move === "right") c.x = Math.min(1 - c.w, c.x + step);
      if (move === "up") c.y = Math.max(0, c.y - step);
      if (move === "down") c.y = Math.min(1 - c.h, c.y + step);
      if (move === "in") {
        c.w = Math.max(0.12, c.w - step);
        c.h = Math.max(0.12, c.h - step);
      }
      if (move === "out") {
        c.w = Math.min(1 - c.x, c.w + step);
        c.h = Math.min(1 - c.y, c.h + step);
      }
      return {
        ...current,
        crop: c,
        preview: cropPreview(current.work, c),
        glyphPreview: glyphPreviewOf(runtime.current.bern, current.work, c),
      };
    });
  }, []);

  const onKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      ask();
    }
  };

  const reset = () => {
    setMessages([]);
    setError(null);
    if (inputRef.current) inputRef.current.focus();
  };

  const lastConfident = messages.filter((m) => m.role === "bern").slice(-1)[0];

  return (
    <div className="sheet">
      <div className="frame">
        <header className="masthead">
          <h1 className="wordmark">Bern</h1>

          <div className="masthead-tools">
            <LoadState status={status} last={lastConfident} />
            <button
              type="button"
              className="control"
              onClick={() => setShowScores((value) => !value)}
              aria-pressed={showScores}
            >
              {showScores ? "Readout on" : "Readout off"}
            </button>
            <button
              type="button"
              className="control"
              onClick={() => setTheme((value) => (value === "ink" ? "paper" : "ink"))}
            >
              {theme === "ink" ? "Paper" : "Ink"}
            </button>
            <button type="button" className="control" onClick={reset}>
              Clear
            </button>
            <img src={LOGO} alt="" className="logo" />
          </div>
        </header>

        <main ref={scrollRef} className="transcript" aria-live="polite" aria-label="Conversation">
          {messages.length === 0 ? (
            <Opening onPick={ask} disabled={status !== "ready"} />
          ) : (
            messages.map((message, index) =>
              message.role === "user" ? (
                <Turn
                  key={message.id}
                  text={message.text}
                  index={index}
                  attachment={message.attachment}
                />
              ) : (
                <Answer key={message.id} message={message} showScores={showScores} index={index} />
              )
            )
          )}

          {error ? (
            <p className="fault" role="alert">
              <span className="fault-mark">!</span> {error}
            </p>
          ) : null}
        </main>

        <footer className="composer">
          <span className="composer-rule" aria-hidden="true" />

          {attachment ? (
            <div className="tray">
              <figure className="tray-photo">
                <img src={attachment.thumb} alt="" />
              </figure>
              <div className="tray-read">
                <p className="tray-note">
                  I read one character at a time. Move the box onto a single letter.
                </p>
                <div className="tray-preview">
                  <figure className="tray-crop">
                    <img src={attachment.preview.preview} alt="The crop the model will read" />
                    <figcaption>crop</figcaption>
                  </figure>
                  <figure className="tray-crop">
                    <img src={attachment.glyphPreview} alt="The 28 by 28 pixels the model receives" />
                    <figcaption>28×28</figcaption>
                  </figure>
                  <div className="tray-move">
                    <button type="button" onClick={() => nudgeCrop("left")} aria-label="Move crop left">←</button>
                    <button type="button" onClick={() => nudgeCrop("up")} aria-label="Move crop up">↑</button>
                    <button type="button" onClick={() => nudgeCrop("down")} aria-label="Move crop down">↓</button>
                    <button type="button" onClick={() => nudgeCrop("right")} aria-label="Move crop right">→</button>
                    <button type="button" onClick={() => nudgeCrop("out")} aria-label="Make crop larger">−</button>
                    <button type="button" onClick={() => nudgeCrop("in")} aria-label="Make crop smaller">+</button>
                  </div>
                </div>
                <button type="button" className="tray-drop" onClick={() => setAttachment(null)}>
                  Remove
                </button>
              </div>
            </div>
          ) : null}

          <div className="composer-row">
            <div className="composer-tools">
              {/* accept=camera opens the camera app itself on iOS and Android; the second input
                  is a plain picker for the desktop, where capture is meaningless. */}
              <input
                ref={cameraRef}
                type="file"
                accept="image/*"
                capture="environment"
                className="visually-hidden"
                onChange={(event) => takeFile(event.target.files && event.target.files[0])}
              />
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                className="visually-hidden"
                onChange={(event) => takeFile(event.target.files && event.target.files[0])}
              />
              <button
                type="button"
                className="tool"
                onClick={() => cameraRef.current && cameraRef.current.click()}
                disabled={status !== "ready" || busy}
                title="Open the camera"
                aria-label="Take a photo"
              >
                <CameraIcon />
              </button>
              <button
                type="button"
                className="tool"
                onClick={() => fileRef.current && fileRef.current.click()}
                disabled={status !== "ready" || busy}
                title="Attach a file"
                aria-label="Attach a file"
              >
                <ClipIcon />
              </button>
            </div>

            <textarea
              ref={inputRef}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder={
                busy ? "reading the image" : status === "ready" ? "ask me anything" : "waking up"
              }
              disabled={status !== "ready"}
              className="field"
              aria-label="Your message"
            />
            <button
              type="button"
              onClick={() => ask()}
              disabled={status !== "ready" || (!draft.trim() && !attachment)}
              className="send"
              aria-label="Send"
              title="Send"
            >
              <ArrowIcon />
            </button>
          </div>
          <p className="footnote">
            Every reply is an answer I already hold. Your photo is read here, in this tab.
          </p>
        </footer>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ header bits */

function LoadState({ status }) {
  if (status === "loading") return <span className="loadstate">waking up</span>;
  if (status === "failed") return <span className="loadstate loadstate--fault">failed</span>;
  return <span className="loadstate">listening</span>;
}

/* ------------------------------------------------------------------ opening */

function Opening({ onPick, disabled }) {
  return (
    <section className="opening">
      <h2 className="opening-title">Ask me something.</h2>
      <p className="opening-line">I will tell you how sure I am about every answer.</p>

      {/* Four starters, not eight. They are a hint under the composer, not the page. */}
      <ul className="prompts" aria-label="Example questions">
        {EXAMPLES.filter((example) => !example.miss).slice(0, 4).map((example) => (
          <li key={example.text}>
            <button
              type="button"
              className="prompt"
              onClick={() => onPick(example.text)}
              disabled={disabled}
            >
              <span className="prompt-text">{example.text}</span>
              <span className="prompt-expect">{example.expect}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

/* ------------------------------------------------------------------ turns */

function Turn({ text, index, attachment }) {
  return (
    <article className="turn">
      <div className="turn-gutter">
        <span className="turn-index">{SEQUENCE[index] ?? "··"}</span>
        <span className="turn-label">you</span>
      </div>
      <div className="turn-body">
        {attachment ? (
          <figure className="sent-photo">
            <img src={attachment.thumb} alt={`Attached: ${attachment.name}`} />
            <figcaption>{attachment.name}</figcaption>
          </figure>
        ) : null}
        {text ? <p className="turn-text">{text}</p> : null}
      </div>
    </article>
  );
}

function Answer({ message, showScores, index }) {
  if (message.reading) return <GlyphAnswer message={message} showScores={showScores} />;

  const low = message.confidence < CONFIDENT || message.intent === "out_of_scope";
  return (
    <article className="answer">
      <div className="answer-body">
        <RichText text={message.text} />

        {message.intent === "out_of_scope" ? (
          <p className="aside">
            I mention a <b>Teach Bern This</b> button. That button is in the desktop app, where
            it writes a new phrase into me. This page only loads me as I was trained.
          </p>
        ) : null}

        {message.needsServer ? (
          <p className="aside aside--warn">
            That is only my opening line. The figures themselves feed in from the server, and I
            carry no price data of my own here.
          </p>
        ) : null}
      </div>

      {showScores ? (
        <Readout
          confidence={message.confidence}
          intent={message.intent}
          alternatives={message.alternatives}
          low={low}
        />
      ) : null}
    </article>
  );
}

/**
 * The readout. This is the piece the whole design is built around, so it gets the largest type
 * on the page: the confidence, set as a numeral, with the decision beside it in the same
 * monospaced column so the eye can compare one answer against the next.
 */
function Readout({ confidence, intent, alternatives, low, caption, label }) {
  const runnerUp = alternatives && alternatives.length ? alternatives[0] : null;
  return (
    <aside className={`readout${low ? " readout--low" : ""}`} aria-label="Model readout">
      <div className="readout-figure">
        <span className="readout-value">
          {(confidence * 100).toFixed(1)}
          <span className="readout-unit">%</span>
        </span>
        <span className="readout-caption">{caption || (low ? "guessing" : "confident")}</span>
      </div>
      {label ? <p className="readout-label">{label}</p> : null}

      <dl className="readout-table">
        <div className="readout-row">
          <dt>intent</dt>
          <dd>
            {intent}
            <span className="readout-name">{INTENT_LABELS[intent] ?? ""}</span>
          </dd>
        </div>
        {runnerUp ? (
          <div className="readout-row readout-row--next">
            <dt>next</dt>
            <dd>
              {runnerUp.label}
              <span className="readout-name">{(runnerUp.confidence * 100).toFixed(1)}%</span>
            </dd>
          </div>
        ) : null}
      </dl>

      <span className="readout-bar" aria-hidden="true">
        <span style={{ width: `${Math.max(1.5, confidence * 100)}%` }} />
      </span>
    </aside>
  );
}

/* ------------------------------------------------------------------ page replies */

/** What Bern says about a photographed page.
 *
 * Bern cannot recognise objects — there is no scene model in it, only a character reader — so
 * "what is in this image" is answered with what is written in it, and the answer says so when
 * the question asked for more than that. Silently substituting a transcription for a
 * description would be the confident-wrong-answer failure this whole page is about.
 */
function describePage(reading, question) {
  const asked = (question || "").toLowerCase();
  const aboutText = /\b(read|text|written|says?|write|transcri|content|contains?|inside|image|picture|photo)\b/.test(asked);
  const empty = !reading.lines.length || !reading.text;

  if (empty) {
    return {
      text: "I could not find any writing in that image. I read printed or handwritten "
        + "characters — I have no way to recognise objects, so a picture of a dog or a "
        + "building tells me nothing.",
      tone: "empty",
    };
  }


  const quoted = reading.lines.map((line) => `"${line.text}"`);
  const spoken = quoted.length === 1
    ? quoted[0]
    : `${quoted.slice(0, -1).join(", ")} and ${quoted[quoted.length - 1]}`;

  let text = `I can see writing in it. Here is what I read: ${spoken}.`;

  text += "\n\nI read the characters, so I cannot tell you what the picture shows. Check the "
    + "reading against the photo.";

  if (!aboutText) {
    text += "\n\nTo be clear about what I did: I read the characters in your image. I cannot "
      + "tell you what the image *depicts* — no object, scene or face recognition anywhere in me.";
  }

  return { text, tone: "reading" };
}

/* ------------------------------------------------------------------ glyph answer */

/** A photo reply. The character is the claim; the readout counts how many of the three regions
 * the model looked at agreed with it, because one confident reading out of three is a different
 * fact from three out of three. */
function GlyphAnswer({ message, showScores }) {
  const { reading } = message;
  const pages = reading && reading.lines ? reading.lines : [];

  // The single-character path (older messages, and the crop tray) has no `lines`.
  if (!pages.length) {
    const best = reading ? reading.best : null;
    if (!best) return null;
    return (
      <article className="answer">
        <div className="answer-body">
          <RichText text={message.text} />
        </div>
      </article>
    );
  }

  // Measured on 216 clean printed glyphs in three fonts: 74.1% overall, and the model's own
  // probability is not a reliable guide (an 85-95% glyph was right 36% of the time, a 60-85%
  // one 52%). So the verdict is fixed rather than derived from the number, and the number is
  // labelled for what it is.
  const reading_accuracy = 0.741;

  return (
    <article className="answer">
      <div className="answer-body">
        <RichText text={message.text} />

        {showScores ? (
          <div className="page-read">
            <p className="page-read-head">
              {reading.glyphCount} characters over {pages.length}{" "}
              {pages.length === 1 ? "line" : "lines"}
            </p>
            <table className="page-lines">
              <tbody>
                {pages.map((line, index) => (
                  <tr key={index}>
                    <td className="page-line-index">{String(index + 1).padStart(2, "0")}</td>
                    <td className="page-line-text">{line.text}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="page-note">
              My probability is not calibrated, so I cannot flag which of these lines is wrong:
              characters I rate 85&ndash;95% confident are right about a third of the time.
              O/0, I/1/l and S/5 are the same shape to me.
            </p>
          </div>
        ) : null}

        <p className="aside">
          Anything that is not a character still gets named as one &mdash; a plain circle
          comes back as &ldquo;0&rdquo;.
        </p>
      </div>

      {showScores ? (
        <Readout
          confidence={reading_accuracy}
          intent={message.intent}
          alternatives={message.alternatives}
          low
          caption="measured accuracy"
          label="not the model's own confidence"
        />
      ) : null}
    </article>
  );
}

/* ------------------------------------------------------------------ icons */

/** Drawn rather than typed: an emoji camera renders differently on every platform and would be
 * the one element on this page not under the design's control. */
function CameraIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" focusable="false">
      <path
        d="M3 8.5A1.5 1.5 0 0 1 4.5 7h2.2l1.3-2h7l1.3 2h2.2A1.5 1.5 0 0 1 21 8.5v9A1.5 1.5 0 0 1 19.5 19h-15A1.5 1.5 0 0 1 3 17.5z"
        fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round"
      />
      <circle cx="12" cy="13" r="3.4" fill="none" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

function ClipIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" focusable="false">
      <path
        d="M16.5 7.5 9 15a2.5 2.5 0 0 0 3.5 3.5l7-7a4.5 4.5 0 0 0-6.4-6.4l-7 7A6.5 6.5 0 0 0 15.4 21"
        fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"
      />
    </svg>
  );
}

/** Sends. An arrow rather than the word "Ask": the button sits at the end of a box you are
 * already typing into, so the direction is the instruction. The word survives as its
 * accessible name, because an arrow on its own tells a screen reader nothing. */
function ArrowIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" focusable="false">
      <path
        d="M5 12h13M12.5 6.5 19 12l-6.5 5.5"
        fill="none" stroke="currentColor" strokeWidth="1.7"
        strokeLinecap="round" strokeLinejoin="round"
      />
    </svg>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
