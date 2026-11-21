/* Bern — try the intent model.
 *
 * Everything runs on this page: the real 24-class classifier from models/ondevice/intent.json,
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
  const bundle = window.__BERN__.INTENT_BUNDLE;
  return { bern: await runtime.BernOnDevice.load(bundle, bundle, bundle), runtime };
})();

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
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
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

/** Human names for the 24 classes. The id stays visible — this is a model, not a persona. */
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
  capabilities: "Capabilities",
  greeting: "Greeting",
  gratitude: "Gratitude",
  goodbye: "Goodbye",
  small_talk: "Small talk",
  humor: "Humour",
  user_name: "Your name",
  time_date: "Date and time",
  out_of_scope: "Outside training",
};

/** Below this the model is not choosing, it is guessing. The threshold sets the colour. */
const CONFIDENT = 0.5;
const SEQUENCE = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"];

/* ------------------------------------------------------------------ app */

function App() {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [showScores, setShowScores] = useState(true);
  const [theme, setTheme] = useState("ink");
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState(null);
  const runtime = useRef(null);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

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

  const ask = useCallback(
    (text) => {
      const message = (text ?? draft).trim();
      if (!message) return;
      if (status !== "ready" || !runtime.current) {
        setError("The model is still loading.");
        return;
      }
      setError(null);
      setDraft("");

      const started = performance.now();
      const result = runtime.current.bern.classify(message);
      const reply = runtime.current.bern.reply(message);
      const elapsed = performance.now() - started;

      setMessages((previous) => [
        ...previous,
        { role: "user", text: message, id: `u${previous.length}` },
        {
          role: "bern",
          id: `b${previous.length}`,
          text: reply.text ?? "No stored answer for this intent.",
          intent: result.intent,
          confidence: result.confidence,
          alternatives: result.alternatives,
          needsServer: reply.needsServer,
          elapsed,
        },
      ]);
    },
    [draft, status]
  );

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
          <div className="masthead-identity">
            <h1 className="wordmark">Bern</h1>
            <p className="strapline">
              24-class intent classifier &middot; 18,898 features &middot; runs in this tab
            </p>
          </div>

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
                <Turn key={message.id} text={message.text} index={index} />
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
          <div className="composer-row">
            <textarea
              ref={inputRef}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder={status === "ready" ? "Type a message" : "Loading model"}
              disabled={status !== "ready"}
              className="field"
              aria-label="Your message"
            />
            <button
              type="button"
              onClick={() => ask()}
              disabled={status !== "ready" || !draft.trim()}
              className="send"
            >
              Ask
            </button>
          </div>
          <p className="footnote">
            Every reply is the model's own stored answer. It recognises the question; it does not
            fetch anything.
          </p>
        </footer>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ header bits */

function LoadState({ status, last }) {
  if (status === "loading") {
    return <span className="loadstate loadstate--wait">loading&nbsp;weights</span>;
  }
  if (status === "failed") {
    return <span className="loadstate loadstate--fault">model failed</span>;
  }
  return (
    <span className="loadstate">
      model ready
      {last ? <span className="loadstate-last"> &middot; {last.elapsed.toFixed(1)} ms</span> : null}
    </span>
  );
}

/* ------------------------------------------------------------------ opening */

function Opening({ onPick, disabled }) {
  return (
    <section className="opening">
      <p className="eyebrow">A measurement, not a demo reel</p>
      <h2 className="opening-title">
        Ask it something.
        <span className="opening-sub">
          Then read the number under the answer.
        </span>
      </h2>
      <p className="opening-body">
        Bern has 24 intents. It picks one and answers with the reply it was trained to give.
        When it is guessing it says so, and the confidence falls through the floor — which is
        the interesting part.
      </p>

      <ul className="prompts">
        {EXAMPLES.map((example, index) => (
          <li key={example.text}>
            <button
              type="button"
              className={`prompt${example.miss ? " prompt--miss" : ""}`}
              onClick={() => onPick(example.text)}
              disabled={disabled}
            >
              <span className="prompt-index">{SEQUENCE[index]}</span>
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

function Turn({ text, index }) {
  return (
    <article className="turn">
      <div className="turn-gutter">
        <span className="turn-index">{SEQUENCE[index] ?? "··"}</span>
        <span className="turn-label">you</span>
      </div>
      <p className="turn-text">{text}</p>
    </article>
  );
}

function Answer({ message, showScores, index }) {
  const low = message.confidence < CONFIDENT || message.intent === "out_of_scope";
  return (
    <article className="answer">
      <div className="answer-body">
        <RichText text={message.text} />

        {message.intent === "out_of_scope" ? (
          <p className="aside">
            It mentions a <b>Teach Bern This</b> button. That is in the desktop app, where the
            phrase is written into the model. This page only loads a trained one.
          </p>
        ) : null}

        {message.needsServer ? (
          <p className="aside aside--warn">
            That is only Bern's opening line. The figures come from the server, from data this
            page does not carry.
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
function Readout({ confidence, intent, alternatives, low }) {
  const runnerUp = alternatives && alternatives.length ? alternatives[0] : null;
  return (
    <aside className={`readout${low ? " readout--low" : ""}`} aria-label="Model readout">
      <div className="readout-figure">
        <span className="readout-value">
          {(confidence * 100).toFixed(1)}
          <span className="readout-unit">%</span>
        </span>
        <span className="readout-caption">{low ? "guessing" : "confident"}</span>
      </div>

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

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
