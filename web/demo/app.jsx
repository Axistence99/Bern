/* Bern — try it in the browser.
 *
 * Everything here runs on this page: the intent model is the real one from
 * models/ondevice/intent.json, evaluated in JavaScript. No server, no network. That is the
 * point of the demo — it is the same code path the phone uses (react-native/bernFeatures.ts
 * and bernOnDevice.ts, compiled to bern-runtime.js), not a mock of it.
 *
 * Build-time placeholders: __RUNTIME__ is replaced with the compiled runtime and
 * __INTENT_BUNDLE__ with the model, so that the result is one file that works from a
 * file:// URL or an iframe with no network access.
 */

const { useState, useRef, useEffect, useMemo, useCallback } = React;

/* Bern's mascot. Inlined as data URIs by scripts/build_web_demo.py so the page keeps working
 * with no network; the PNGs themselves live in web/demo/assets/. */
/** The app logo. Tight face crop — at 40 px the body and ears are just noise. */
const LOGO = __MASCOT_FACE__;

/* ------------------------------------------------------------------ model */

const runtimeReady = (async () => {
  const runtime = window.__BERN_RUNTIME__;
  if (!runtime) throw new Error("The model runtime failed to load.");
  const bundle = __INTENT_BUNDLE__;
  return { bern: await runtime.BernOnDevice.load(bundle, bundle, bundle), runtime };
})();

/* ------------------------------------------------------------------ markdown */

/** Render the tiny markdown the stored replies use: **bold**, `code`, and numbered lines. */
function RichText({ text }) {
  const blocks = useMemo(() => splitBlocks(text), [text]);
  return (
    <div className="space-y-2">
      {blocks.map((block, index) =>
        block.type === "code" ? (
          <pre key={index} className="overflow-x-auto rounded-lg bg-slate-950/90 px-3 py-2 text-[12px] leading-relaxed text-slate-200 ring-1 ring-white/10">
            <code>{block.text}</code>
          </pre>
        ) : (
          <p key={index} className="whitespace-pre-wrap break-words">{inline(block.text)}</p>
        )
      )}
    </div>
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

/** **bold** and `code` inline. Deliberately not a full parser — the replies only use these. */
function inline(text) {
  const parts = [];
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let cursor = 0;
  let match;
  let key = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) parts.push(text.slice(cursor, match.index));
    const token = match[0];
    if (token.startsWith("**")) {
      parts.push(<strong key={key++} className="font-semibold text-slate-900 dark:text-white">{token.slice(2, -2)}</strong>);
    } else {
      parts.push(<code key={key++} className="rounded bg-slate-200/80 px-1 py-0.5 font-mono text-[12px] text-slate-800 dark:bg-white/10 dark:text-slate-100">{token.slice(1, -1)}</code>);
    }
    cursor = pattern.lastIndex;
  }
  if (cursor < text.length) parts.push(text.slice(cursor));
  return parts;
}

/* ------------------------------------------------------------------ pieces */

const EXAMPLES = [
  { label: "Price question", text: "magkano ang bigas ngayon?", kind: "in" },
  { label: "Handwriting help", text: "paano mag transcribe ng handwriting?", kind: "in" },
  { label: "Small talk", text: "kumusta ka naman?", kind: "in" },
  { label: "A joke", text: "tell me a joke about machine learning", kind: "in" },
  { label: "Who are you", text: "who are you and what can you do?", kind: "in" },
  { label: "Outside its training", text: "what's the weather in Manila tomorrow?", kind: "out" },
  { label: "Cursive", text: "read my cursive handwriting", kind: "out" },
  { label: "Not its job", text: "write me a poem about the sea", kind: "out" },
];

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
  train_help: "How to teach it",
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
  humor: "Humor",
  user_name: "Your name",
  time_date: "Date and time",
  out_of_scope: "Outside training",
};

function App() {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [showScores, setShowScores] = useState(true);
  const [theme, setTheme] = useState("dark");
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState(null);
  const [latency, setLatency] = useState(null);
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
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  const ask = useCallback(
    (text) => {
      const message = (text ?? draft).trim();
      if (!message) return;
      if (status !== "ready" || !runtime.current) {
        setError("The model is still loading. Give it a moment.");
        return;
      }
      setError(null);
      setDraft("");

      const started = performance.now();
      const result = runtime.current.bern.classify(message);
      const reply = runtime.current.bern.reply(message);
      const elapsed = performance.now() - started;

      setLatency(elapsed);
      setMessages((previous) => [
        ...previous,
        { role: "user", text: message, id: `u${previous.length}` },
        {
          role: "bern",
          id: `b${previous.length}`,
          text: reply.text ?? "I don't have a stored answer for that one.",
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
    setLatency(null);
    if (inputRef.current) inputRef.current.focus();
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 antialiased dark:bg-slate-950 dark:text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-4 sm:px-6">
        <header className="flex flex-col gap-3 border-b border-slate-200 py-4 sm:flex-row sm:items-center sm:justify-between dark:border-white/10">
          <div>
            <h1 className="text-base font-semibold leading-tight">Bern</h1>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Intent model, running in this tab
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <StatusPill status={status} latency={latency} />
            <button
              type="button"
              onClick={() => setShowScores((value) => !value)}
              className="whitespace-nowrap rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:bg-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 dark:border-white/10 dark:bg-white/5 dark:text-slate-300 dark:hover:bg-white/10"
              aria-pressed={showScores}
            >
              {showScores ? "Hide scores" : "Show scores"}
            </button>
            <button
              type="button"
              onClick={() => setTheme((value) => (value === "dark" ? "light" : "dark"))}
              className="whitespace-nowrap rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:bg-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 dark:border-white/10 dark:bg-white/5 dark:text-slate-300 dark:hover:bg-white/10"
              aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            >
              {theme === "dark" ? "Light" : "Dark"}
            </button>
            <button
              type="button"
              onClick={reset}
              className="whitespace-nowrap rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:bg-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 dark:border-white/10 dark:bg-white/5 dark:text-slate-300 dark:hover:bg-white/10"
            >
              Start over
            </button>
            <img
              src={LOGO}
              alt="Bern"
              className="h-10 w-10 shrink-0 rounded-xl object-cover object-top ring-1 ring-black/5 shadow-lg shadow-indigo-500/25 dark:ring-white/10"
            />
          </div>
        </header>

        <main
          ref={scrollRef}
          className="flex-1 space-y-6 overflow-y-auto py-6"
          aria-live="polite"
          aria-label="Conversation"
        >
          {messages.length === 0 ? (
            <EmptyState onPick={ask} disabled={status !== "ready"} />
          ) : (
            messages.map((message) =>
              message.role === "user" ? (
                <UserBubble key={message.id} text={message.text} />
              ) : (
                <BernBubble key={message.id} message={message} showScores={showScores} />
              )
            )
          )}

          {error ? (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-500/30 dark:bg-rose-500/10 dark:text-rose-200" role="alert">
              {error}
            </div>
          ) : null}
        </main>

        <footer className="sticky bottom-0 -mx-4 border-t border-slate-200 bg-slate-50/90 px-4 py-4 backdrop-blur sm:-mx-6 sm:px-6 dark:border-white/10 dark:bg-slate-950/90">
          <div className="flex items-end gap-2">
            <textarea
              ref={inputRef}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder={status === "ready" ? "Ask Bern something…" : "Loading the model…"}
              disabled={status !== "ready"}
              className="max-h-40 min-h-[46px] flex-1 resize-none rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm placeholder:text-slate-400 focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/30 disabled:opacity-60 dark:border-white/10 dark:bg-white/5 dark:placeholder:text-slate-500"
              aria-label="Your message"
            />
            <button
              type="button"
              onClick={() => ask()}
              disabled={status !== "ready" || !draft.trim()}
              className="h-[46px] shrink-0 rounded-xl bg-indigo-600 px-5 text-sm font-semibold text-white shadow-lg shadow-indigo-600/25 transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Send
            </button>
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-slate-500 dark:text-slate-500">
            Real intent classifier, 24 classes, evaluated on this device — nothing leaves the page.
            <span className="hidden sm:inline"> Replies are the ones stored with the model, so anything needing a live lookup says so.</span>
          </p>
        </footer>
      </div>
    </div>
  );
}

function StatusPill({ status, latency }) {
  if (status === "loading") {
    return (
      <span className="inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full bg-amber-100 px-2.5 py-1 text-[11px] font-medium text-amber-800 dark:bg-amber-400/15 dark:text-amber-200">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500" />
        Loading model
      </span>
    );
  }
  if (status === "failed") {
    return (
      <span className="inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full bg-rose-100 px-2.5 py-1 text-[11px] font-medium text-rose-800 dark:bg-rose-400/15 dark:text-rose-200">
        <span className="h-1.5 w-1.5 rounded-full bg-rose-500" />
        Model unavailable
      </span>
    );
  }
  return (
    <span className="inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full bg-emerald-100 px-2.5 py-1 text-[11px] font-medium text-emerald-800 dark:bg-emerald-400/15 dark:text-emerald-200">
      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
      Ready
      {latency !== null ? <span className="hidden font-mono sm:inline">· {latency.toFixed(1)} ms</span> : null}
    </span>
  );
}

function EmptyState({ onPick, disabled }) {
  return (
    <div className="mx-auto max-w-2xl py-4">
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-white/10 dark:bg-white/5">
        <h2 className="text-lg font-semibold">Try it yourself</h2>
        <p className="mt-1.5 text-sm leading-relaxed text-slate-600 dark:text-slate-300">
          Type anything. Bern runs a 24-class intent model over it and answers with what it was
          taught. The confidence score is the model's, not a decoration — watch it drop when you
          ask something outside its training.
        </p>
        <div className="mt-5 grid gap-2 sm:grid-cols-2">
          {EXAMPLES.map((example) => (
            <button
              key={example.text}
              type="button"
              onClick={() => onPick(example.text)}
              disabled={disabled}
              className={`group rounded-xl border px-3.5 py-3 text-left transition disabled:cursor-not-allowed disabled:opacity-50 ${
                example.kind === "out"
                  ? "border-rose-200 bg-rose-50/60 hover:border-rose-300 hover:bg-rose-50 dark:border-rose-400/25 dark:bg-rose-400/5 dark:hover:bg-rose-400/10"
                  : "border-slate-200 bg-slate-50 hover:border-indigo-300 hover:bg-indigo-50 dark:border-white/10 dark:bg-white/5 dark:hover:border-indigo-400/40 dark:hover:bg-indigo-400/10"
              }`}
            >
              <span className={`text-[11px] font-semibold uppercase tracking-wide ${example.kind === "out" ? "text-rose-600 dark:text-rose-300" : "text-indigo-600 dark:text-indigo-300"}`}>
                {example.label}
              </span>
              <span className="mt-0.5 block text-sm text-slate-700 dark:text-slate-200">
                &ldquo;{example.text}&rdquo;
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function UserBubble({ text }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-indigo-600 px-4 py-2.5 text-sm text-white shadow-md shadow-indigo-600/20 sm:max-w-[75%]">
        {text}
      </div>
    </div>
  );
}

function BernBubble({ message, showScores }) {
  const low = message.confidence < 0.35;
  return (
    <div className="flex">
      <div className="min-w-0 flex-1 space-y-2">
        <div className="rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-4 py-3 text-sm leading-relaxed shadow-sm dark:border-white/10 dark:bg-white/5">
          <RichText text={message.text} />
        </div>

        {message.intent === "out_of_scope" ? (
          <p className="text-[12px] leading-relaxed text-slate-500 dark:text-slate-400">
            The reply mentions a <span className="font-medium text-slate-700 dark:text-slate-300">Teach Bern This</span>
            {" "}button. That lives in the desktop app, where teaching writes a new phrase into the
            model; this page only loads a trained one. Rewriting weights in a browser tab would
            give you a model you could not keep.
          </p>
        ) : null}

        {message.needsServer ? (
          <p className="flex items-start gap-1.5 rounded-lg bg-amber-50 px-3 py-2 text-[12px] leading-relaxed text-amber-800 dark:bg-amber-400/10 dark:text-amber-200">
            <span className="mt-px shrink-0 whitespace-nowrap font-semibold">Heads up:</span>
            <span>
              That line is only Bern's opening — the real result comes from the server, from data
              this page does not have.
            </span>
          </p>
        ) : null}

        {showScores ? <ScorePanel message={message} low={low} /> : null}
      </div>
    </div>
  );
}

function ScorePanel({ message, low }) {
  const top = message.alternatives.length ? message.alternatives[0] : null;
  return (
    <div className="rounded-xl border border-slate-200 bg-white/70 px-3 py-2.5 text-[12px] dark:border-white/10 dark:bg-white/5">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="rounded-md bg-slate-900 px-1.5 py-0.5 font-mono text-[11px] text-slate-100 dark:bg-white/10">
          {message.intent}
        </span>
        <span className="text-slate-500 dark:text-slate-400">
          {INTENT_LABELS[message.intent] ?? "intent"}
        </span>
        <span
          className={`ml-auto rounded-md px-1.5 py-0.5 font-mono text-[11px] font-semibold ${
            low
              ? "bg-rose-100 text-rose-700 dark:bg-rose-400/15 dark:text-rose-200"
              : "bg-emerald-100 text-emerald-700 dark:bg-emerald-400/15 dark:text-emerald-200"
          }`}
        >
          {(message.confidence * 100).toFixed(1)}%
        </span>
        <span className="font-mono text-[11px] text-slate-400 dark:text-slate-500">
          {message.elapsed.toFixed(1)} ms
        </span>
      </div>

      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-white/10">
        <div
          className={`h-full rounded-full ${low ? "bg-rose-500" : "bg-emerald-500"}`}
          style={{ width: `${Math.max(2, message.confidence * 100)}%` }}
        />
      </div>

      {top ? (
        <p className="mt-1.5 text-slate-500 dark:text-slate-400">
          Next best: <span className="font-mono">{top.label}</span> at{" "}
          {(top.confidence * 100).toFixed(1)}%
          {low ? " — a low score here means Bern is guessing, and it says so rather than inventing an answer." : ""}
        </p>
      ) : null}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
