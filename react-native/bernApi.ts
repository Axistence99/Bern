/**
 * Bern API client — drop this file into a React Native / Expo app and you have every
 * endpoint typed and wrapped.
 *
 * No dependencies: it uses `fetch` and `FormData`, both of which Expo provides. Works in
 * Expo Go, bare React Native, and the web.
 *
 * Typical use:
 *
 * ```ts
 * import { BernClient } from "./bernApi";
 *
 * const bern = new BernClient("http://192.168.1.20:8000");
 *
 * const reply = await bern.chat("magkano ang bigas");
 * console.log(reply.reply);
 *
 * const pack = await bern.studyFile(uri, "lecture.pdf");
 * console.log(pack.quiz.length);
 * ```
 *
 * **Finding the base URL.** `localhost` on a phone means the phone itself, not your
 * laptop. Use your machine's LAN address (`http://192.168.x.x:8000`), or the tunnel URL
 * your dev tool prints. Everyone on the same Wi-Fi is required when testing on a device.
 */

// --------------------------------------------------------------------------------------
// Response types — these mirror what the handlers in server.py actually return.
// --------------------------------------------------------------------------------------

export interface ChatResponse {
  reply: string;
  intent: string;
  confidence: number;
  needs_training: boolean;
  top_predictions: Array<{ intent: string; probability: number }>;
  top_signals: string[];
  inference_ms: number;
  ai_name: string;
  model_version: string;
  trained_at: string | null;
  guessed_intent?: string;
  /** `"local"` or `"llm"` — which engine actually produced this reply. */
  engine?: string;
  /** Items the reply was about, e.g. `["Regular Milled Rice"]`. */
  entities?: string[];
  /**
   * Set when a follow-up was rewritten using earlier turns, e.g. `"magkano ang Regular
   * Milled Rice"`. Show it if you want to make the interpretation visible.
   */
  resolved_as?: string | null;
  /** What the server remembers for this session. */
  conversation?: { turns: number; subject: string | null; discussed: string[] };
}

/** Response from `POST /api/tutor`. Requires a language model on the server. */
export interface TutorResponse {
  answer: string;
  engine: string;
  grounded: boolean;
  session_id: string;
  turns: number;
}

export interface HandwritingLine {
  text: string;
  confidence: number | null;
  glyphs: number;
}

export interface HandwritingResponse {
  text: string;
  lines: HandwritingLine[];
  backend: string;
  confidence: number | null;
  glyph_count: number;
  inference_ms: number;
  unsupported_symbols: string[];
  ai_name: string;
  /**
   * What the photo pipeline corrected before reading. Useful to surface: "straightened the
   * page" explains a surprising result, and `rules_removed: 0` on ruled paper tells you the
   * detection missed and the reading is probably bad.
   */
  repairs?: { perspective: boolean; rules_removed: number; columns: number };
}

export interface ExtractedDocument {
  text: string;
  source_type: "text" | "pdf" | "docx" | "image";
  engine: string;
  characters: number;
  lines: number;
  words: number;
}

export interface Flashcard {
  front: string;
  back: string;
  source: string;
}

export interface QuizQuestion {
  type: "multiple_choice" | "fill_in";
  question: string;
  options: string[];
  answer: string;
  answer_index: number;
  explanation: string;
}

export interface TrueFalseItem {
  type: "true_false";
  statement: string;
  answer: boolean;
  explanation: string;
}

export interface StudyPack {
  reviewer: {
    title: string;
    sections: Array<{ heading: string; points: string[] }>;
    glossary: Array<{ term: string; definition: string }>;
    key_terms: string[];
  };
  flashcards: Flashcard[];
  quiz: QuizQuestion[];
  mock_exam: {
    title: string;
    sections: Array<{ name: string; instructions: string; items: unknown[] }>;
    answer_key: Array<{ number: number; answer: string; explanation: string }>;
  };
  source_stats: Record<string, string | number>;
  ai_name: string;
}

export interface MoneyResult {
  total_php: number;
  items: Array<{ label: string; denomination: number; confidence: number; bbox: number[] }>;
  buying_power?: unknown;
}

export interface BernError extends Error {
  status?: number;
  payload?: unknown;
}

// --------------------------------------------------------------------------------------
// Client
// --------------------------------------------------------------------------------------

export class BernClient {
  readonly baseUrl: string;
  private timeoutMs: number;

  /**
   * @param baseUrl Root of the Bern server, e.g. `http://192.168.1.20:8000`.
   *                No trailing slash needed.
   * @param timeoutMs Request ceiling. Recognition on a large photo can take a second or
   *                  two; 30 s is generous without hanging forever on a dead host.
   */
  constructor(baseUrl: string, timeoutMs = 30_000) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.timeoutMs = timeoutMs;
  }

  // -- plumbing ------------------------------------------------------------------------

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        signal: controller.signal,
        headers: {
          // Never set Content-Type for FormData: the runtime must add the multipart
          // boundary itself, and overriding it produces an unparseable request.
          ...(init.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
          ...(init.headers ?? {}),
        },
      });

      const text = await response.text();
      const payload = text ? safeJsonParse(text) : null;

      if (!response.ok) {
        const message =
          (payload && typeof payload === "object" && "error" in payload
            ? String((payload as { error: unknown }).error)
            : null) ?? `HTTP ${response.status}`;
        const error: BernError = new Error(message);
        error.status = response.status;
        error.payload = payload;
        throw error;
      }
      return payload as T;
    } finally {
      clearTimeout(timer);
    }
  }

  private async postJson<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>(path, { method: "POST", body: JSON.stringify(body) });
  }

  /**
   * Attach a file chosen by the user to a `FormData` body.
   *
   * React Native does not accept a bare `{ uri }` the way the web accepts a `Blob`, so the
   * shape below is deliberate — it is the one that works on both.
   */
  private fileForm(uri: string, filename: string, field = "file"): FormData {
    const form = new FormData();
    const name = filename || uri.split("/").pop() || "upload";
    const match = /\.(\w+)$/.exec(name);
    const type = match ? mimeFor(match[1].toLowerCase()) : "application/octet-stream";
    // @ts-expect-error React Native accepts {uri, name, type} where the web expects Blob.
    form.append(field, { uri, name, type });
    return form;
  }

  // -- conversation --------------------------------------------------------------------

  /** Health probe. Use this to confirm the phone can reach the laptop before debugging anything else. */
  health(): Promise<{ status: string; ai_name: string; num_intents: number; language_model?: boolean }> {
    return this.request("/api/health");
  }

  /** Free-form chat. Returns the reply plus the routing metadata behind it. */
  /**
   * Send one message, continuing the conversation identified by `sessionId`.
   *
   * Follow-ups work: the server remembers what earlier turns were about, so "what about
   * the other one?" resolves against the previous answer. Keep the same `sessionId` for a
   * conversation and use a new one to start over.
   *
   * @param engine `"auto"` (default) uses the server's language model when one is
   *               configured; `"local"` never does. Both behave identically without a key.
   */
  chat(message: string, sessionId = "default", engine: "auto" | "local" = "auto"): Promise<ChatResponse> {
    return this.postJson("/api/chat", { message, session_id: sessionId, engine });
  }

  /** Raw classifier output without the response text. Useful for debugging misrouting. */
  /**
   * Ask a follow-up question about a set of notes. Requires `BERN_LLM_API_KEY` on the
   * server; without one the call rejects with HTTP 503 and a readable message.
   *
   * Distinct from {@link chat}: this carries the notes as grounding and the model answers
   * in its own words. The server keeps the thread per `sessionId`, so pass the same one to
   * continue.
   */
  tutor(question: string, options: { sessionId?: string; notes?: string } = {}): Promise<TutorResponse> {
    return this.postJson("/api/tutor", {
      question,
      session_id: options.sessionId ?? "default",
      notes: options.notes ?? "",
    });
  }

  /** Forget a tutoring session on the server. */
  clearTutorSession(sessionId = "default"): Promise<{ cleared: string }> {
    return this.request(`/api/tutor?session_id=${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    });
  }

  predict(message: string): Promise<ChatResponse> {
    return this.postJson("/api/predict", { message });
  }

  // -- handwriting ---------------------------------------------------------------------

  /**
   * Transcribe a photo of handwriting.
   *
   * @param uri Local file URI from ImagePicker or Camera.
   * @param options.backend `auto` | `glyph` | `trocr` | `tesseract`.
   * @param options.caseMode `auto` | `upper` | `lower`.
   */
  async handwriting(uri: string, options: { backend?: string; caseMode?: string } = {}): Promise<HandwritingResponse> {
    const form = this.fileForm(uri, uri.split("/").pop() ?? "note.jpg", "image");
    if (options.backend) form.append("backend", options.backend);
    if (options.caseMode) form.append("case_mode", options.caseMode);
    return this.request("/api/handwriting", { method: "POST", body: form });
  }

  // -- documents and study material ----------------------------------------------------

  /** Extract plain text from a PDF, DOCX, image or text file. */
  extractFile(uri: string, filename: string): Promise<ExtractedDocument> {
    return this.request("/api/extract", { method: "POST", body: this.fileForm(uri, filename) });
  }

  /** Extract text that the app already holds in memory (e.g. pasted notes). */
  extractText(text: string): Promise<ExtractedDocument> {
    return this.postJson("/api/extract", { text });
  }

  /**
   * Build a reviewer, quiz, mock exam and flashcards from an uploaded document.
   *
   * Handwritten photos work too — they are recognised first, then turned into questions.
   */
  studyFile(
    uri: string,
    filename: string,
    options: { flashcards?: number; quizItems?: number; trueFalseItems?: number; seed?: number } = {},
  ): Promise<StudyPack> {
    const form = this.fileForm(uri, filename);
    appendStudyOptions(form, options);
    return this.request("/api/study", { method: "POST", body: form });
  }

  /** Same as {@link studyFile} but from text already in memory. */
  studyText(
    text: string,
    options: { flashcards?: number; quizItems?: number; trueFalseItems?: number; seed?: number } = {},
  ): Promise<StudyPack> {
    return this.postJson("/api/study", {
      text,
      flashcards: options.flashcards ?? 20,
      quiz_items: options.quizItems ?? 10,
      true_false_items: options.trueFalseItems ?? 8,
      seed: options.seed ?? 7,
    });
  }

  // -- prices --------------------------------------------------------------------------

  /** Curated PH goods catalog. `q` filters by name, `category` by group. */
  prices(q = "", category = ""): Promise<unknown> {
    return this.request(`/api/ph-prices?q=${encodeURIComponent(q)}&category=${encodeURIComponent(category)}`);
  }

  /** Search the 1,836-item PH retail SKU catalog. */
  skus(q = "", category = "", limit = 20): Promise<{ count: number; total_skus: number; items: unknown[] }> {
    return this.request(
      `/api/ph-prices/skus?q=${encodeURIComponent(q)}&category=${encodeURIComponent(category)}&limit=${limit}`,
    );
  }

  /** Suggest a shopping basket that fits a budget, in pesos. */
  budget(budgetPhp: number): Promise<unknown> {
    return this.postJson("/api/ph-prices/budget", { budget_php: budgetPhp });
  }

  // -- nutrition -----------------------------------------------------------------------

  /** Search 19,594 foods by name. `source` may be `philfct`, `usda`, `off` or `curated`. */
  nutrition(q: string, options: { limit?: number; source?: string; type?: string; category?: string } = {}) {
    const params = new URLSearchParams({ q, limit: String(options.limit ?? 10) });
    if (options.source) params.set("source", options.source);
    if (options.type) params.set("type", options.type);
    if (options.category) params.set("category", options.category);
    return this.request(`/api/nutrition?${params.toString()}`);
  }

  /** One food by id, optionally scaled to a weight in grams. */
  nutritionItem(id: string, grams?: number) {
    const suffix = grams === undefined ? "" : `&grams=${grams}`;
    return this.request(`/api/nutrition/item?id=${encodeURIComponent(id)}${suffix}`);
  }

  /** Totals for a meal written in Taglish, e.g. `"2 pc chickenjoy and 1 cup rice"`. */
  meal(text: string): Promise<unknown> {
    return this.postJson("/api/nutrition/meal", { text });
  }

  nutritionSummary(): Promise<unknown> {
    return this.request("/api/nutrition/summary");
  }

  // -- money ---------------------------------------------------------------------------

  /** Count Philippine bills and coins in a photo. */
  countMoney(uri: string): Promise<MoneyResult> {
    const form = this.fileForm(uri, uri.split("/").pop() ?? "money.jpg", "image");
    return this.request("/api/count-money", { method: "POST", body: form });
  }
}

// --------------------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------------------

function safeJsonParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function appendStudyOptions(
  form: FormData,
  options: { flashcards?: number; quizItems?: number; trueFalseItems?: number; seed?: number },
): void {
  if (options.flashcards !== undefined) form.append("flashcards", String(options.flashcards));
  if (options.quizItems !== undefined) form.append("quiz_items", String(options.quizItems));
  if (options.trueFalseItems !== undefined) form.append("true_false_items", String(options.trueFalseItems));
  if (options.seed !== undefined) form.append("seed", String(options.seed));
}

function mimeFor(extension: string): string {
  const map: Record<string, string> = {
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    png: "image/png",
    webp: "image/webp",
    bmp: "image/bmp",
    tif: "image/tiff",
    tiff: "image/tiff",
    pdf: "application/pdf",
    docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    txt: "text/plain",
    md: "text/markdown",
  };
  return map[extension] ?? "application/octet-stream";
}
