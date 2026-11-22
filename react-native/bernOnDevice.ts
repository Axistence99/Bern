/**
 * Bern running on the phone.
 *
 * Loads the three exported bundles (`models/ondevice/*.json`) and evaluates them in
 * JavaScript. No Python, no network, no OpenCV — just the weights and the arithmetic that was
 * fitted against them.
 *
 * What this can do on its own: recognise a message's intent, answer arithmetic by actually
 * evaluating it, return Bern's stored answer for everything else, read a single handwritten
 * character, read a handwritten symbol. What it cannot do: split a photographed page into
 * characters, look up prices, correct spellings with the dictionary, or answer anything outside
 * its 25 intents. Those need the server, and `bernApi.ts` still speaks to it.
 *
 * ```
 * const leaf = await BernOnDevice.load(intentJson, glyphJson, symbolJson);
 * leaf.classify('magkano ang bigas?').intent   // 'price_query'
 * leaf.readGlyph(gray28).char                  // 'a'
 * ```
 */

import {
  evaluateMath,
  extractMathExpression,
  formatMathValue,
  glyphFeatures,
  inkMask,
  isPureMath,
  normaliseGlyph,
  softmaxScores,
  unionRow,
  type VectorizerSpec,
} from './bernFeatures';

/**
 * Answer an arithmetic message, or null if it holds no expression.
 *
 * Two shapes of answer, because the question has two shapes. `10 + 10` gets the result and
 * nothing else — the reader typed the expression and can check it. `how much is 12 x 12` gets
 * the expression back beside the number, because the words had to be read to find it and a bare
 * `144` would not show what was understood.
 */
export function computeArithmetic(message: string): string | null {
  const expression = extractMathExpression(message);
  if (!expression) return null;

  const value = evaluateMath(expression);
  if (value === null) return null;

  const answer = formatMathValue(value);
  const shown = expression.trim();

  if (isPureMath(message)) {
    return `**${shown}** = **${answer}**`;
  }
  return `**${answer}**`;
}

/** The exported bundle, as JSON. Bundlers can `require` it directly. */
export interface IntentBundle {
  format: string;
  classes: string[];
  intercept: string;
  coefficients: string;
  coef_shape: [number, number];
  feature_count: number;
  layers: Array<{
    name: string;
    weight: number;
    analyzer: 'word' | 'char_wb';
    ngram_range: [number, number];
    sublinear_tf: boolean;
    size: number;
    offset: number;
    terms: string;
    idf: string;
  }>;
  intent_responses: Record<string, string[]>;
  /** Intents whose stored reply announces an answer only the server holds. */
  server_backed: string[];
  gating: Record<string, number>;
  ai_name: string;
}

export interface GlyphBundle {
  format: string;
  classes: string[];
  intercept: string;
  coefficients: string;
  coef_shape: [number, number];
  descriptor: Record<string, number | string | boolean>;
  metrics: Record<string, unknown>;
}

export interface ClassScore {
  label: string;
  confidence: number;
}

export interface IntentResult {
  intent: string;
  confidence: number;
  alternatives: ClassScore[];
}

export interface GlyphResult {
  char: string;
  confidence: number;
  candidates: ClassScore[];
}

const BASE64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

/**
 * Decode base64 to bytes.
 *
 * Written out rather than using `atob` or `Buffer`: Hermes has no `Buffer`, `atob` is absent
 * on older React Native versions, and this file must not add a dependency.
 */
function decodeBase64(text: string): Uint8Array {
  const clean = text.replace(/[^A-Za-z0-9+/]/g, '');
  const bytes = new Uint8Array(Math.floor((clean.length * 3) / 4));
  let accumulator = 0;
  let bits = 0;
  let written = 0;
  for (let i = 0; i < clean.length; i += 1) {
    accumulator = (accumulator << 6) | BASE64.indexOf(clean[i]);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      bytes[written++] = (accumulator >> bits) & 0xff;
    }
  }
  return bytes;
}

/**
 * Decode base64 little-endian float32, the layout the exporter writes.
 *
 * Read explicitly as little-endian rather than through a typed-array view: that is correct on
 * every platform instead of only on the ones that happen to be little-endian.
 */
function decodeFloats(text: string): Float32Array {
  const bytes = decodeBase64(text);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = new Float32Array(bytes.byteLength / 4);
  for (let i = 0; i < values.length; i += 1) values[i] = view.getFloat32(i * 4, true);
  return values;
}

function buildSpecs(bundle: IntentBundle): VectorizerSpec[] {
  return bundle.layers.map((layer) => {
    const terms = layer.terms.split('\n');
    const index = new Map<string, number>();
    for (let i = 0; i < terms.length; i += 1) index.set(terms[i], i);
    return {
      terms,
      index,
      idf: decodeFloats(layer.idf),
      analyzer: layer.analyzer,
      ngramRange: layer.ngram_range,
      sublinearTf: layer.sublinear_tf,
      weight: layer.weight,
    };
  });
}

function rankClassScores(scores: Float32Array, classes: string[], limit: number): ClassScore[] {
  const order = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]);
  return order.slice(0, limit).map((index) => ({ label: classes[index], confidence: scores[index] }));
}

/** Case twins pooled onto whichever form wins, as `GlyphBackend` does server-side. */
function foldCaseTwins(scores: Float32Array, classes: string[]): Float32Array {
  const groups = new Map<string, number[]>();
  for (let i = 0; i < classes.length; i += 1) {
    const label = classes[i];
    if (label.length === 1 && /\p{L}/u.test(label)) {
      const key = label.toLowerCase();
      const group = groups.get(key);
      if (group) group.push(i);
      else groups.set(key, [i]);
    }
  }

  const folded = Float32Array.from(scores);
  for (const group of groups.values()) {
    if (group.length < 2) continue;
    let mass = 0;
    let winner = group[0];
    for (const index of group) {
      mass += folded[index];
      if (folded[index] > folded[winner]) winner = index;
    }
    for (const index of group) folded[index] = 0;
    folded[winner] = mass;
  }
  return folded;
}

/**
 * Below this the nearest intent is only a guess, so `reply` refuses instead of answering.
 *
 * Measured, not guessed: on eight prompts with no right answer and seventeen real ones,
 * refusing everything under this score catches five of the eight while every real question
 * still gets through. `bern/engine.py` applies the same floor in `gating_rejects`.
 */
const REFUSE_BELOW = 0.4;

export class BernOnDevice {
  readonly intentBundle: IntentBundle;
  readonly glyphBundle: GlyphBundle;
  readonly symbolBundle: GlyphBundle;
  readonly aiName: string;

  private readonly specs: VectorizerSpec[];
  private readonly intentCoefficients: Float32Array;
  private readonly intentIntercept: Float32Array;
  private readonly glyphCoefficients: Float32Array;
  private readonly glyphIntercept: Float32Array;
  private readonly symbolCoefficients: Float32Array;
  private readonly symbolIntercept: Float32Array;

  private constructor(intent: IntentBundle, glyph: GlyphBundle, symbol: GlyphBundle) {
    this.intentBundle = intent;
    this.glyphBundle = glyph;
    this.symbolBundle = symbol;
    this.aiName = intent.ai_name;
    this.specs = buildSpecs(intent);
    this.intentCoefficients = decodeFloats(intent.coefficients);
    this.intentIntercept = decodeFloats(intent.intercept);
    this.glyphCoefficients = decodeFloats(glyph.coefficients);
    this.glyphIntercept = decodeFloats(glyph.intercept);
    this.symbolCoefficients = decodeFloats(symbol.coefficients);
    this.symbolIntercept = decodeFloats(symbol.intercept);
  }

  /** Build the runtime from the three parsed JSON bundles. */
  static async load(intent: IntentBundle, glyph: GlyphBundle, symbol: GlyphBundle): Promise<BernOnDevice> {
    for (const [name, bundle] of [['intent', intent], ['glyph', glyph], ['symbol', symbol]] as const) {
      if (!bundle || typeof bundle.format !== 'string') {
        throw new Error(`Bern: the ${name} bundle is missing or was not parsed as JSON`);
      }
    }
    return new BernOnDevice(intent, glyph, symbol);
  }

  /** The winning intent for a message, with runners-up for context. */
  classify(message: string, limit = 3): IntentResult {
    const features = unionRow(this.specs, message);
    const scores = softmaxScores(features, this.intentCoefficients, this.intentIntercept);
    const ranked = rankClassScores(scores, this.intentBundle.classes, limit);
    return {
      intent: ranked[0].label,
      confidence: ranked[0].confidence,
      alternatives: ranked.slice(1),
    };
  }

  /**
   * Classify a message and return Bern's stored answer for that intent.
   *
   * The reply is the first stored variant, chosen deterministically rather than at random so
   * a caller can test what it gets.
   *
   * `needsServer` marks the intents whose stored reply is only an *announcement* — "Let me
   * check my 2026 Philippine Price Knowledge Base!" — with the actual answer produced
   * server-side from data the phone does not carry. The intent is recognised correctly; the
   * content is missing. Show the text as a holding message and call the server, or show
   * nothing and say why. Do not present it as the answer.
   */
  /**
   * Bern's answer to a message.
   *
   * Arithmetic is evaluated here rather than looked up: the `calculator` intent ships a stored
   * offer ("Let me run the numbers for you!") because a stored answer cannot contain a number
   * that depends on the question. `needsServer` is cleared when we can produce the result
   * ourselves, so the caller does not chase a server it no longer needs.
   */
  reply(message: string): { intent: string; confidence: number; text: string | null; needsServer: boolean } {
    const result = this.classify(message);
    const stored = this.intentBundle.intent_responses[result.intent];
    let text = stored && stored.length > 0 ? stored[0] : null;
    let needsServer = this.intentBundle.server_backed.includes(result.intent);

    // Below this the answer is a guess dressed as an answer. The server measures the same
    // number in `gating_rejects`, and the two drifting apart would make the browser and the
    // API disagree about the same sentence, so the threshold lives here too.
    if (result.confidence < REFUSE_BELOW || result.intent === 'out_of_scope') {
      const guessed = result.intent;
      const percent = Math.round(result.confidence * 100);
      return {
        intent: 'unknown',
        confidence: result.confidence,
        text: `I do not know this one. My closest guess was \`${guessed}\` at ${percent}%, which `
          + 'is not enough to answer from.\n\nGeneral questions need a language model. This '
          + 'page carries only the trained classifier, so it has none - the same question sent '
          + 'to a server with one configured is answered properly.',
        needsServer: false,
      };
    }

    if (result.intent === 'calculator') {
      const computed = computeArithmetic(message);
      if (computed) {
        text = computed;
        needsServer = false;
      }
    }

    return {
      intent: result.intent,
      confidence: result.confidence,
      text,
      needsServer,
    };
  }

  /** Read one handwritten character from a 28x28 ink-bright glyph. */
  readGlyph(glyph: Uint8Array | Float32Array | number[]): GlyphResult {
    const scores = softmaxScores(glyphFeatures(glyph), this.glyphCoefficients, this.glyphIntercept);
    const folded = foldCaseTwins(scores, this.glyphBundle.classes);
    const ranked = rankClassScores(folded, this.glyphBundle.classes, 5);
    return { char: ranked[0].label, confidence: ranked[0].confidence, candidates: ranked };
  }

  /** Read one handwritten symbol from a 28x28 ink-bright glyph. */
  readSymbol(glyph: Uint8Array | Float32Array | number[]): GlyphResult {
    const scores = softmaxScores(glyphFeatures(glyph), this.symbolCoefficients, this.symbolIntercept);
    const ranked = rankClassScores(scores, this.symbolBundle.classes, 4);
    return { char: ranked[0].label, confidence: ranked[0].confidence, candidates: ranked };
  }

  /**
   * Turn a camera crop into the 28x28 the models expect.
   *
   * Goes through `inkMask` first, so black ink on white paper and white ink on black paper both
   * end up the same way round. Without it `normaliseGlyph` fits the *paper* to the box and the
   * character becomes a hole in the middle of the glyph — which the model happily reads as some
   * other character, at high confidence.
   *
   * `inkBright` skips the mask for callers whose image is already a binary mask, such as one
   * produced by a segmentation step of your own.
   */
  prepare(
    gray: Uint8Array | Float32Array | number[],
    width: number,
    height: number,
    inkBright = false,
  ): Uint8Array {
    const source = inkBright ? gray : inkMask(gray, width, height);
    return normaliseGlyph(source, width, height);
  }
}
