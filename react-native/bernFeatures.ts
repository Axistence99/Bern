/**
 * Bern's feature extractors, in plain TypeScript.
 *
 * This file exists so the phone can describe a glyph or a message exactly the way the trained
 * models expect, without OpenCV, scikit-learn or a Python runtime. It is a line-for-line port
 * of `bern/portable_features.py` and `bern/portable_text.py`, and the two sides are checked
 * against each other on identical inputs — see `scripts/check_ondevice_parity.py`.
 *
 * Correctness here is not stylistic. A descriptor that is off by one cell, or a tokenizer that
 * forgets to lowercase, produces numbers in the right shape that score the wrong character.
 * Every non-obvious choice below is carried over from Python with the reason it exists.
 */

export const GLYPH_SIZE = 28;
export const CELL_SIZE = 7;
export const BLOCK_CELLS = 2;
export const BINS = 9;
export const L2_HYS_CLIP = 0.2;
export const FEATURE_DIM = 341;

/** Ink-bright 8-bit grey of any size, row-major. */
export type Gray = Uint8Array | Float32Array | number[];

function at(gray: Gray, index: number): number {
  return gray[index] as number;
}

/**
 * Bilinear resample with OpenCV's pixel-centre mapping.
 *
 * OpenCV samples at `(x + 0.5) * in/out - 0.5`, not at `x * in/out`. Half a pixel decides
 * whether a one-pixel stroke survives a downscale or is averaged away completely.
 */
function bilinearResize(image: ArrayLike<number>, inW: number, inH: number, outW: number, outH: number): Float64Array {
  const out = new Float64Array(outW * outH);
  if (inW === outW && inH === outH) {
    for (let i = 0; i < out.length; i += 1) out[i] = image[i];
    return out;
  }

  const scaleX = inW / outW;
  const scaleY = inH / outH;
  for (let y = 0; y < outH; y += 1) {
    let srcY = (y + 0.5) * scaleY - 0.5;
    if (srcY < 0) srcY = 0;
    if (srcY > inH - 1) srcY = inH - 1;
    const y0 = Math.floor(srcY);
    const y1 = Math.min(y0 + 1, inH - 1);
    const wy = srcY - y0;

    for (let x = 0; x < outW; x += 1) {
      let srcX = (x + 0.5) * scaleX - 0.5;
      if (srcX < 0) srcX = 0;
      if (srcX > inW - 1) srcX = inW - 1;
      const x0 = Math.floor(srcX);
      const x1 = Math.min(x0 + 1, inW - 1);
      const wx = srcX - x0;

      const top = image[y0 * inW + x0] * (1 - wx) + image[y0 * inW + x1] * wx;
      const bottom = image[y1 * inW + x0] * (1 - wx) + image[y1 * inW + x1] * wx;
      out[y * outW + x] = top * (1 - wy) + bottom * wy;
    }
  }
  return out;
}

/**
 * Otsu's threshold: the cut that best separates two populations of grey values.
 *
 * Between-class variance is maximised over every possible cut. That is the standard
 * derivation, kept because the alternative — a fixed cut at 128 — fails on any photo whose
 * exposure is off, which is most photos taken indoors.
 */
export function otsuThreshold(gray: Gray, length: number): number {
  const histogram = new Float64Array(256);
  for (let i = 0; i < length; i += 1) {
    const value = at(gray, i);
    histogram[value < 0 ? 0 : value > 255 ? 255 : Math.round(value)] += 1;
  }

  const total = length;
  let sum = 0;
  for (let level = 0; level < 256; level += 1) sum += level * histogram[level];

  let sumBelow = 0;
  let weightBelow = 0;
  let bestVariance = -1;
  let bestLevel = 0;
  for (let level = 0; level < 256; level += 1) {
    weightBelow += histogram[level];
    if (weightBelow === 0) continue;
    const weightAbove = total - weightBelow;
    if (weightAbove === 0) break;

    sumBelow += level * histogram[level];
    const meanBelow = sumBelow / weightBelow;
    const meanAbove = (sum - sumBelow) / weightAbove;
    const variance = weightBelow * weightAbove * (meanBelow - meanAbove) * (meanBelow - meanAbove);
    if (variance > bestVariance) {
      bestVariance = variance;
      bestLevel = level;
    }
  }
  return bestLevel;
}

/**
 * Turn a greyscale photograph into an ink-bright mask: white strokes on black.
 *
 * The models were trained on ink-bright 28x28 glyphs and cannot tell a bright page from a
 * bright stroke — `normaliseGlyph` simply scales whatever is above 8. Feed it a camera photo of
 * black ink on white paper and it fits the *paper* to the box with the character as a hole in
 * the middle, then returns a confident, wrong answer. Measured: a photographed "T" read as "8"
 * at 1.00 confidence before this step existed.
 *
 * Otsu picks a global cut, which is right for a page under even light. Two corrections are
 * carried over from the server's `ink_mask` because they were needed in practice, not as
 * defensive padding: a page that comes out mostly "ink" was photographed inverted, and one
 * that comes out almost entirely empty is washed out and needs a local threshold instead.
 */
export function inkMask(gray: Gray, width: number, height: number): Uint8Array {
  const length = width * height;
  // cv2.GaussianBlur with a 3x3 kernel and sigma 0 derives sigma = 0.8 from the size, which
  // gives these three taps. A box filter would be simpler and would threshold slightly
  // differently; matching the kernel keeps the two implementations comparable.
  const centre = 0.5221;
  const side = 0.2389;

  const horizontal = new Float32Array(length);
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      const left = at(gray, row + (x > 0 ? x - 1 : 0));
      const middle = at(gray, row + x);
      const right = at(gray, row + (x < width - 1 ? x + 1 : width - 1));
      horizontal[row + x] = side * left + centre * middle + side * right;
    }
  }

  const blurred = new Uint8Array(length);
  for (let y = 0; y < height; y += 1) {
    const up = (y > 0 ? y - 1 : 0) * width;
    const here = y * width;
    const down = (y < height - 1 ? y + 1 : height - 1) * width;
    for (let x = 0; x < width; x += 1) {
      const value = side * horizontal[up + x] + centre * horizontal[here + x] + side * horizontal[down + x];
      blurred[here + x] = value < 0 ? 0 : value > 255 ? 255 : Math.round(value);
    }
  }

  const cut = otsuThreshold(blurred, length);
  const mask = new Uint8Array(length);
  let ink = 0;
  for (let i = 0; i < length; i += 1) {
    if (blurred[i] <= cut) {
      mask[i] = 255;
      ink += 1;
    }
  }

  const coverage = ink / length;
  if (coverage > 0.5) {
    // The page was inverted: white text on a dark background. Flip it.
    for (let i = 0; i < length; i += 1) mask[i] = mask[i] === 0 ? 255 : 0;
    ink = length - ink;
  } else if (coverage < 0.002) {
    // Faint pencil or a bad exposure, where a global cut wipes the page out. Use a local
    // mean over a 31-pixel window with a 15-level bias, as the server does.
    const radius = 15;
    const integral = new Float64Array((width + 1) * (height + 1));
    for (let y = 0; y < height; y += 1) {
      let rowSum = 0;
      for (let x = 0; x < width; x += 1) {
        rowSum += blurred[y * width + x];
        integral[(y + 1) * (width + 1) + x + 1] = integral[y * (width + 1) + x + 1] + rowSum;
      }
    }
    const windowMean = (x: number, y: number): number => {
      const x0 = Math.max(0, x - radius);
      const y0 = Math.max(0, y - radius);
      const x1 = Math.min(width - 1, x + radius);
      const y1 = Math.min(height - 1, y + radius);
      const total =
        integral[(y1 + 1) * (width + 1) + x1 + 1]
        - integral[y0 * (width + 1) + x1 + 1]
        - integral[(y1 + 1) * (width + 1) + x0]
        + integral[y0 * (width + 1) + x0];
      return total / ((x1 - x0 + 1) * (y1 - y0 + 1));
    };
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        mask[y * width + x] = blurred[y * width + x] < windowMean(x, y) - 15 ? 255 : 0;
      }
    }
  }
  return mask;
}

/**
 * Fit one glyph crop into a 28x28 square the way the server does.
 *
 * Tighten to the ink, scale so the longest side fills 86% of the box, centre it, resample
 * bilinearly. The 86% leaves the margin the EMNIST glyphs were trained with; filling the box
 * completely makes every character look like every other character.
 */
export function normaliseGlyph(gray: Gray, width: number, height: number, size: number = GLYPH_SIZE): Uint8Array {
  let minRow = height;
  let maxRow = -1;
  let minCol = width;
  let maxCol = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (at(gray, y * width + x) > 8) {
        if (y < minRow) minRow = y;
        if (y > maxRow) maxRow = y;
        if (x < minCol) minCol = x;
        if (x > maxCol) maxCol = x;
      }
    }
  }
  if (maxRow < 0) return new Uint8Array(size * size);

  const tightW = maxCol - minCol + 1;
  const tightH = maxRow - minRow + 1;
  const tight = new Float64Array(tightW * tightH);
  for (let y = 0; y < tightH; y += 1) {
    for (let x = 0; x < tightW; x += 1) {
      tight[y * tightW + x] = at(gray, (minRow + y) * width + minCol + x);
    }
  }

  const scale = (size * 0.86) / Math.max(tightH, tightW);
  const newW = Math.max(1, Math.round(tightW * scale));
  const newH = Math.max(1, Math.round(tightH * scale));
  const resized = bilinearResize(tight, tightW, tightH, newW, newH);

  const canvas = new Uint8Array(size * size);
  const offsetX = Math.floor((size - newW) / 2);
  const offsetY = Math.floor((size - newH) / 2);
  for (let y = 0; y < newH; y += 1) {
    for (let x = 0; x < newW; x += 1) {
      const value = resized[y * newW + x];
      canvas[(offsetY + y) * size + offsetX + x] = value < 0 ? 0 : value > 255 ? 255 : Math.floor(value);
    }
  }
  return canvas;
}

/**
 * Central-difference gradients with replicated edges, after sqrt gamma correction.
 *
 * Gamma first: OpenCV's HOGDescriptor does it by default, and without it the descriptor is
 * dominated by whichever strokes are darkest, so how hard the pen pressed outweighs which
 * way it moved.
 *
 * Edges replicate rather than zero-pad. A zero border invents a hard edge around every glyph,
 * which lands as spurious vertical and horizontal energy in the outer cells — exactly where
 * thin letters like `l` and `1` keep their only ink.
 */
export function gradients(glyph: Float32Array): [Float32Array, Float32Array] {
  const size = GLYPH_SIZE;
  const corrected = new Float32Array(size * size);
  for (let i = 0; i < corrected.length; i += 1) {
    const value = glyph[i] < 0 ? 0 : glyph[i] > 255 ? 255 : glyph[i];
    // numpy's `astype(np.uint8)` truncates; rounding here would shift every gradient.
    corrected[i] = Math.floor(Math.sqrt(value));
  }

  const gx = new Float32Array(size * size);
  const gy = new Float32Array(size * size);
  const clamp = (v: number) => (v < 0 ? 0 : v > size - 1 ? size - 1 : v);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const left = corrected[y * size + clamp(x - 1)];
      const right = corrected[y * size + clamp(x + 1)];
      const up = corrected[clamp(y - 1) * size + x];
      const down = corrected[clamp(y + 1) * size + x];
      gx[y * size + x] = right - left;
      gy[y * size + x] = down - up;
    }
  }
  return [gx, gy];
}

/**
 * Orientation histograms: hard angle bins, interpolated in space.
 *
 * The two are deliberately opposite ways round. Smoothing the angle blurs the difference
 * between a vertical stroke and a slanted one, which is what separates `1` from `l`; treating
 * each pixel as belonging to exactly one cell makes the descriptor jump when a stroke sits one
 * pixel across a boundary. Getting these backwards cost 4.9 points of accuracy when measured.
 */
export function cellHistograms(gx: Float32Array, gy: Float32Array): Float64Array {
  const size = GLYPH_SIZE;
  const cellsPerAxis = size / CELL_SIZE;
  const binsPerCell = BINS;
  const histograms = new Float64Array(cellsPerAxis * cellsPerAxis * binsPerCell);
  const degreesPerBin = 180 / binsPerCell;

  for (let y = 0; y < size; y += 1) {
    // Continuous cell coordinates: the pixel centre expressed in cell units.
    const cellY = (y + 0.5) / CELL_SIZE - 0.5;
    const y0 = Math.floor(cellY);
    const wy = cellY - y0;

    for (let x = 0; x < size; x += 1) {
      const gxv = gx[y * size + x];
      const gyv = gy[y * size + x];
      const magnitude = Math.sqrt(gxv * gxv + gyv * gyv);
      if (magnitude === 0) continue;

      let angle = (Math.atan2(gyv, gxv) * 180) / Math.PI;
      angle %= 180;
      if (angle < 0) angle += 180;
      let bin = Math.floor(angle / degreesPerBin);
      if (bin > binsPerCell - 1) bin = binsPerCell - 1;

      const cellX = (x + 0.5) / CELL_SIZE - 0.5;
      const x0 = Math.floor(cellX);
      const wx = cellX - x0;

      for (let dy = 0; dy < 2; dy += 1) {
        const weightY = dy === 0 ? 1 - wy : wy;
        let ty = y0 + dy;
        if (ty < 0) ty = 0;
        else if (ty > cellsPerAxis - 1) ty = cellsPerAxis - 1;

        for (let dx = 0; dx < 2; dx += 1) {
          const weightX = dx === 0 ? 1 - wx : wx;
          let tx = x0 + dx;
          if (tx < 0) tx = 0;
          else if (tx > cellsPerAxis - 1) tx = cellsPerAxis - 1;

          histograms[(ty * cellsPerAxis + tx) * binsPerCell + bin] += magnitude * weightY * weightX;
        }
      }
    }
  }
  return histograms;
}

/**
 * Overlapping 2x2-cell blocks, each L2-Hys normalised.
 *
 * Normalising per block is what makes the descriptor tolerate lighting and pen weight: the
 * relative mix of edge directions inside a block survives a change in overall contrast.
 */
export function blockNormalise(histograms: ArrayLike<number>): Float32Array {
  const cellsPerAxis = GLYPH_SIZE / CELL_SIZE;
  const blocksPerAxis = cellsPerAxis - BLOCK_CELLS + 1;
  const blockSize = BLOCK_CELLS * BLOCK_CELLS * BINS;
  const output = new Float32Array(blocksPerAxis * blocksPerAxis * blockSize);
  let cursor = 0;

  for (let blockRow = 0; blockRow < blocksPerAxis; blockRow += 1) {
    for (let blockCol = 0; blockCol < blocksPerAxis; blockCol += 1) {
      const block = new Float32Array(blockSize);
      let index = 0;
      for (let dy = 0; dy < BLOCK_CELLS; dy += 1) {
        for (let dx = 0; dx < BLOCK_CELLS; dx += 1) {
          const base = ((blockRow + dy) * cellsPerAxis + blockCol + dx) * BINS;
          for (let b = 0; b < BINS; b += 1) block[index++] = histograms[base + b];
        }
      }

      let sum = 0;
      for (let i = 0; i < blockSize; i += 1) sum += block[i] * block[i];
      let norm = Math.sqrt(sum + 1e-6);
      for (let i = 0; i < blockSize; i += 1) block[i] /= norm;

      // L2-Hys: clip the outliers, then renormalise.
      sum = 0;
      for (let i = 0; i < blockSize; i += 1) {
        if (block[i] > L2_HYS_CLIP) block[i] = L2_HYS_CLIP;
        sum += block[i] * block[i];
      }
      norm = Math.sqrt(sum + 1e-6);
      for (let i = 0; i < blockSize; i += 1) output[cursor++] = block[i] / norm;
    }
  }
  return output;
}

/** Describe one 28x28 glyph as 341 floats: 324 HOG, 16 zoning, 1 ink ratio. */
export function glyphFeatures(glyph: Gray): Float32Array {
  if (glyph.length !== GLYPH_SIZE * GLYPH_SIZE) {
    throw new Error(`expected a ${GLYPH_SIZE}x${GLYPH_SIZE} glyph, got ${glyph.length} values`);
  }
  const normalised = new Float32Array(glyph.length);
  for (let i = 0; i < glyph.length; i += 1) {
    const value = at(glyph, i);
    normalised[i] = value < 0 ? 0 : value > 255 ? 255 : value;
  }

  const [gx, gy] = gradients(normalised);
  const hog = blockNormalise(cellHistograms(gx, gy));

  const output = new Float32Array(FEATURE_DIM);
  output.set(hog, 0);

  // Zoning: mean ink per cell of a 4x4 grid.
  let cursor = hog.length;
  let total = 0;
  for (let i = 0; i < normalised.length; i += 1) total += normalised[i];
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      let sum = 0;
      for (let y = 0; y < CELL_SIZE; y += 1) {
        for (let x = 0; x < CELL_SIZE; x += 1) {
          sum += normalised[(row * CELL_SIZE + y) * GLYPH_SIZE + col * CELL_SIZE + x];
        }
      }
      output[cursor++] = sum / (CELL_SIZE * CELL_SIZE) / 255;
    }
  }
  output[cursor] = total / normalised.length / 255;
  return output;
}

/**
 * Word runs plus standalone arithmetic operators.
 *
 * Python's `\w` is Unicode-aware but excludes combining marks; `[\p{L}\p{N}_]` is the same
 * set for the languages Bern handles.
 */
const WORD_PATTERN = /[\p{L}\p{N}_]+|[+\-*/=()]/gu;

export function wordTokens(text: string): string[] {
  return text.match(WORD_PATTERN) ?? [];
}

/**
 * Character n-grams inside word boundaries, each word padded with one space each side.
 *
 * A word shorter than `n` yields that n-gram once rather than once per position — scikit-learn
 * guards this with `if offset == 0: break`, and dropping the guard changes the counts.
 */
export function charWbTokens(text: string, minimum: number, maximum: number): string[] {
  const tokens: string[] = [];
  for (const raw of text.split(/\s+/)) {
    if (raw.length === 0) continue;
    const padded = ` ${raw} `;
    const length = padded.length;
    for (let n = minimum; n <= maximum; n += 1) {
      let offset = 0;
      tokens.push(padded.slice(0, n));
      while (offset + n < length) {
        offset += 1;
        tokens.push(padded.slice(offset, offset + n));
      }
      if (offset === 0) break;
    }
  }
  return tokens;
}

/** One vectoriser's vocabulary, IDF weights and analyser settings. */
export interface VectorizerSpec {
  terms: string[];
  index: Map<string, number>;
  idf: Float32Array;
  analyzer: 'word' | 'char_wb';
  ngramRange: [number, number];
  sublinearTf: boolean;
  weight: number;
}

/**
 * Word tokens with the joined n-grams scikit-learn generates.
 *
 * With `ngram_range` of `(1, 2)` the vectoriser emits every unigram and every adjacent pair
 * joined by a space, so `bakit mahal` is itself a feature. Omitting the pairs leaves the row
 * with the right length and the wrong contents.
 */
export function wordNgrams(text: string, minimum: number, maximum: number): string[] {
  const tokens = wordTokens(text);
  if (maximum === 1) return tokens;

  const generated = minimum === 1 ? [...tokens] : [];
  const count = tokens.length;
  for (let n = Math.max(minimum, 2); n <= Math.min(maximum, count); n += 1) {
    for (let start = 0; start < count - n + 1; start += 1) {
      generated.push(tokens.slice(start, start + n).join(' '));
    }
  }
  return generated;
}

/** One L2-normalised TF-IDF row. */
export function tfidfRow(spec: VectorizerSpec, text: string): Float32Array {
  // Lowercasing belongs here, not inside either analyser: scikit-learn runs the preprocessor
  // over the whole document first, so the character analyser sees lowercased text too.
  const document = text.toLowerCase();
  const tokens =
    spec.analyzer === 'char_wb'
      ? charWbTokens(document, spec.ngramRange[0], spec.ngramRange[1])
      : wordNgrams(document, spec.ngramRange[0], spec.ngramRange[1]);

  const counts = new Map<number, number>();
  for (const token of tokens) {
    const position = spec.index.get(token);
    if (position !== undefined) counts.set(position, (counts.get(position) ?? 0) + 1);
  }

  const row = new Float32Array(spec.terms.length);
  if (counts.size === 0) return row;

  let sum = 0;
  const weighted: Array<[number, number]> = [];
  for (const [position, count] of counts) {
    const value = (spec.sublinearTf ? 1 + Math.log(count) : count) * spec.idf[position];
    weighted.push([position, value]);
    sum += value * value;
  }

  const norm = Math.sqrt(sum);
  if (norm > 0) {
    for (const [position, value] of weighted) row[position] = value / norm;
  }
  return row;
}

/**
 * The FeatureUnion output: each vectoriser's row scaled by its own weight, concatenated.
 *
 * `layers` carries the order the vectorisers were listed in when the model was fitted,
 * because that is the column order of the weight matrix. Sorting by name instead shifts every
 * column by one block and produces confident nonsense.
 */
export function unionRow(layers: VectorizerSpec[], text: string): Float32Array {
  const rows = layers.map((layer) => tfidfRow(layer, text));
  const total = rows.reduce((sum, row) => sum + row.length, 0);
  const output = new Float32Array(total);
  let cursor = 0;
  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i];
    const weight = layers[i].weight;
    for (let j = 0; j < row.length; j += 1) output[cursor + j] = row[j] * weight;
    cursor += row.length;
  }
  return output;
}

/** Softmax over `coef @ x + intercept`, matching `LogisticRegression.predict_proba`. */
export function softmaxScores(
  features: ArrayLike<number>,
  coefficients: Float32Array,
  intercept: Float32Array,
): Float32Array {
  const classCount = intercept.length;
  const featureCount = features.length;
  const scores = new Float32Array(classCount);
  let highest = -Infinity;

  for (let c = 0; c < classCount; c += 1) {
    let score = intercept[c];
    const base = c * featureCount;
    for (let f = 0; f < featureCount; f += 1) {
      const value = features[f];
      if (value !== 0) score += coefficients[base + f] * value;
    }
    scores[c] = score;
    if (score > highest) highest = score;
  }

  let total = 0;
  for (let c = 0; c < classCount; c += 1) {
    const exponentiated = Math.exp(scores[c] - highest);
    scores[c] = exponentiated;
    total += exponentiated;
  }
  for (let c = 0; c < classCount; c += 1) scores[c] /= total;
  return scores;
}

/* ------------------------------------------------------------------ arithmetic */

/**
 * Arithmetic the on-device runtime can do by itself.
 *
 * Ported from `bern/engine.py` (`extract_math_expression`, `safe_eval_math_ast`,
 * `is_pure_math`) so the phone and the browser give the same answer as the server. The
 * intent model only decides that a message *is* arithmetic; it cannot produce the number,
 * which is why the stored answer for `calculator` is an offer rather than a result.
 *
 * Expressions are evaluated by a small recursive-descent parser, never by `eval`: this code
 * runs on input from strangers.
 */

const MATH_FILLERS = new Set([
  'how', 'much', 'is', 'what', 'whats', "what's", 'calculate', 'compute', 'solve', 'equals',
  'equal', 'the', 'answer', 'of', 'to', 'plus', 'minus', 'times', 'divided', 'by', 'please',
  'pls', 'x', 'ano', 'ang', 'ba', 'result', 'value', 'sum', 'total', 'and', 'multiplied',
  'over', 'squared', 'cubed', 'product', 'difference', 'between', 'add', 'subtract',
  'multiply', 'divide', 'from', 'pakicompute', 'pakicalculate', 'pakisolve', 'pakisagot', 'ilan', 'kuwentahin', 'less',
]);

const NUMBER = '(-?\\d+(?:\\.\\d+)?)';

/** `15% of 400`, `divided by`, `sum of a and b` … Returns a plain expression or null. */
export function extractMathExpression(message: string): string | null {
  const text = message.trim().toLowerCase();

  const percent = text.match(
    new RegExp(`${NUMBER}\\s*(?:%|percent)\\s*of\\s*${NUMBER}`),
  );
  if (percent) {
    return `(${percent[1]} / 100) * ${percent[2]}`;
  }

  const divide = text.match(new RegExp(`divide\\s+${NUMBER}\\s+by\\s+${NUMBER}`));
  if (divide) return `${divide[1]} / ${divide[2]}`;

  const multiply = text.match(new RegExp(`multiply\\s+${NUMBER}\\s+(?:by|and)\\s+${NUMBER}`));
  if (multiply) return `${multiply[1]} * ${multiply[2]}`;

  const add = text.match(new RegExp(`add\\s+${NUMBER}\\s+(?:and|to)\\s+${NUMBER}`));
  if (add) return `${add[1]} + ${add[2]}`;

  const subtract = text.match(new RegExp(`subtract\\s+${NUMBER}\\s+from\\s+${NUMBER}`));
  if (subtract) return `${subtract[2]} - ${subtract[1]}`;

  const sum = text.match(new RegExp(`(?:sum|total)\\s+of\\s+${NUMBER}\\s+and\\s+${NUMBER}`));
  if (sum) return `${sum[1]} + ${sum[2]}`;

  const product = text.match(new RegExp(`product\\s+of\\s+${NUMBER}\\s+and\\s+${NUMBER}`));
  if (product) return `${product[1]} * ${product[2]}`;

  const difference = text.match(
    new RegExp(`difference\\s+(?:of|between)\\s+${NUMBER}\\s+and\\s+${NUMBER}`),
  );
  if (difference) {
    const a = Number(difference[1]);
    const b = Number(difference[2]);
    return `${Math.max(a, b)} - ${Math.min(a, b)}`;
  }

  let cleaned = text.replace(/(\d+(?:\.\d+)?)\s*squared/g, '($1**2)');
  cleaned = cleaned.replace(/(\d+(?:\.\d+)?)\s*cubed/g, '($1**3)');

  // Operators only count between numbers, so "6 x 9" converts but "box 6" does not.
  const left = '(?<=[\\d)])\\s*';
  const right = '\\s*(?=[\\d(])';
  cleaned = cleaned.replace(new RegExp(`${left}x${right}`, 'g'), '*');
  cleaned = cleaned.replace(new RegExp(`${left}(?:times|multiplied by)${right}`, 'g'), '*');
  cleaned = cleaned.replace(new RegExp(`${left}(?:divided by|over)${right}`, 'g'), '/');
  cleaned = cleaned.replace(new RegExp(`${left}plus${right}`, 'g'), '+');
  cleaned = cleaned.replace(new RegExp(`${left}(?:minus|less)${right}`, 'g'), '-');

  const candidates = cleaned.match(/[\d.()\s+\-*/%^]+/g) || [];
  const ordered = [...candidates].sort((a, b) => b.length - a.length);
  for (const candidate of ordered) {
    const expression = candidate.trim().replace(/\^/g, '**');
    const hasOperator = ['+', '-', '*', '/', '%'].some((op) => expression.includes(op));
    if (hasOperator && /\d/.test(expression) && evaluateMath(expression) !== null) {
      return expression;
    }
  }
  return null;
}

/** True when arithmetic is all that surrounds the numbers: "how much is 12 x 12". */
export function isPureMath(message: string): boolean {
  const words = message.toLowerCase().match(/[a-z']+/g) || [];
  return words.length > 0 && words.every((word) => MATH_FILLERS.has(word)) && /\d/.test(message);
}

/**
 * Evaluate `** * / + -` with parentheses and a leading unary sign.
 *
 * Written as a parser rather than by translating to JavaScript: an expression arriving from a
 * chat box must never reach `eval`, and `**` in JavaScript is right-associative in the same way
 * Python's is, so the shared `15% of 400` and `2 ** 10` forms agree.
 */
export function evaluateMath(expression: string): number | null {
  let cursor = 0;
  const source = expression.replace(/\s+/g, '');

  const parseExpression = (): number => {
    let value = parseTerm();
    while (cursor < source.length && (source[cursor] === '+' || source[cursor] === '-')) {
      const operator = source[cursor];
      cursor += 1;
      const right = parseTerm();
      value = operator === '+' ? value + right : value - right;
    }
    return value;
  };

  const parseTerm = (): number => {
    let value = parseUnary();
    while (cursor < source.length && (source[cursor] === '*' || source[cursor] === '/' || source[cursor] === '%')) {
      const operator = source[cursor];
      cursor += 1;
      const right = parseUnary();
      if (operator === '*') value *= right;
      else if (operator === '/') value /= right;
      else value %= right;
    }
    return value;
  };

  const parseUnary = (): number => {
    if (source[cursor] === '+') {
      cursor += 1;
      return parseUnary();
    }
    if (source[cursor] === '-') {
      cursor += 1;
      return -parseUnary();
    }
    return parsePower();
  };

  const parsePower = (): number => {
    const base = parseAtom();
    if (source.startsWith('**', cursor)) {
      cursor += 2;
      const exponent = parseUnary();
      if (Math.abs(exponent) > 100) throw new Error('exponent too large');
      return Math.pow(base, exponent);
    }
    return base;
  };

  const parseAtom = (): number => {
    if (source[cursor] === '(') {
      cursor += 1;
      const value = parseExpression();
      if (source[cursor] !== ')') throw new Error('unbalanced parentheses');
      cursor += 1;
      return value;
    }
    const match = /^\d+(?:\.\d+)?/.exec(source.slice(cursor));
    if (!match) throw new Error('expected a number');
    cursor += match[0].length;
    return Number(match[0]);
  };

  try {
    const value = parseExpression();
    if (cursor !== source.length) return null;
    return Number.isFinite(value) ? value : null;
  } catch {
    return null;
  }
}

/** `20`, not `20.0`; six decimals for anything that does not divide cleanly. */
export function formatMathValue(value: number): string {
  if (Math.abs(value - Math.round(value)) < 1e-9) return String(Math.round(value));
  return String(Math.round(value * 1e6) / 1e6);
}
