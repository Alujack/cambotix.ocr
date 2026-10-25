/**
 * ONNX CTC text-line recognizer (the Python `onnx_ctc` backend).
 *
 * Model contract: input `[batch, 1, height, width]` float32 grayscale in
 * [0, 1] with a fixed height and dynamic width; first output: logits
 * `[time, batch, classes]` with the CTC blank at class 0 and token `i` at
 * class `i + tokenOffset`. Classes that map to no token are dropped.
 *
 * Each crop is trimmed to its ink first, since line recognizers are trained
 * on tight crops. Decoded lines are checked against Khmer orthography: stray
 * marks at a line's edges are dropped (capping the confidence), and any
 * other impossible sequence sets the confidence to 0.
 */
import { ocrError } from "../errors.js";
import { otsuBinarize } from "../image/filters.js";
import { crop, meanStd, roundHalfEven, toGray, type Raster } from "../image/raster.js";
import { resize } from "../image/resize.js";
import { Recognizer, type RecognizedText, type RegionHint } from "./base.js";

/** The part of an ONNX Runtime session the recognizer needs. */
export interface OnnxSession {
  /** Run the model on one input tensor and return its first output. */
  run(input: Float32Array, dims: readonly number[]): Promise<{ data: Float32Array; dims: readonly number[] }>;
  release(): Promise<void>;
}

// Blank column runs at least this many line-heights wide count as word gaps.
const SPACE_GAP_RATIO = 0.22;
const CROP_MARGIN_Y = 4;
const CROP_MARGIN_X = 6;
// Edge specks with less than this share of a crop's ink can be trimmed off.
const SPECK_INK_RATIO = 0.05;

const COENG = "្";
const isKhmerConsonant = (cp: number) => cp >= 0x1780 && cp < 0x17a3;
const isKhmerBase = (cp: number) => cp >= 0x1780 && cp < 0x17b4;
// Upper bound on the confidence of a read that had edge noise removed, low
// enough that the engine flags it for review.
const CLEANED_CONFIDENCE = 0.5;
// Dependent vowels and signs that must follow a Khmer letter. U+17C8 (ៈ) is
// left out: recognizers read a printed ":" as ៈ, often after a space.
const isKhmerDependent = (cp: number) =>
  (cp >= 0x17b6 && cp < 0x17c8) || (cp >= 0x17c9 && cp < 0x17d2) || cp === 0x17d3 || cp === 0x17dd;

export interface OnnxCtcOptions {
  modelFile: string;
  tokens: string;
  tokenOffset?: number;
  inputHeight?: number;
  restoreSpaces?: boolean;
  confidenceCalibration?: ReadonlyArray<readonly number[]> | null;
}

export class OnnxCtcRecognizer extends Recognizer {
  readonly name = "onnx-ctc";
  readonly tokens: string[];
  readonly tokenOffset: number;
  readonly inputHeight: number;
  readonly restoreSpaces: boolean;
  readonly calibration: [number[], number[]] | null;
  private readonly modelFile: string;
  private session: OnnxSession | null;

  constructor(session: OnnxSession, options: OnnxCtcOptions) {
    super();
    this.calibration = options.confidenceCalibration == null
      ? null : checkCalibration(options.confidenceCalibration);
    if (!options.tokens) throw ocrError("MODEL_NOT_AVAILABLE", "The recognizer vocabulary is empty.");
    this.session = session;
    this.modelFile = options.modelFile;
    this.tokens = Array.from(options.tokens);
    this.tokenOffset = options.tokenOffset ?? 3;
    this.inputHeight = options.inputHeight ?? 32;
    this.restoreSpaces = (options.restoreSpaces ?? true) && !this.tokens.includes(" ");
  }

  override describe(): Record<string, string> {
    return { backend: this.name, model: this.modelFile, vocabulary_size: String(this.tokens.length) };
  }

  async recognizeLine(image: Raster, _hint?: RegionHint): Promise<RecognizedText> {
    const gray = trimToInk(toGray(image));
    if (gray === null) return { text: "", confidence: null };
    const width = Math.max(1, roundHalfEven(gray.width * this.inputHeight / gray.height));
    const line = resize(gray, width, this.inputHeight, "area");
    const tensor = new Float32Array(line.data.length);
    for (let i = 0; i < tensor.length; i++) tensor[i] = line.data[i]! / 255;
    if (this.session === null) throw ocrError("MODEL_NOT_AVAILABLE", "The engine has been closed.");
    let output: { data: Float32Array; dims: readonly number[] };
    try {
      output = await this.session.run(tensor, [1, 1, this.inputHeight, width]);
    } catch (cause) {
      throw ocrError("INFERENCE_FAILED", "Text recognition failed.", undefined, cause);
    }
    const [timeSteps, batch, classes] = output.dims as [number, number, number];
    let { chars, steps, confidences } = this.decode(output.data, timeSteps, batch, classes);
    if (chars.length === 0) return { text: "", confidence: null };
    let scores: Array<number | null> = [...confidences];
    if (this.restoreSpaces) {
      // The vocabulary has no space, so every space here was inserted.
      chars = insertWordSpaces(chars, steps, line, timeSteps);
      let next = 0;
      scores = chars.map((ch) => (ch === " " ? null : confidences[next++]!));
    }
    const cleaned = dropEdgeNoise(chars, scores);
    const kept = cleaned.scores.filter((s): s is number => s !== null);
    if (kept.length === 0) return { text: "", confidence: null };
    const text = cleaned.chars.join("");
    let confidence = kept.reduce((a, b) => a + b, 0) / kept.length;
    if (this.calibration !== null) confidence = interp(confidence, ...this.calibration);
    if (cleaned.chars.length < chars.length) confidence = Math.min(confidence, CLEANED_CONFIDENCE);
    if (!khmerStructureValid(text)) confidence = 0;
    return { text, confidence };
  }

  override async close(): Promise<void> {
    const session = this.session;
    this.session = null;
    await session?.release();
  }

  /** Greedy CTC decoding: collapse repeats, drop blanks and unmapped classes. */
  private decode(logits: Float32Array, timeSteps: number, batch: number, classes: number) {
    const chars: string[] = [];
    const steps: number[] = [];
    const confidences: number[] = [];
    let previous = -1;
    for (let t = 0; t < timeSteps; t++) {
      const base = t * batch * classes; // batch item 0
      let max = -Infinity;
      for (let c = 0; c < classes; c++) max = Math.max(max, logits[base + c]!);
      let total = 0;
      let best = 0;
      let bestValue = -Infinity;
      for (let c = 0; c < classes; c++) {
        const v = logits[base + c]! - max;
        total += Math.exp(v);
        if (v > bestValue) {
          bestValue = v;
          best = c;
        }
      }
      const token = best - this.tokenOffset;
      if (best !== previous && best !== 0 && token >= 0 && token < this.tokens.length) {
        chars.push(this.tokens[token]!);
        steps.push(t);
        confidences.push(Math.exp(bestValue) / total);
      }
      previous = best;
    }
    return { chars, steps, confidences };
  }
}

function checkCalibration(points: ReadonlyArray<readonly number[]>): [number[], number[]] {
  const isPairs = Array.isArray(points) && points.every((p) => Array.isArray(p));
  const raw = isPairs ? points.map((p) => Number(p[0])) : [];
  const calibrated = isPairs ? points.map((p) => Number(p[1])) : [];
  const inUnit = (v: number) => Number.isFinite(v) && v >= 0 && v <= 1;
  const valid = isPairs && points.length >= 2 && points.every((p) => p.length === 2)
    && [...raw, ...calibrated].every(inUnit)
    && raw.every((v, i) => i === 0 || raw[i - 1]! < v)
    && calibrated.every((v, i) => i === 0 || calibrated[i - 1]! <= v);
  if (!valid) {
    throw ocrError("MODEL_NOT_AVAILABLE",
      "confidence_calibration must be at least two [raw, calibrated] pairs in [0, 1], "
      + "with raw strictly increasing and calibrated non-decreasing.");
  }
  return [raw, calibrated];
}

/** NumPy `interp(x, xp, fp)`. */
export function interp(x: number, xp: readonly number[], fp: readonly number[]): number {
  if (x <= xp[0]!) return fp[0]!;
  const last = xp.length - 1;
  if (x >= xp[last]!) return fp[last]!;
  let k = 1;
  while (xp[k]! < x) k++;
  const x0 = xp[k - 1]!;
  const x1 = xp[k]!;
  if (x === x1) return fp[k]!;
  return fp[k - 1]! + (fp[k]! - fp[k - 1]!) * (x - x0) / (x1 - x0);
}

const LETTER_OR_NUMBER = /[\p{L}\p{N}]/u;

function isNoiseWord(chars: readonly string[]): boolean {
  // Punctuation-only words such as "។", ")" or "-" are real text; a Khmer
  // mark with no letter to attach to is not.
  return !chars.some((ch) => LETTER_OR_NUMBER.test(ch))
    && chars.some((ch) => ch === COENG || isKhmerDependent(ch.codePointAt(0)!));
}

/**
 * Drop leading and trailing words made of a Khmer mark and punctuation with
 * no letter or digit, such as `.្` read from a neighbouring line at the crop
 * edge. A line that is a single word is left alone.
 */
export function dropEdgeNoise(chars: string[], scores: Array<number | null>):
  { chars: string[]; scores: Array<number | null> } {
  let start = 0;
  let end = chars.length;
  for (;;) {
    while (start < end && chars[start] === " ") start++;
    while (end > start && chars[end - 1] === " ") end--;
    const inner = chars.slice(start, end);
    if (!inner.includes(" ")) break;
    const firstSpace = inner.indexOf(" ");
    const lastSpace = inner.lastIndexOf(" ");
    if (isNoiseWord(inner.slice(lastSpace + 1))) {
      end = start + lastSpace;
    } else if (isNoiseWord(inner.slice(0, firstSpace))) {
      start += firstSpace;
    } else {
      break;
    }
  }
  return { chars: chars.slice(start, end), scores: scores.slice(start, end) };
}

/**
 * False when the text has a Khmer sequence no correctly read word can
 * contain: a coeng not between a base letter and a consonant, or a dependent
 * vowel or sign with no Khmer character before it.
 */
export function khmerStructureValid(text: string): boolean {
  const cps = Array.from(text).map((ch) => ch.codePointAt(0)!);
  for (let i = 0; i < cps.length; i++) {
    const cp = cps[i]!;
    const prev = i > 0 ? cps[i - 1]! : null;
    if (cp === 0x17d2) {
      const next = i + 1 < cps.length ? cps[i + 1]! : null;
      if (prev === null || !isKhmerBase(prev) || next === null || !isKhmerConsonant(next)) return false;
    } else if (isKhmerDependent(cp) && (prev === null || !(prev >= 0x1780 && prev <= 0x17ff))) {
      return false;
    }
  }
  return true;
}

/**
 * Crop a field region to its text, assuming dark text on a light background.
 *
 * Specks at the left or right edge, such as a vowel mark from a neighbouring
 * line, are left out when a blank gap of at least one line height separates
 * them from the text: the recognizer would otherwise read the gap as letters.
 *
 * Returns null when the region contains no ink at all.
 */
export function trimToInk(gray: Raster): Raster | null {
  if (gray.data.length === 0 || meanStd(gray.data).std < 8) return null;
  const ink = otsuBinarize(gray, 1, true);
  const { width: w, height: h } = gray;
  const minRowInk = Math.max(1, Math.floor(w / 200));
  const rowInk = new Int32Array(h);
  const columnInk = new Int32Array(w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = ink.data[y * w + x]!;
      rowInk[y]! += v;
      columnInk[x]! += v;
    }
  }
  const rows = indicesWhere(rowInk, (v) => v >= minRowInk);
  let cols = indicesWhere(columnInk, (v) => v >= 1);
  if (rows.length === 0 || cols.length === 0) return null;
  cols = dropEdgeSpecks(cols, columnInk, rows[rows.length - 1]! - rows[0]! + 1);
  const left = cols[0]!;
  const right = cols[cols.length - 1]!;
  const keptRows: number[] = [];
  for (let y = 0; y < h; y++) {
    let sum = 0;
    for (let x = left; x <= right; x++) sum += ink.data[y * w + x]!;
    if (sum >= minRowInk) keptRows.push(y);
  }
  if (keptRows.length === 0) return null;
  const y0 = Math.max(0, keptRows[0]! - CROP_MARGIN_Y);
  const y1 = Math.min(h, keptRows[keptRows.length - 1]! + CROP_MARGIN_Y + 1);
  const x0 = Math.max(0, left - CROP_MARGIN_X);
  const x1 = Math.min(w, right + CROP_MARGIN_X + 1);
  return crop(gray, x0, y0, x1 - x0, y1 - y0);
}

function indicesWhere(values: Int32Array, test: (v: number) => boolean): number[] {
  const out: number[] = [];
  values.forEach((v, i) => { if (test(v)) out.push(i); });
  return out;
}

/**
 * Drop edge clusters of ink columns that hold under 5% of the ink and are
 * separated from the rest by a blank gap of at least `lineHeight`.
 */
function dropEdgeSpecks(cols: number[], columnInk: Int32Array, lineHeight: number): number[] {
  const minGap = Math.max(2, lineHeight);
  const clusters: number[][] = [[cols[0]!]];
  for (let i = 1; i < cols.length; i++) {
    if (cols[i]! - cols[i - 1]! - 1 >= minGap) clusters.push([]);
    clusters[clusters.length - 1]!.push(cols[i]!);
  }
  const mass = (cluster: number[]) => cluster.reduce((sum, c) => sum + columnInk[c]!, 0);
  const total = clusters.reduce((sum, c) => sum + mass(c), 0);
  while (clusters.length > 1) {
    if (mass(clusters[clusters.length - 1]!) < SPECK_INK_RATIO * total) clusters.pop();
    else if (mass(clusters[0]!) < SPECK_INK_RATIO * total) clusters.shift();
    else break;
  }
  return clusters.flat();
}

/**
 * Insert a space between two decoded characters when a wide blank column
 * run in the line image lies between their positions.
 */
export function insertWordSpaces(chars: string[], steps: number[], line: Raster,
                                 timeSteps: number): string[] {
  const ink = otsuBinarize(line, 1, true);
  const { width: w, height: h } = line;
  const empty: boolean[] = [];
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = 0; y < h; y++) sum += ink.data[y * w + x]!;
    empty.push(sum === 0);
  }
  const minGap = Math.max(2, roundHalfEven(SPACE_GAP_RATIO * h));
  const gapCenters: number[] = [];
  let runStart: number | null = null;
  [...empty, false].forEach((isEmpty, x) => {
    if (isEmpty && runStart === null) {
      runStart = x;
    } else if (!isEmpty && runStart !== null) {
      if (x - runStart >= minGap && runStart > 0 && x < empty.length) gapCenters.push((runStart + x) / 2);
      runStart = null;
    }
  });
  if (gapCenters.length === 0) return chars;
  const columnPerStep = w / Math.max(1, timeSteps);
  const positions = steps.map((step) => (step + 0.5) * columnPerStep);
  const spaced = [chars[0]!];
  for (let i = 1; i < chars.length; i++) {
    if (gapCenters.some((gap) => positions[i - 1]! < gap && gap < positions[i]!)) spaced.push(" ");
    spaced.push(chars[i]!);
  }
  return spaced;
}
