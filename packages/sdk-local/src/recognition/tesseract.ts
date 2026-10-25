/**
 * Tesseract recognizer backend, running tesseract.js (Tesseract compiled to
 * WebAssembly). Language data is read only from the model bundle's
 * `tessdata` folder, never from tesseract.js's default CDN.
 *
 * tesseract.js bundles its own Tesseract build, which can read some lines
 * differently from the `tesseract` binary the Python engine calls.
 */
import { ocrError } from "../errors.js";
import { otsuBinarize } from "../image/filters.js";
import { padConstant, roundHalfEven, toGray, type Raster } from "../image/raster.js";
import { resize } from "../image/resize.js";
import type { TextRegion } from "../schemas.js";
import { DEFAULT_HINT, Recognizer, type RecognizedText, type RegionHint, type Script } from "./base.js";

const SCRIPT_LANGUAGES: Readonly<Record<Script, string>> = {
  khmer: "khm", latin: "eng", digits: "eng", mixed: "khm+eng",
};
const MIN_LINE_HEIGHT = 48;
// Retry ladder for a line crop: (preprocessing, page segmentation mode).
// psm 7 (single line) fails on short, widely spaced text such as numbers
// under a barcode, where psm 8 (single word) succeeds; Otsu binarization
// recovers text printed over patterned backgrounds.
const LINE_ATTEMPTS: ReadonlyArray<["gray" | "otsu", number]> = [["gray", 7], ["otsu", 7]];
const DIGIT_ATTEMPTS: ReadonlyArray<["gray" | "otsu", number]> = [["gray", 7], ["otsu", 7], ["gray", 8]];
// Stop retrying once a read reaches this confidence.
const ACCEPT_CONFIDENCE = 0.85;

/** One recognized word with its line, as Tesseract's `image_to_data` reports it. */
export interface TesseractWord {
  text: string;
  /** 0-100, or negative when Tesseract has no score. */
  confidence: number;
  line: number;
  bbox: { x0: number; y0: number; x1: number; y1: number };
}

/** What the recognizer needs from a tesseract.js worker for one language set. */
export interface TesseractRunner {
  recognize(image: Raster, psm: number, whitelist: string | null): Promise<TesseractWord[]>;
  terminate(): Promise<void>;
}

export type TesseractRunnerFactory = (languages: string) => Promise<TesseractRunner>;

export class TesseractRecognizer extends Recognizer {
  readonly name = "tesseract";
  readonly defaultLanguages: string;
  private readonly createRunner: TesseractRunnerFactory;
  private runners = new Map<string, Promise<TesseractRunner>>();
  private closed = false;

  constructor(createRunner: TesseractRunnerFactory, defaultLanguages = "khm+eng") {
    super();
    this.createRunner = createRunner;
    this.defaultLanguages = defaultLanguages;
  }

  override describe(): Record<string, string> {
    return { backend: this.name, engine: "tesseract.js", languages: this.defaultLanguages };
  }

  async recognizeLine(image: Raster, hint: RegionHint = DEFAULT_HINT): Promise<RecognizedText> {
    const languages = hint.languages ?? SCRIPT_LANGUAGES[hint.script] ?? this.defaultLanguages;
    const attempts = hint.script === "digits" ? DIGIT_ATTEMPTS : LINE_ATTEMPTS;
    let best: RecognizedText = { text: "", confidence: null };
    for (const [mode, psm] of attempts) {
      const words = await this.run(prepareLine(image, mode), languages, psm, hint.charWhitelist ?? null);
      const result = joinWords(words);
      if (result.text && (best.confidence === null || (result.confidence ?? 0) > best.confidence)) {
        best = result;
      }
      if (best.confidence !== null && best.confidence >= ACCEPT_CONFIDENCE) break;
    }
    return best;
  }

  override async recognizePage(image: Raster): Promise<TextRegion[]> {
    const words = await this.run(image, this.defaultLanguages, 3, null);
    const lines = new Map<number, TesseractWord[]>();
    for (const word of words) {
      if (!word.text.trim() || word.confidence < 0) continue;
      const line = lines.get(word.line) ?? [];
      line.push(word);
      lines.set(word.line, line);
    }
    return [...lines.values()].map((line) => {
      const x0 = Math.min(...line.map((w) => w.bbox.x0));
      const y0 = Math.min(...line.map((w) => w.bbox.y0));
      const x1 = Math.max(...line.map((w) => w.bbox.x1));
      const y1 = Math.max(...line.map((w) => w.bbox.y1));
      return {
        text: line.map((w) => w.text.trim()).join(" "),
        confidence: line.reduce((sum, w) => sum + w.confidence, 0) / line.length / 100,
        bbox: { x: x0, y: y0, width: x1 - x0, height: y1 - y0 },
      };
    });
  }

  override async close(): Promise<void> {
    this.closed = true;
    const runners = [...this.runners.values()];
    this.runners.clear();
    await Promise.all(runners.map(async (runner) => {
      try {
        await (await runner).terminate();
      } catch {
        // A worker that failed to start has nothing to release.
      }
    }));
  }

  private async run(image: Raster, languages: string, psm: number,
                    whitelist: string | null): Promise<TesseractWord[]> {
    if (this.closed) throw ocrError("MODEL_NOT_AVAILABLE", "The engine has been closed.");
    let runner = this.runners.get(languages);
    if (runner === undefined) {
      runner = this.createRunner(languages);
      this.runners.set(languages, runner);
      runner.catch(() => this.runners.delete(languages));
    }
    try {
      return await (await runner).recognize(image, psm, whitelist);
    } catch (cause) {
      throw ocrError("INFERENCE_FAILED", "Text recognition failed.", undefined, cause);
    }
  }
}

function joinWords(words: readonly TesseractWord[]): RecognizedText {
  const kept = words.filter((w) => w.text.trim() && w.confidence >= 0);
  if (kept.length === 0) return { text: "", confidence: null };
  return {
    text: kept.map((w) => w.text.trim()).join(" "),
    confidence: kept.reduce((sum, w) => sum + w.confidence / 100, 0) / kept.length,
  };
}

/** The Python engine's `_prepare_line`: upscale, optionally binarize, add a white border. */
export function prepareLine(image: Raster, mode: "gray" | "otsu"): Raster {
  let gray = toGray(image);
  const height = Math.max(1, gray.height);
  const scale = mode === "otsu" ? Math.max(3, MIN_LINE_HEIGHT / height) : MIN_LINE_HEIGHT / height;
  if (scale > 1) {
    gray = resize(gray, roundHalfEven(gray.width * scale), roundHalfEven(gray.height * scale), "cubic");
  }
  if (mode === "otsu") gray = otsuBinarize(gray, 255);
  // A white border helps Tesseract's line finder on tight crops.
  return padConstant(gray, 10, 255);
}

/** Netpbm bytes (PGM or PPM), which Tesseract reads without any image library. */
export function toNetpbm(image: Raster): Uint8Array {
  const header = new TextEncoder().encode(
    `${image.channels === 1 ? "P5" : "P6"}\n${image.width} ${image.height}\n255\n`);
  const out = new Uint8Array(header.length + image.data.length);
  out.set(header);
  out.set(image.data, header.length);
  return out;
}
