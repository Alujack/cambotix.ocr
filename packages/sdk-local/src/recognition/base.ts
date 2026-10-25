/**
 * Recognizer interface: every OCR backend is wrapped in a `Recognizer` so the
 * rest of the engine never depends on a specific model.
 */
import { detectTextLines } from "../detection/textLines.js";
import { crop, type Raster } from "../image/raster.js";
import type { TextRegion } from "../schemas.js";

export type Script = "khmer" | "latin" | "digits" | "mixed";

/**
 * What the engine expects to find in a crop. Backends may use hints to select
 * languages or character sets, but must return what they actually read.
 */
export interface RegionHint {
  name?: string | null;
  script: Script;
  languages?: string | null;
  charWhitelist?: string | null;
}

export interface RecognizedText {
  text: string;
  /** Recognition confidence in [0, 1], or null if the backend has no score. */
  confidence: number | null;
}

export const DEFAULT_HINT: RegionHint = { script: "mixed" };

export abstract class Recognizer {
  abstract readonly name: string;

  /** Recognize a single text line or field crop. */
  abstract recognizeLine(image: Raster, hint?: RegionHint): Promise<RecognizedText>;

  /**
   * Reads in order of preference. Callers stop as soon as a read validates,
   * so later candidates are only computed when needed.
   */
  async *recognizeCandidates(image: Raster, hint?: RegionHint): AsyncGenerator<RecognizedText> {
    yield await this.recognizeLine(image, hint);
  }

  /**
   * Recognize all text on a page. Backends with native layout analysis
   * override this; the default detects lines and reads each one.
   */
  async recognizePage(image: Raster): Promise<TextRegion[]> {
    const regions: TextRegion[] = [];
    for (const box of detectTextLines(image)) {
      const result = await this.recognizeLine(crop(image, box.x, box.y, box.width, box.height));
      if (result.text.trim()) {
        regions.push({ text: result.text, confidence: result.confidence, bbox: box });
      }
    }
    return regions;
  }

  describe(): Record<string, string> {
    return { backend: this.name };
  }

  /** Release backend resources. Must be idempotent. */
  async close(): Promise<void> {}
}
