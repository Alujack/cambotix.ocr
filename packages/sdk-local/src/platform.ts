/**
 * What differs between Node and browsers: reading bundle files, hashing,
 * decoding images, and creating ONNX Runtime and tesseract.js instances.
 * `node.ts` and `browser.ts` each provide one implementation.
 */
import type { Raster } from "./image/raster.js";
import type { OnnxSession } from "./recognition/onnxCtc.js";
import type { TesseractRunner } from "./recognition/tesseract.js";

export type ImageFormat = "image/jpeg" | "image/png" | "image/webp";

export interface Platform {
  readonly name: "node" | "browser";
  /** Read `path` (relative, `/`-separated) inside the bundle at `root`. */
  readFile(root: string, path: string): Promise<Uint8Array>;
  sha256Hex(data: Uint8Array): Promise<string>;
  createOnnxSession(model: Uint8Array): Promise<OnnxSession>;
  /**
   * `tessdata` is the bundle's tessdata folder (from `resolve`); `cacheKey`
   * identifies its content, for platforms that cache language files.
   */
  createTesseractRunner(languages: string, tessdata: string, cacheKey: string): Promise<TesseractRunner>;
  /** Location of `path` inside the bundle at `root`. */
  resolve(root: string, path: string): string;
  /** Decode encoded image bytes into upright RGB pixels (EXIF orientation applied). */
  decodeImage(data: Uint8Array, format: ImageFormat): Promise<Raster>;
}
