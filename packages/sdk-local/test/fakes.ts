import { createHash } from "node:crypto";
import sharp from "sharp";
import type { Raster } from "../src/image/raster.js";
import type { Platform } from "../src/platform.js";
import { Recognizer, type RecognizedText, type RegionHint } from "../src/recognition/base.js";
import type { OnnxSession } from "../src/recognition/onnxCtc.js";
import type { TextRegion } from "../src/schemas.js";

/** Returns pre-set text per field name, like the Python `ScriptedRecognizer`. */
export class ScriptedRecognizer extends Recognizer {
  readonly name = "scripted";
  calls: Array<string | null> = [];
  closed = false;

  constructor(private readonly responses: Record<string, [string, number | null]> = {},
              private readonly pageRegions: TextRegion[] = []) {
    super();
  }

  async recognizeLine(_image: Raster, hint?: RegionHint): Promise<RecognizedText> {
    const name = hint?.name ?? null;
    this.calls.push(name);
    const [text, confidence] = this.responses[name ?? ""] ?? ["", null];
    return { text, confidence };
  }

  override async recognizePage(): Promise<TextRegion[]> {
    return [...this.pageRegions];
  }

  override async close(): Promise<void> {
    this.closed = true;
  }
}

/** Emits a fixed class sequence, one class per time step (`[time, 1, classes]` logits). */
export class FakeSession implements OnnxSession {
  lastDims: readonly number[] | null = null;
  released = false;

  constructor(private readonly classes: number[], private readonly nClasses = 8) {}

  async run(_input: Float32Array, dims: readonly number[]) {
    this.lastDims = dims;
    const data = new Float32Array(this.classes.length * this.nClasses).fill(-10);
    this.classes.forEach((c, t) => { data[t * this.nClasses + c] = 10; });
    return { data, dims: [this.classes.length, 1, this.nClasses] };
  }

  async release() {
    this.released = true;
  }
}

export const sha256 = (data: Uint8Array | string) => createHash("sha256").update(data).digest("hex");

/** A platform over in-memory bundle files. */
export function memoryPlatform(files: Record<string, Uint8Array | string>,
                               session: () => OnnxSession): Platform {
  return {
    name: "node",
    async readFile(root, path) {
      const data = files[`${root}/${path}`];
      if (data === undefined) throw new Error("ENOENT");
      return typeof data === "string" ? new TextEncoder().encode(data) : data;
    },
    sha256Hex: async (data) => sha256(data),
    createOnnxSession: async () => session(),
    createTesseractRunner: async () => { throw new Error("no tesseract in tests"); },
    resolve: (root, path) => `${root}/${path}`,
    async decodeImage(data): Promise<Raster> {
      const { data: pixels, info } = await sharp(data).removeAlpha().raw().toBuffer({ resolveWithObject: true });
      return { width: info.width, height: info.height, channels: 3, data: new Uint8Array(pixels) };
    },
  };
}

/** Manifest JSON for in-memory bundle files, with correct checksums. */
export function manifestFor(files: Record<string, string | Uint8Array>, recognizer: object): string {
  const entries = Object.entries(files).map(([path, data]) => ({ path, sha256: sha256(data) }))
    .sort((a, b) => (a.path < b.path ? -1 : 1));
  const digest = sha256(entries.map((f) => `${f.path}\0${f.sha256}\n`).join(""));
  return JSON.stringify({
    model_id: "test-model", task: "khmer_id_extraction", runtime: "local", recognition_model: "test",
    model_version: "0.0.1", schema_version: "1.0", license: "Apache-2.0", license_approved: false,
    recognizer, files: entries, weights_sha256: digest,
  });
}
