/**
 * Node entry point: `import { loadKhmerOCR } from "@khmerocr/local/node"`.
 *
 * Needs the optional peer dependencies `onnxruntime-node` (1.27 or older)
 * and `sharp` (to decode JPEG, PNG and WebP files).
 */
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { KhmerOCRLocal, type LoadOptions } from "./engine.js";
import { ocrError } from "./errors.js";
import type { Raster } from "./image/raster.js";
import type { Platform } from "./platform.js";
import { createOrtSession, createTesseractJsRunner, type OrtModule, type TesseractModule } from "./recognition/runtimes.js";

export * from "./index.js";

export interface NodeLoadOptions extends LoadOptions {
  /** `model` is a folder path, or an http(s) URL. */
  model: string;
  /** Threads ONNX Runtime may use for one inference. Default: ONNX Runtime's choice. */
  intraOpNumThreads?: number;
}

const isUrl = (root: string) => /^https?:\/\//iu.test(root);

/** Literal specifiers so bundlers can see the imports. */
async function optionalImport<T>(load: () => Promise<unknown>, name: string, purpose: string): Promise<T> {
  try {
    const module = await load() as { default?: T } & T;
    return module.default ?? module;
  } catch (cause) {
    throw ocrError("MODEL_NOT_AVAILABLE", `${purpose} needs the '${name}' package: npm install ${name}`,
      undefined, cause);
  }
}

function nodePlatform(options: NodeLoadOptions): Platform {
  return {
    name: "node",
    async readFile(root, path) {
      if (isUrl(root)) {
        const response = await fetch(new URL(path, root.endsWith("/") ? root : `${root}/`));
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return new Uint8Array(await response.arrayBuffer());
      }
      return new Uint8Array(await readFile(join(root, ...path.split("/"))));
    },
    async sha256Hex(data) {
      return createHash("sha256").update(data).digest("hex");
    },
    async createOnnxSession(model) {
      const ort = await optionalImport<OrtModule>(() => import("onnxruntime-node"), "onnxruntime-node", "Running the model in Node");
      return createOrtSession(ort, model, {
        executionProviders: ["cpu"],
        ...(options.intraOpNumThreads === undefined ? {} : { intraOpNumThreads: options.intraOpNumThreads }),
      });
    },
    async createTesseractRunner(languages, tessdata) {
      const tesseract = await optionalImport<TesseractModule>(() => import("tesseract.js"), "tesseract.js", "The Tesseract fallback");
      return createTesseractJsRunner(tesseract, languages, tessdata, (bytes) => Buffer.from(bytes), {});
    },
    resolve(root, path) {
      return isUrl(root) ? new URL(path, root.endsWith("/") ? root : `${root}/`).href : join(root, ...path.split("/"));
    },
    async decodeImage(data): Promise<Raster> {
      type Sharp = (input: Uint8Array, options: Record<string, unknown>) => {
        removeAlpha(): ReturnType<Sharp>;
        toColourspace(space: string): ReturnType<Sharp>;
        raw(options: Record<string, unknown>): ReturnType<Sharp>;
        toBuffer(options: { resolveWithObject: true }):
          Promise<{ data: Buffer; info: { width: number; height: number; channels: number } }>;
      };
      const sharp = await optionalImport<Sharp>(() => import("sharp"), "sharp", "Decoding image files in Node");
      // Like the Python engine (Pillow): apply EXIF orientation, ignore ICC profiles, drop alpha.
      const { data: pixels, info } = await sharp(data, { autoOrient: true, ignoreIcc: true, limitInputPixels: false })
        .removeAlpha().toColourspace("srgb").raw({ depth: "uchar" }).toBuffer({ resolveWithObject: true });
      if (info.channels !== 3) throw new Error("unexpected channel count");
      return { width: info.width, height: info.height, channels: 3, data: new Uint8Array(pixels) };
    },
  };
}

/** Load a model bundle and return a ready engine. Reuse it: loading takes a second or two. */
export function loadKhmerOCR(options: NodeLoadOptions): Promise<KhmerOCRLocal> {
  return KhmerOCRLocal.create(nodePlatform(options), options);
}
