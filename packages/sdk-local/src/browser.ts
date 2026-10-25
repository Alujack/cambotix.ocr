/**
 * Browser entry point: `import { loadKhmerOCR } from "@khmerocr/local/browser"`.
 *
 * Everything runs on the user's device: the photo never leaves it. Works in
 * pages and in Web Workers (recommended, so the page stays responsive).
 */
import { KhmerOCRLocal, type LoadOptions } from "./engine.js";
import type { Raster } from "./image/raster.js";
import type { Platform } from "./platform.js";
import { createOrtSession, createTesseractJsRunner, type OrtModule, type TesseractModule } from "./recognition/runtimes.js";

export * from "./index.js";

export interface BrowserLoadOptions extends LoadOptions {
  /** URL of the model bundle folder, e.g. `/models/khmerocr-id-v1/`. Relative URLs resolve against the page. */
  model: string;
  onnx?: {
    /**
     * Where ONNX Runtime's `.wasm` files are served (`ort.env.wasm.wasmPaths`), e.g.
     * `/khmerocr/ort/`. `npx khmerocr-local-assets public/khmerocr` copies them there.
     */
    wasmPaths?: string | Record<string, string>;
    /** WebAssembly threads; more than 1 needs a cross-origin isolated page. */
    numThreads?: number;
  };
  /**
   * tesseract.js `workerPath` and `corePath`, e.g. `/khmerocr/tesseract/worker.min.js` and
   * `/khmerocr/tesseract/`. Without them tesseract.js loads its code from the jsDelivr CDN.
   */
  tesseractWorker?: { workerPath?: string; corePath?: string };
}

function baseUrl(root: string): string {
  const location = (globalThis as { location?: { href: string } }).location;
  const url = new URL(root, location?.href);
  if (!url.pathname.endsWith("/")) url.pathname += "/";
  return url.href;
}

function browserPlatform(options: BrowserLoadOptions): Platform {
  return {
    name: "browser",
    async readFile(root, path) {
      const response = await fetch(new URL(path, baseUrl(root)));
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return new Uint8Array(await response.arrayBuffer());
    },
    async sha256Hex(data) {
      const digest = await crypto.subtle.digest("SHA-256", data as Uint8Array<ArrayBuffer>);
      return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
    },
    async createOnnxSession(model) {
      // The CPU (WebAssembly) build: half the download of the default WebGPU-capable build.
      const module = await import("onnxruntime-web/wasm") as unknown as { default?: OrtModule } & OrtModule;
      const ort = module.default ?? module;
      const wasm = (ort.env as unknown as { wasm: Record<string, unknown> }).wasm;
      if (options.onnx?.wasmPaths !== undefined) wasm["wasmPaths"] = options.onnx.wasmPaths;
      if (options.onnx?.numThreads !== undefined) wasm["numThreads"] = options.onnx.numThreads;
      return createOrtSession(ort, model, { executionProviders: ["wasm"] });
    },
    async createTesseractRunner(languages, tessdata, cacheKey) {
      const module = await import("tesseract.js") as unknown as { default?: TesseractModule } & TesseractModule;
      const tesseract = module.default ?? module;
      // Each language set gets its own worker; caching the (checked) language files in
      // IndexedDB under the bundle's digest downloads them once instead of per worker.
      return createTesseractJsRunner(tesseract, languages, tessdata,
        (bytes) => new Blob([bytes as Uint8Array<ArrayBuffer>]),
        { ...options.tesseractWorker, cacheMethod: "write", cachePath: `khmerocr-${cacheKey}` });
    },
    resolve(root, path) {
      return new URL(path, baseUrl(root)).href.replace(/\/$/u, "");
    },
    async decodeImage(data, format): Promise<Raster> {
      // Like the Python engine (Pillow): apply EXIF orientation, no colour management.
      const bitmap = await createImageBitmap(new Blob([data as Uint8Array<ArrayBuffer>], { type: format }), {
        imageOrientation: "from-image", colorSpaceConversion: "none", premultiplyAlpha: "none",
      });
      try {
        const { width, height } = bitmap;
        const canvas = typeof OffscreenCanvas !== "undefined"
          ? new OffscreenCanvas(width, height)
          : Object.assign(document.createElement("canvas"), { width, height });
        const context = canvas.getContext("2d") as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
        if (context === null) throw new Error("no 2D canvas context");
        context.drawImage(bitmap, 0, 0);
        const rgba = context.getImageData(0, 0, width, height).data;
        const rgb = new Uint8Array(width * height * 3);
        for (let i = 0, j = 0; j < rgba.length; i += 3, j += 4) {
          rgb[i] = rgba[j]!;
          rgb[i + 1] = rgba[j + 1]!;
          rgb[i + 2] = rgba[j + 2]!;
        }
        return { width, height, channels: 3, data: rgb };
      } finally {
        bitmap.close();
      }
    },
  };
}

/** Load a model bundle and return a ready engine. Reuse it: loading downloads the model once. */
export function loadKhmerOCR(options: BrowserLoadOptions): Promise<KhmerOCRLocal> {
  return KhmerOCRLocal.create(browserPlatform(options), options);
}
