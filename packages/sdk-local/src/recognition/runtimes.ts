/** Adapters from ONNX Runtime and tesseract.js to the recognizers' interfaces. */
import { ocrError } from "../errors.js";
import type { Raster } from "../image/raster.js";
import type { OnnxSession } from "./onnxCtc.js";
import { toNetpbm, type TesseractRunner, type TesseractWord } from "./tesseract.js";

/** The parts of `onnxruntime-node` / `onnxruntime-web` used here. */
export interface OrtModule {
  env: { versions: Record<string, string | undefined> };
  Tensor: new (type: "float32", data: Float32Array, dims: readonly number[]) => unknown;
  InferenceSession: {
    create(model: Uint8Array, options?: Record<string, unknown>): Promise<OrtSession>;
  };
}

interface OrtSession {
  inputNames: readonly string[];
  outputNames: readonly string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: unknown; dims: readonly number[] }>>;
  release(): Promise<void>;
}

// onnxruntime 1.29 added telemetry that contacts Microsoft even when
// disabled (see docs/privacy.md); refuse it, as the Python engine does.
const FIRST_TELEMETRY_VERSION = [1, 29] as const;

export function checkOrtVersion(version: string | undefined): void {
  const [major, minor] = (version ?? "").split(".").map(Number);
  if (major === undefined || minor === undefined || Number.isNaN(major) || Number.isNaN(minor)) {
    throw ocrError("MODEL_NOT_AVAILABLE", "Unrecognized onnxruntime version.");
  }
  if (major > FIRST_TELEMETRY_VERSION[0] || (major === FIRST_TELEMETRY_VERSION[0] && minor >= FIRST_TELEMETRY_VERSION[1])) {
    throw ocrError("MODEL_NOT_AVAILABLE",
      "This onnxruntime version sends telemetry to Microsoft, which breaks KhmerOCR's offline guarantee. "
      + "Install onnxruntime 1.27 or older.", { onnxruntime_version: version });
  }
}

export async function createOrtSession(ort: OrtModule, model: Uint8Array,
                                       options: Record<string, unknown>): Promise<OnnxSession> {
  checkOrtVersion(ort.env.versions["common"]);
  const session = await ort.InferenceSession.create(model, options);
  const inputName = session.inputNames[0]!;
  const outputName = session.outputNames[0]!;
  return {
    async run(input, dims) {
      const output = await session.run({ [inputName]: new ort.Tensor("float32", input, dims) });
      const tensor = output[outputName]!;
      return { data: tensor.data as Float32Array, dims: tensor.dims };
    },
    release: () => session.release(),
  };
}

interface TesseractWorker {
  setParameters(params: Record<string, string>): Promise<unknown>;
  recognize(image: unknown, options: object, output: object): Promise<{ data: TesseractPage }>;
  terminate(): Promise<unknown>;
}

interface TesseractPage {
  blocks: Array<{ paragraphs: Array<{ lines: Array<{ words: Array<{
    text: string; confidence: number; bbox: { x0: number; y0: number; x1: number; y1: number };
  }> }> }> }> | null;
}

export interface TesseractModule {
  createWorker(langs: string, oem: number, options: Record<string, unknown>): Promise<TesseractWorker>;
}

/**
 * A tesseract.js worker for one language set. Calls are serialized because
 * parameters are set on the worker before each read.
 */
export async function createTesseractJsRunner(
  tesseract: TesseractModule, languages: string, langPath: string,
  toImage: (netpbm: Uint8Array) => unknown, workerOptions: Record<string, unknown>,
): Promise<TesseractRunner> {
  let failure: unknown = null;
  // LSTM only (--oem 1). Language data comes from the bundle, uncompressed; it is
  // not cached unless the platform asks for it.
  const worker = await tesseract.createWorker(languages, 1, {
    cacheMethod: "none",
    ...workerOptions,
    langPath,
    gzip: false,
    errorHandler: (error: unknown) => {
      failure = error;
    },
  });
  let queue: Promise<unknown> = Promise.resolve();
  return {
    recognize(image: Raster, psm: number, whitelist: string | null): Promise<TesseractWord[]> {
      const run = async () => {
        if (failure !== null) throw failure;
        await worker.setParameters({
          tessedit_pageseg_mode: String(psm),
          tessedit_char_whitelist: whitelist ?? "",
        });
        const { data } = await worker.recognize(toImage(toNetpbm(image)), {}, { blocks: true, text: false });
        const words: TesseractWord[] = [];
        let line = 0;
        for (const block of data.blocks ?? []) {
          for (const paragraph of block.paragraphs) {
            for (const textLine of paragraph.lines) {
              for (const word of textLine.words) {
                words.push({ text: word.text, confidence: word.confidence, line, bbox: word.bbox });
              }
              line++;
            }
          }
        }
        return words;
      };
      const result = queue.then(run, run);
      queue = result.catch(() => undefined);
      return result;
    },
    terminate: async () => {
      await worker.terminate();
    },
  };
}
