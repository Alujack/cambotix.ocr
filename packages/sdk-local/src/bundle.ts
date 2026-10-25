/**
 * Model bundles: the folder the Python engine installs (`manifest.json` plus
 * the files it lists). Every file is checked against the manifest's SHA-256
 * before it is used, and the manifest's file list against its digest.
 */
import type { ModelSummary } from "@khmerocr/client";
import { ocrError } from "./errors.js";
import type { Platform } from "./platform.js";
import type { Recognizer } from "./recognition/base.js";
import { OnnxCtcRecognizer } from "./recognition/onnxCtc.js";
import { ScriptRouter } from "./recognition/router.js";
import { TesseractRecognizer } from "./recognition/tesseract.js";

const REQUIRED_KEYS = ["model_id", "task", "runtime", "recognition_model", "model_version",
  "schema_version", "license", "recognizer", "files", "weights_sha256"] as const;

export interface ModelFile {
  path: string;
  sha256: string;
}

export interface ModelManifest {
  modelId: string;
  task: string;
  runtime: string;
  recognitionModel: string;
  modelVersion: string;
  schemaVersion: string;
  license: string;
  licenseApproved: boolean;
  recognizerBackend: string;
  recognizerOptions: Record<string, unknown>;
  files: ModelFile[];
  weightsSha256: string;
}

export function parseManifest(data: unknown): ModelManifest {
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw ocrError("MODEL_NOT_AVAILABLE", "The model manifest is malformed.");
  }
  const record = data as Record<string, unknown>;
  const missing = REQUIRED_KEYS.filter((key) => !(key in record));
  if (missing.length > 0) {
    throw ocrError("MODEL_NOT_AVAILABLE", "The model manifest is incomplete.", { missing_keys: missing });
  }
  try {
    const recognizer = record["recognizer"] as Record<string, unknown>;
    const files = (record["files"] as Array<Record<string, unknown>>).map((f) => {
      if (typeof f["path"] !== "string" || typeof f["sha256"] !== "string") throw new Error("file");
      return { path: f["path"], sha256: f["sha256"].toLowerCase() };
    });
    if (typeof recognizer["backend"] !== "string") throw new Error("backend");
    return {
      modelId: String(record["model_id"]),
      task: String(record["task"]),
      runtime: String(record["runtime"]),
      recognitionModel: String(record["recognition_model"]),
      modelVersion: String(record["model_version"]),
      schemaVersion: String(record["schema_version"]),
      license: String(record["license"]),
      licenseApproved: record["license_approved"] === true,
      recognizerBackend: recognizer["backend"],
      recognizerOptions: { ...(recognizer["options"] as Record<string, unknown> | undefined) },
      files,
      weightsSha256: String(record["weights_sha256"]).toLowerCase(),
    };
  } catch (cause) {
    throw ocrError("MODEL_NOT_AVAILABLE", "The model manifest is malformed.", undefined, cause);
  }
}

export function manifestSummary(manifest: ModelManifest): ModelSummary {
  return {
    model_id: manifest.modelId,
    model_version: manifest.modelVersion,
    task: manifest.task,
    runtime: manifest.runtime,
    recognition_model: manifest.recognitionModel,
    schema_version: manifest.schemaVersion,
    license: manifest.license,
    license_approved: manifest.licenseApproved,
    recognizer_backend: manifest.recognizerBackend,
    weights_sha256: manifest.weightsSha256,
  };
}

/** Digest identifying the exact set of files in a bundle (Python `bundle_digest`). */
export async function bundleDigest(platform: Platform, files: readonly ModelFile[]): Promise<string> {
  const lines = [...files].sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0))
    .map((f) => `${f.path}\0${f.sha256}\n`).join("");
  return platform.sha256Hex(new TextEncoder().encode(lines));
}

/** Reject paths that could leave the bundle folder. */
export function checkBundlePath(path: string): string {
  const segments = path.split("/");
  if (!path || path.includes("\\") || path.includes(":")
      || segments.some((s) => s === "" || s === "." || s === "..")) {
    throw ocrError("MODEL_NOT_AVAILABLE", "Model files must be inside the model bundle.", { path });
  }
  return path;
}

/** Reads bundle files, checking each one against the manifest. */
export class Bundle {
  private readonly checksums: Map<string, string>;

  private constructor(readonly platform: Platform, readonly root: string,
                      readonly manifest: ModelManifest, readonly verify: boolean) {
    this.checksums = new Map(manifest.files.map((f) => [f.path, f.sha256]));
  }

  static async open(platform: Platform, root: string, verify: boolean): Promise<Bundle> {
    let raw: Uint8Array;
    try {
      raw = await platform.readFile(root, "manifest.json");
    } catch (cause) {
      throw ocrError("MODEL_NOT_AVAILABLE",
        "No model manifest found. Point `model` at a model bundle folder.", { model: root }, cause);
    }
    let data: unknown;
    try {
      data = JSON.parse(new TextDecoder().decode(raw));
    } catch (cause) {
      throw ocrError("MODEL_NOT_AVAILABLE", "The model manifest could not be read.", undefined, cause);
    }
    const manifest = parseManifest(data);
    manifest.files.forEach((f) => checkBundlePath(f.path));
    if (verify && await bundleDigest(platform, manifest.files) !== manifest.weightsSha256) {
      throw ocrError("MODEL_NOT_AVAILABLE", "The model bundle digest does not match its file list.");
    }
    return new Bundle(platform, root, manifest, verify);
  }

  has(path: string): boolean {
    return this.checksums.has(path);
  }

  /** Read a file listed in the manifest and check its SHA-256. */
  async read(path: string): Promise<Uint8Array> {
    checkBundlePath(path);
    const expected = this.checksums.get(path);
    if (expected === undefined) {
      throw ocrError("MODEL_NOT_AVAILABLE", "A model file is not listed in the manifest.", { path });
    }
    let data: Uint8Array;
    try {
      data = await this.platform.readFile(this.root, path);
    } catch (cause) {
      throw ocrError("MODEL_NOT_AVAILABLE", "A model file is missing.", { path }, cause);
    }
    if (this.verify && await this.platform.sha256Hex(data) !== expected) {
      throw ocrError("MODEL_NOT_AVAILABLE", "A model file failed checksum verification.", { path });
    }
    return data;
  }
}

export interface RecognizerOptions {
  /** false: leave out Tesseract backends (fewer downloads, but no fallback reads). */
  tesseract: boolean;
}

/** Build the recognizer a manifest describes (Python `create_recognizer`). */
export async function createRecognizer(bundle: Bundle, backend: string,
                                       options: Record<string, unknown>,
                                       settings: RecognizerOptions): Promise<Recognizer> {
  if (backend === "onnx-ctc") return createOnnxCtc(bundle, options);
  if (backend === "tesseract") return createTesseract(bundle, options);
  if (backend === "router") return createRouter(bundle, options, settings);
  throw ocrError("MODEL_NOT_AVAILABLE", `Recognizer backend '${backend}' is not available.`,
    { available_backends: ["onnx-ctc", "router", "tesseract"] });
}

async function createOnnxCtc(bundle: Bundle, options: Record<string, unknown>): Promise<Recognizer> {
  const tokensFile = String(options["tokens_file"] ?? "tokens.txt");
  const modelFile = String(options["model"] ?? "rec.onnx");
  const tokens = new TextDecoder().decode(await bundle.read(tokensFile)).replace(/\n+$/u, "");
  const model = await bundle.read(modelFile);
  let session;
  try {
    session = await bundle.platform.createOnnxSession(model);
  } catch (cause) {
    if (cause instanceof Error && cause.name === "KhmerOCRError") throw cause;
    throw ocrError("MODEL_NOT_AVAILABLE", "The ONNX model could not be loaded.", undefined, cause);
  }
  return new OnnxCtcRecognizer(session, {
    modelFile: modelFile.split("/").pop()!,
    tokens,
    tokenOffset: Math.trunc(Number(options["token_offset"] ?? 3)),
    inputHeight: Math.trunc(Number(options["input_height"] ?? 32)),
    restoreSpaces: Boolean(options["restore_spaces"] ?? true),
    confidenceCalibration: (options["confidence_calibration"] as number[][] | undefined) ?? null,
  });
}

function createTesseract(bundle: Bundle, options: Record<string, unknown>): Recognizer {
  const tessdataDir = checkBundlePath(String(options["tessdata_dir"] ?? "tessdata"));
  const languages = String(options["languages"] ?? "khm+eng");
  const required = new Set([...languages.split("+"), "khm", "eng"]);
  const missing = [...required].filter((lang) => !bundle.has(`${tessdataDir}/${lang}.traineddata`)).sort();
  if (missing.length > 0) {
    throw ocrError("MODEL_NOT_AVAILABLE", "Tesseract language data is missing from the model bundle.",
      { missing_languages: missing });
  }
  const verified = new Map<string, Promise<void>>();
  const location = bundle.platform.resolve(bundle.root, tessdataDir);
  return new TesseractRecognizer(async (langs) => {
    // tesseract.js reads the language files itself, so check them first.
    await Promise.all(langs.split("+").map((lang) => {
      const path = `${tessdataDir}/${lang}.traineddata`;
      let check = verified.get(path);
      if (check === undefined) {
        check = bundle.read(path).then(() => undefined);
        verified.set(path, check);
        check.catch(() => verified.delete(path));
      }
      return check;
    }));
    return bundle.platform.createTesseractRunner(langs, location);
  }, languages);
}

async function createRouter(bundle: Bundle, options: Record<string, unknown>,
                            settings: RecognizerOptions): Promise<Recognizer> {
  const backends = new Map<string, Recognizer>();
  try {
    const specs = options["backends"] as Record<string, { backend: string; options?: Record<string, unknown> }>;
    if (typeof specs !== "object" || specs === null) throw new Error("backends");
    for (const [name, spec] of Object.entries(specs)) {
      if (spec.backend === "router") throw ocrError("MODEL_NOT_AVAILABLE", "Routers cannot be nested.");
      if (spec.backend === "tesseract" && !settings.tesseract) continue;
      backends.set(name, await createRecognizer(bundle, spec.backend, { ...spec.options }, settings));
    }
    const available = (name: string) => backends.has(name);
    const routes: Record<string, string[]> = {};
    for (const [script, names] of Object.entries(options["routes"] as Record<string, string | string[]> ?? {})) {
      const kept = (typeof names === "string" ? [names] : names).filter(available);
      if (kept.length > 0) routes[script] = kept;
    }
    const fallback = [...backends.keys()][0];
    const pick = (name: unknown) => (typeof name === "string" && available(name) ? name : fallback);
    if (fallback === undefined || typeof options["default"] !== "string") throw new Error("default");
    return new ScriptRouter(backends, routes, pick(options["default"])!, pick(options["page"] ?? options["default"]));
  } catch (cause) {
    await Promise.all([...backends.values()].map((b) => b.close()));
    if (cause instanceof Error && cause.name === "KhmerOCRError") throw cause;
    throw ocrError("MODEL_NOT_AVAILABLE", "The recognizer routing configuration is invalid.", undefined, cause);
  }
}
