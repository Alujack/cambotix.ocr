import { KhmerOCRError } from "./errors.js";
import { prepareImage, type ImageInput } from "./image.js";
import type {
  HealthStatus,
  ModelsInfo,
  NationalIdResult,
  OcrResult,
  QualityReport,
  StudentIdResult,
} from "./types.js";

export interface KhmerOCRClientOptions {
  /** Server root, e.g. `http://192.168.1.10:8000`. `/v1` is appended unless present. */
  baseUrl: string;
  /** Sent as `Authorization: Bearer <key>`. Keep it on your server, never in browser code. */
  apiKey?: string | undefined;
  /** Per-request timeout. Default 60 s. */
  timeoutMs?: number | undefined;
  /** Images larger than this are rejected before upload. Default 10 MiB (the server default). */
  maxUploadBytes?: number | undefined;
  /** Extra headers on every request, e.g. your own auth token for a proxy in front of KhmerOCR. */
  headers?: Record<string, string> | undefined;
  /** A `fetch` implementation. Defaults to the global `fetch`. */
  fetch?: typeof globalThis.fetch | undefined;
}

export interface RequestOptions {
  signal?: AbortSignal | undefined;
  /** Sent as `X-Request-ID` (`[A-Za-z0-9._-]{1,64}`); the server echoes it. */
  requestId?: string | undefined;
}

export interface ImageOptions extends RequestOptions {
  /** Upload filename. Defaults to the `File` name or `image.<ext>`. */
  filename?: string | undefined;
  /** `image/jpeg`, `image/png` or `image/webp`. Detected from the bytes when omitted. */
  contentType?: string | undefined;
}

export interface ExtractOptions extends ImageOptions {
  /** Also return the raw text read in every template region. */
  includeTextRegions?: boolean | undefined;
}

const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

/** Client for a KhmerOCR server. Recognition runs on the server, not in this process. */
export class KhmerOCRClient {
  readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly timeoutMs: number;
  private readonly maxUploadBytes: number;
  private readonly headers: Record<string, string>;
  private readonly fetchImpl: typeof globalThis.fetch;

  constructor(options: KhmerOCRClientOptions) {
    if (!options.baseUrl) throw new TypeError("baseUrl is required");
    const root = options.baseUrl.replace(/\/+$/, "");
    this.baseUrl = root.endsWith("/v1") ? root : `${root}/v1`;
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.maxUploadBytes = options.maxUploadBytes ?? DEFAULT_MAX_UPLOAD_BYTES;
    this.headers = { ...options.headers };
    if (options.fetch === undefined && typeof globalThis.fetch !== "function") {
      throw new TypeError("No fetch implementation available; pass options.fetch");
    }
    // Calling the global fetch as a method of this object throws "Illegal invocation" in browsers.
    this.fetchImpl = options.fetch ?? ((input, init) => globalThis.fetch(input, init));
  }

  /** Service liveness and whether a model is loaded. Needs no API key. */
  health(options: RequestOptions = {}): Promise<HealthStatus> {
    return this.request("GET", "/health", undefined, options);
  }

  /** The active model and the installed model bundles. */
  models(options: RequestOptions = {}): Promise<ModelsInfo> {
    return this.request("GET", "/models", undefined, options);
  }

  /** Extract Cambodian National ID fields from a photo of the card front. */
  async extractNationalId(image: ImageInput, options: ExtractOptions = {}): Promise<NationalIdResult> {
    return this.upload("/documents/national-id/extract", image, options, options.includeTextRegions ?? false);
  }

  /** Extract university student ID fields (provisional layout; results always need review). */
  async extractStudentId(image: ImageInput, options: ExtractOptions = {}): Promise<StudentIdResult> {
    return this.upload("/documents/student-id/extract", image, options, options.includeTextRegions ?? false);
  }

  /** Check whether an image is usable before extraction. */
  async checkQuality(image: ImageInput, options: ImageOptions = {}): Promise<QualityReport> {
    return this.upload("/documents/quality-check", image, options);
  }

  /** Recognize general Khmer and English text. */
  async recognizeText(image: ImageInput, options: ImageOptions = {}): Promise<OcrResult> {
    return this.upload("/ocr", image, options);
  }

  private async upload<T>(path: string, image: ImageInput, options: ImageOptions,
                          includeTextRegions?: boolean): Promise<T> {
    const prepared = await prepareImage(image, {
      filename: options.filename, contentType: options.contentType, maxUploadBytes: this.maxUploadBytes,
    });
    const form = new FormData();
    form.append("file", prepared.blob, prepared.filename);
    if (includeTextRegions !== undefined) {
      form.append("include_text_regions", includeTextRegions ? "true" : "false");
    }
    return this.request("POST", path, form, options);
  }

  private async request<T>(method: "GET" | "POST", path: string, body: FormData | undefined,
                           options: RequestOptions): Promise<T> {
    // Content-Type is left to fetch so the multipart boundary is set correctly.
    const headers: Record<string, string> = { Accept: "application/json", ...this.headers };
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;
    if (options.requestId) headers["X-Request-ID"] = options.requestId;

    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, this.timeoutMs);
    const onAbort = () => controller.abort();
    const callerSignal = options.signal;
    if (callerSignal?.aborted) {
      clearTimeout(timer);
      throw new KhmerOCRError("ABORTED", "The request was aborted.");
    }
    callerSignal?.addEventListener("abort", onAbort, { once: true });

    try {
      let response: Response;
      try {
        const init: RequestInit = { method, headers, signal: controller.signal };
        if (body !== undefined) init.body = body;
        response = await this.fetchImpl(`${this.baseUrl}${path}`, init);
      } catch (error) {
        throw this.transportError(error, timedOut);
      }
      let text: string;
      try {
        text = await response.text();
      } catch (error) {
        throw this.transportError(error, timedOut);
      }
      return parseResponse<T>(response, text);
    } finally {
      clearTimeout(timer);
      callerSignal?.removeEventListener("abort", onAbort);
    }
  }

  private transportError(error: unknown, timedOut: boolean): KhmerOCRError {
    if (timedOut) {
      return new KhmerOCRError("TIMEOUT", `No response within ${this.timeoutMs} ms.`, { cause: error });
    }
    if (isAbortError(error)) {
      return new KhmerOCRError("ABORTED", "The request was aborted.", { cause: error });
    }
    return new KhmerOCRError("NETWORK_ERROR", "Could not reach the KhmerOCR server.", { cause: error });
  }
}

function isAbortError(error: unknown): boolean {
  return typeof error === "object" && error !== null && (error as { name?: unknown }).name === "AbortError";
}

function parseResponse<T>(response: Response, text: string): T {
  const headerId = response.headers.get("x-request-id");
  let parsed: unknown;
  let isJson = true;
  try {
    parsed = JSON.parse(text);
  } catch {
    isJson = false;
  }

  if (response.ok) {
    if (!isJson || typeof parsed !== "object" || parsed === null) {
      throw new KhmerOCRError("UNEXPECTED_RESPONSE", "The server returned a response that is not JSON.",
        { status: response.status, requestId: headerId });
    }
    return parsed as T;
  }

  const error = isJson && typeof parsed === "object" && parsed !== null
    ? (parsed as { error?: unknown }).error
    : undefined;
  if (typeof error === "object" && error !== null
      && typeof (error as { code?: unknown }).code === "string"
      && typeof (error as { message?: unknown }).message === "string") {
    const envelope = error as { code: string; message: string; request_id?: unknown; details?: unknown };
    throw new KhmerOCRError(envelope.code, envelope.message, {
      status: response.status,
      requestId: typeof envelope.request_id === "string" ? envelope.request_id : headerId,
      details: typeof envelope.details === "object" && envelope.details !== null
        ? envelope.details as Record<string, unknown>
        : null,
    });
  }
  throw new KhmerOCRError("UNEXPECTED_RESPONSE",
    `The server returned HTTP ${response.status} without an error body.`,
    { status: response.status, requestId: headerId });
}
