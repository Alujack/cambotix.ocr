/**
 * The local engine: the Python `KhmerOCR` engine's behaviour, running in
 * this process. Results use the REST API's JSON shapes.
 *
 * Logging is deliberately absent: the engine never logs images, recognized
 * text, field values, or anything derived from document content.
 */
import type {
  ActiveModel,
  DocumentResult,
  NationalIdField,
  OcrResult,
  QualityReport,
  StudentIdField,
} from "@khmerocr/client";
import { Bundle, createRecognizer, manifestSummary, type ModelManifest } from "./bundle.js";
import { resolveConfig, type EngineConfig, type EngineConfigOverrides } from "./config.js";
import { TemplateExtractor, type Extraction } from "./documents/extractor.js";
import { DOCUMENT_TEMPLATES, templateForDocument } from "./documents/layout.js";
import { KhmerOCRError, ocrError } from "./errors.js";
import { loadImage, type LocalImageInput } from "./image/input.js";
import type { Raster } from "./image/raster.js";
import type { Platform } from "./platform.js";
import { assessQuality } from "./preprocessing/quality.js";
import type { Recognizer } from "./recognition/base.js";
import { qualityReport, roundConfidence, SCHEMA_VERSION, type QualityMeasurements } from "./schemas.js";
import { VERSION } from "./version.js";

export interface LoadOptions {
  /**
   * The model bundle: a folder path (Node) or URL (browser, e.g.
   * `/models/khmerocr-id-v1/`) holding `manifest.json` and its files.
   */
  model: string;
  /** Check every model file's SHA-256 against the manifest. Default true. */
  verify?: boolean;
  /**
   * Use the bundle's Tesseract fallback. Default true. With false, the
   * Tesseract language files are never downloaded, but fields the ONNX model
   * misreads get no second reading, so results differ from the Python engine.
   */
  tesseract?: boolean;
  /** Engine settings; the defaults match the Python engine. */
  config?: EngineConfigOverrides;
}

export interface CallOptions {
  /** Abort the call; it then rejects with code `ABORTED`. */
  signal?: AbortSignal | undefined;
}

export interface LocalExtractOptions extends CallOptions {
  /** Also return the raw text read in every template region. */
  includeTextRegions?: boolean | undefined;
}

export type NationalIdResult = DocumentResult<NationalIdField>;
export type StudentIdResult = DocumentResult<StudentIdField>;

/** Runs KhmerOCR in this process. Create one with `loadKhmerOCR` and reuse it. */
export class KhmerOCRLocal {
  readonly config: EngineConfig;
  readonly manifest: ModelManifest;
  private recognizer: Recognizer | null;
  private readonly extractors = new Map<string, TemplateExtractor>();
  private queue: Promise<unknown> = Promise.resolve();

  private constructor(private readonly platform: Platform, recognizer: Recognizer,
                      manifest: ModelManifest, config: EngineConfig) {
    this.recognizer = recognizer;
    this.manifest = manifest;
    this.config = config;
  }

  /** Load and verify a model bundle. Prefer the `loadKhmerOCR` of your platform's entry point. */
  static async create(platform: Platform, options: LoadOptions): Promise<KhmerOCRLocal> {
    if (!options.model) throw new TypeError("options.model is required");
    const bundle = await Bundle.open(platform, options.model, options.verify ?? true);
    const { manifest } = bundle;
    const recognizer = await createRecognizer(bundle, manifest.recognizerBackend,
      manifest.recognizerOptions, { tesseract: options.tesseract ?? true });
    return new KhmerOCRLocal(platform, recognizer, manifest, resolveConfig(options.config));
  }

  /** Document types `extractDocument` accepts. */
  static supportedDocuments(): string[] {
    return Object.keys(DOCUMENT_TEMPLATES).sort();
  }

  /** Cambodian National ID, front side. */
  extractNationalId(image: LocalImageInput, options: LocalExtractOptions = {}): Promise<NationalIdResult> {
    return this.extractDocument(image, "khmer_national_id", options) as Promise<NationalIdResult>;
  }

  /** Student ID card, front side. */
  extractStudentId(image: LocalImageInput, options: LocalExtractOptions = {}): Promise<StudentIdResult> {
    return this.extractDocument(image, "student_id_card", options) as Promise<StudentIdResult>;
  }

  extractDocument(image: LocalImageInput, documentType: string,
                  options: LocalExtractOptions = {}): Promise<DocumentResult> {
    return this.serialize(options.signal, async () => {
      const started = now();
      const extractor = this.extractor(documentType);
      const pixels = await this.decode(image);
      const quality = this.gate(pixels, "extraction");
      const extraction = await guarded(() => extractor.extract(pixels, quality, options.signal));
      return documentResult(extraction, options.includeTextRegions ?? false, {
        quality: qualitySummary(quality),
        processing_ms: elapsed(started),
        ...this.modelMetadata(),
      });
    });
  }

  /** Blur, glare, exposure, size, and orientation problems. */
  checkQuality(image: LocalImageInput, options: CallOptions = {}): Promise<QualityReport> {
    return this.serialize(options.signal, async () =>
      qualityReport(assessQuality(await this.decode(image), this.config.quality)));
  }

  /** General Khmer/English text recognition. */
  recognizeText(image: LocalImageInput, options: CallOptions = {}): Promise<OcrResult> {
    return this.serialize(options.signal, async () => {
      const started = now();
      const recognizer = this.activeRecognizer();
      const pixels = await this.decode(image);
      const quality = this.gate(pixels, "recognition");
      const regions = await guarded(() => recognizer.recognizePage(pixels));
      const confidences = regions.map((r) => r.confidence).filter((c): c is number => c !== null);
      const low = regions.filter((r) => r.confidence === null
        || r.confidence < this.config.reviewConfidenceThreshold);
      const warnings = [...quality.issues];
      if (regions.length === 0) warnings.push("no_text_detected");
      if (low.length > 0) warnings.push("low_confidence_regions");
      const review = low.length > 0 || regions.length === 0 || quality.issues.length > 0;
      return {
        schema_version: SCHEMA_VERSION,
        document_type: "general_text",
        status: regions.length === 0 ? "failed" : review ? "review_required" : "success",
        text: regions.map((r) => r.text).join("\n"),
        regions: regions.map((r) => ({ text: r.text, confidence: roundConfidence(r.confidence),
                                       bbox: r.bbox, label: r.label ?? null })),
        warnings,
        review_required: review,
        metadata: {
          quality: qualitySummary(quality),
          region_count: regions.length,
          mean_confidence: confidences.length
            ? Math.round(confidences.reduce((a, b) => a + b, 0) / confidences.length * 1e4) / 1e4 : null,
          processing_ms: elapsed(started),
          ...this.modelMetadata(),
        },
      };
    });
  }

  /** The loaded model, in the shape of the REST API's `GET /models` `active` entry. */
  info(): ActiveModel {
    const recognizer = this.activeRecognizer();
    const templates: ActiveModel["templates"] = {};
    for (const documentType of KhmerOCRLocal.supportedDocuments()) {
      const { template } = this.extractor(documentType);
      templates[documentType] = { template_id: template.templateId,
        template_version: template.templateVersion, verified: template.verified };
    }
    return { sdk_version: VERSION, model: manifestSummary(this.manifest),
             recognizer: recognizer.describe(), templates };
  }

  /** Release ONNX Runtime sessions and tesseract.js workers. */
  async close(): Promise<void> {
    const recognizer = this.recognizer;
    this.recognizer = null;
    await recognizer?.close();
  }

  private activeRecognizer(): Recognizer {
    if (this.recognizer === null) throw ocrError("MODEL_NOT_AVAILABLE", "The engine has been closed.");
    return this.recognizer;
  }

  private extractor(documentType: string): TemplateExtractor {
    const recognizer = this.activeRecognizer();
    let extractor = this.extractors.get(documentType);
    if (extractor === undefined) {
      extractor = new TemplateExtractor(recognizer, this.config, templateForDocument(documentType));
      this.extractors.set(documentType, extractor);
    }
    return extractor;
  }

  private decode(image: LocalImageInput): Promise<Raster> {
    return loadImage(image, { maxBytes: this.config.maxImageBytes, maxPixels: this.config.quality.maxPixels },
      this.platform);
  }

  private gate(image: Raster, purpose: "extraction" | "recognition"): QualityMeasurements {
    const quality = assessQuality(image, this.config.quality);
    if (quality.blockingIssues.length > 0 && this.config.rejectLowQuality) {
      throw ocrError("IMAGE_QUALITY_TOO_LOW",
        `The document image is not usable for reliable ${purpose}.`, { issues: quality.blockingIssues });
    }
    return quality;
  }

  private modelMetadata(): Record<string, unknown> {
    return {
      sdk_version: VERSION,
      engine: `local-${this.platform.name}`,
      recognizer_backend: this.activeRecognizer().describe()["backend"],
      model_id: this.manifest.modelId,
      model_version: this.manifest.modelVersion,
    };
  }

  /** One call at a time: ONNX Runtime sessions and Tesseract workers are not reentrant. */
  private serialize<T>(signal: AbortSignal | undefined, task: () => Promise<T>): Promise<T> {
    const run = async () => {
      if (signal?.aborted) throw ocrError("ABORTED", "The call was aborted.", undefined, signal.reason);
      try {
        return await task();
      } catch (error) {
        if (signal?.aborted) throw ocrError("ABORTED", "The call was aborted.", undefined, signal.reason);
        throw error;
      }
    };
    const result = this.queue.then(run, run);
    this.queue = result.catch(() => undefined);
    return result;
  }
}

async function guarded<T>(step: () => Promise<T>): Promise<T> {
  try {
    return await step();
  } catch (error) {
    if (error instanceof KhmerOCRError) throw error;
    if (error instanceof Error && error.name === "AbortError") throw error;
    // The original message is not included: it may echo document content.
    throw ocrError("INFERENCE_FAILED", "Document processing failed.", undefined, error);
  }
}

function documentResult(extraction: Extraction, includeTextRegions: boolean,
                        metadata: Record<string, unknown>): DocumentResult {
  const result: DocumentResult = {
    schema_version: SCHEMA_VERSION,
    document_type: extraction.documentType,
    status: extraction.status,
    fields: Object.fromEntries(Object.entries(extraction.fields).map(([name, f]) =>
      [name, { value: f.value, confidence: roundConfidence(f.confidence), source: f.source, valid: f.valid }])),
    warnings: [...extraction.warnings],
    review_required: extraction.reviewRequired,
    metadata: { ...extraction.metadata, ...metadata },
  };
  if (includeTextRegions) {
    result.text_regions = extraction.regions.map((r) => ({
      text: r.text, confidence: roundConfidence(r.confidence), bbox: r.bbox, label: r.label ?? null }));
  }
  return result;
}

function qualitySummary(q: QualityMeasurements): Record<string, unknown> {
  return {
    usable: q.blockingIssues.length === 0,
    issues: [...q.issues],
    blur_score: Math.round(q.blurScore * 100) / 100,
    glare_ratio: Math.round(q.glareRatio * 1e4) / 1e4,
  };
}

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
const elapsed = (started: number) => Math.trunc(now() - started);
