/**
 * Template-driven field extraction for card documents.
 *
 * Fields are mapped deterministically: the card is rectified to a canonical
 * size and each field is read from a region defined by a versioned layout
 * template. No language model is used to guess or complete identity fields.
 */
import type { EngineConfig } from "../config.js";
import { ocrError } from "../errors.js";
import { crop, rotateQuarterTurns, type Raster } from "../image/raster.js";
import { normalizeCard } from "../preprocessing/geometry.js";
import type { RecognizedText, Recognizer } from "../recognition/base.js";
import type { BoundingBox, QualityMeasurements, ResultStatus, TextRegion } from "../schemas.js";
import { NORMALIZERS, type Normalized } from "../validation/fields.js";
import { normalizeText, stripLabels } from "../validation/text.js";
import {
  criticalFields,
  fieldHint,
  fieldOptions,
  type DocumentTemplate,
  type FieldSpec,
} from "./layout.js";

// Quality issues that make field positions or pixels unreliable.
const REVIEW_QUALITY_ISSUES = new Set(["blurry", "glare_detected"]);

export interface Field {
  value: string | null;
  confidence: number | null;
  source: string;
  valid: boolean;
  rawText: string | null;
}

export interface Extraction {
  documentType: string;
  fields: Record<string, Field>;
  reviewRequired: boolean;
  warnings: string[];
  status: ResultStatus;
  regions: TextRegion[];
  metadata: Record<string, unknown>;
}

interface Pass {
  quarterTurns: number;
  fields: Record<string, Field>;
  regions: TextRegion[];
  warnings: string[];
  needsReview: Set<string>;
}

/** (number of valid critical fields, mean confidence over the critical fields). */
function score(pass: Pass, critical: readonly string[]): [number, number] {
  const valid = critical.map((name) => pass.fields[name]).filter((f): f is Field => f?.valid === true);
  const confidence = valid.reduce((sum, f) => sum + (f.confidence ?? 0), 0) / Math.max(1, critical.length);
  return [valid.length, confidence];
}

const better = (a: [number, number], b: [number, number]) => a[0] > b[0] || (a[0] === b[0] && a[1] > b[1]);

export class TemplateExtractor {
  constructor(readonly recognizer: Recognizer, readonly config: EngineConfig,
              readonly template: DocumentTemplate) {
    const unknown = [...new Set(template.fields.map((f) => f.type))].filter((t) => !(t in NORMALIZERS)).sort();
    if (unknown.length > 0) {
      throw ocrError("UNSUPPORTED_DOCUMENT", "The template uses unknown field types.", { field_types: unknown });
    }
  }

  async extract(image: Raster, quality: QualityMeasurements, signal?: AbortSignal): Promise<Extraction> {
    const template = this.template;
    const critical = criticalFields(template);
    const card = normalizeCard(image, template.canonicalSize);

    const turnsToTry = this.config.tryAlternateOrientations ? [0, 2] : [0];
    let best: Pass | null = null;
    for (const extraTurns of turnsToTry) {
      const attempt = await this.readFields(rotateQuarterTurns(card.image, extraTurns),
        (card.quarterTurns + extraTurns) % 4, signal);
      if (best === null || better(score(attempt, critical), score(best, critical))) best = attempt;
      // Any field needing review triggers the 180-degree pass. Cheaper
      // heuristics fail because upside-down garbage can pass weak format checks.
      if (attempt.needsReview.size === 0) break;
    }
    const chosen = best!;

    const warnings: string[] = [];
    let review = chosen.needsReview.size > 0;
    const qualityFlags = quality.issues.filter((issue) => REVIEW_QUALITY_ISSUES.has(issue));
    if (qualityFlags.length > 0) {
      warnings.push(...qualityFlags);
      review = true;
    }
    if (!card.detected) {
      warnings.push("card_boundary_not_detected");
      review = true;
    }
    if (chosen.quarterTurns) warnings.push("orientation_corrected");
    if (!template.verified) {
      warnings.push("layout_template_unverified");
      review = true;
    }
    warnings.push(...chosen.warnings);

    let status: ResultStatus;
    if (Object.values(chosen.fields).every((f) => f.value === null)) {
      status = "failed";
      review = true;
    } else {
      status = review ? "review_required" : "success";
    }

    return {
      documentType: template.documentType,
      fields: chosen.fields,
      reviewRequired: review,
      warnings: [...new Set(warnings)],
      status,
      regions: chosen.regions,
      metadata: {
        template_id: template.templateId,
        template_version: template.templateVersion,
        template_verified: template.verified,
        card_detected: card.detected,
        quarter_turns: chosen.quarterTurns,
        critical_fields: critical,
        review_confidence_threshold: this.config.reviewConfidenceThreshold,
        coordinate_space: { width: template.canonicalSize[0], height: template.canonicalSize[1] },
      },
    };
  }

  private async readFields(card: Raster, quarterTurns: number, signal?: AbortSignal): Promise<Pass> {
    const result: Pass = { quarterTurns, fields: {}, regions: [], warnings: [], needsReview: new Set() };
    const padding = this.template.padding ?? this.config.regionPadding;
    // Fields that share a region (e.g. "Khmer / English" on one line) reuse the same reads.
    const readsByRegion = new Map<string, RegionReads>();
    for (const spec of this.template.fields) {
      signal?.throwIfAborted();
      const key = JSON.stringify([spec.box, spec.script, spec.languages, spec.charWhitelist]);
      const bbox = pixelBox(spec, card.width, card.height, padding);
      let reads = readsByRegion.get(key);
      const firstUse = reads === undefined;
      if (reads === undefined) {
        const region = crop(card, bbox.x, bbox.y, bbox.width, bbox.height);
        reads = new RegionReads(this.recognizer.recognizeCandidates(region, fieldHint(spec)));
        readsByRegion.set(key, reads);
      }
      const [field, recognized] = await this.mapField(spec, reads, result);
      result.fields[spec.name] = field;
      if (firstUse) {
        result.regions.push({ text: normalizeText(recognized.text), confidence: recognized.confidence,
                              bbox, label: spec.name });
      }
    }
    return result;
  }

  /**
   * Map the first candidate read whose value validates; if none does, fall
   * back to the primary read so its problems are reported.
   */
  private async mapField(spec: FieldSpec, reads: RegionReads, result: Pass): Promise<[Field, RecognizedText]> {
    const name = spec.name;
    type Choice = [number, RecognizedText, string, Normalized | null];
    let chosen: Choice | null = null;
    let primary: Choice | null = null;
    for (let index = 0; ; index++) {
      const recognized = await reads.get(index);
      if (recognized === null) break;
      const raw = selectPart(normalizeText(recognized.text), spec);
      const normalized = raw ? NORMALIZERS[spec.type]!(stripLabels(raw, spec.labels), fieldOptions(spec)) : null;
      primary ??= [index, recognized, raw, normalized];
      if (normalized !== null && normalized.valid) {
        chosen = [index, recognized, raw, normalized];
        break;
      }
    }
    const [used, recognized, raw, maybeNormalized] = chosen ?? primary!;
    if (used) result.warnings.push(`fallback_recognizer:${name}`);
    const confidence = recognized.confidence;
    if (!raw || maybeNormalized === null) {
      result.warnings.push(`field_not_detected:${name}`);
      if (spec.critical) result.needsReview.add(name);
      return [{ value: null, confidence: null, source: spec.source, valid: false, rawText: null }, recognized];
    }

    const normalized = maybeNormalized;
    result.warnings.push(...normalized.warnings.map((code) => `${code}:${name}`));
    if (normalized.value === null) result.warnings.push(`field_unreadable:${name}`);
    const lowConfidence = confidence === null || confidence < this.config.reviewConfidenceThreshold;
    if (lowConfidence) result.warnings.push(`low_confidence:${name}`);

    if (normalized.value === null) {
      if (spec.critical) result.needsReview.add(name);
    } else if (!normalized.valid || lowConfidence || normalized.warnings.includes("ocr_corrected")) {
      result.needsReview.add(name);
    }
    return [{ value: normalized.value, confidence, source: spec.source, valid: normalized.valid,
              rawText: raw }, recognized];
  }
}

/** Lazily evaluated candidate reads of one region, shared between fields. */
class RegionReads {
  private readonly reads: RecognizedText[] = [];
  private done = false;

  constructor(private readonly candidates: AsyncIterator<RecognizedText>) {}

  async get(index: number): Promise<RecognizedText | null> {
    while (this.reads.length <= index && !this.done) {
      const next = await this.candidates.next();
      if (next.done) this.done = true;
      else this.reads.push(next.value);
    }
    return this.reads[index] ?? null;
  }
}

/** Pick one side of a bilingual line such as `"ប្រុស / Male"`. */
export function selectPart(text: string, spec: FieldSpec): string {
  if (spec.separator === null || spec.part === null) return text;
  const parts = text.split(spec.separator);
  // Separator not recognized: return nothing rather than guess which side the text belongs to.
  if (parts.length < 2 || spec.part >= parts.length) return "";
  return parts[spec.part]!.trim();
}

export function pixelBox(spec: FieldSpec, width: number, height: number, padding: number): BoundingBox {
  const [x, y, w, h] = spec.box;
  const x0 = Math.max(0, Math.trunc((x - padding) * width));
  const y0 = Math.max(0, Math.trunc((y - padding) * height));
  const x1 = Math.min(width, Math.trunc((x + w + padding) * width));
  const y1 = Math.min(height, Math.trunc((y + h + padding) * height));
  return { x: x0, y: y0, width: Math.max(1, x1 - x0), height: Math.max(1, y1 - y0) };
}
