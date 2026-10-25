/**
 * Result types. They are the REST API's JSON contract (`schemas/` at the
 * repository root), re-exported from `@khmerocr/client` so code written for
 * either package handles both.
 */
import type {
  BoundingBox,
  DocumentResult,
  ExtractedField,
  OcrResult,
  QualityReport,
  ResultStatus,
  TextRegion,
} from "@khmerocr/client";

export type {
  BoundingBox,
  DocumentResult,
  ExtractedField,
  OcrResult,
  QualityReport,
  ResultStatus,
  TextRegion,
};

export const SCHEMA_VERSION = "1.0" as const;

/** Python `round(value, 4)` for confidences in results. */
export function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

export function roundConfidence(value: number | null): number | null {
  return value === null ? null : round(value, 4);
}

/** Quality measurements before rounding (the Python `QualityReport`). */
export interface QualityMeasurements {
  width: number;
  height: number;
  blurScore: number;
  glareRatio: number;
  brightness: number;
  contrast: number;
  issues: string[];
  blockingIssues: string[];
}

export function qualityReport(q: QualityMeasurements): QualityReport {
  return {
    schema_version: SCHEMA_VERSION,
    usable: q.blockingIssues.length === 0,
    width: q.width,
    height: q.height,
    blur_score: round(q.blurScore, 2),
    glare_ratio: round(q.glareRatio, 4),
    brightness: round(q.brightness, 2),
    contrast: round(q.contrast, 2),
    issues: [...q.issues],
    blocking_issues: [...q.blockingIssues],
  };
}
