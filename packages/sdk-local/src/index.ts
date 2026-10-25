/**
 * Platform-independent exports. Load the engine from `@khmerocr/local/browser`
 * or `@khmerocr/local/node`; both also re-export everything here.
 */
export { KhmerOCRLocal } from "./engine.js";
export type { CallOptions, LoadOptions, LocalExtractOptions, NationalIdResult, StudentIdResult } from "./engine.js";
export { DEFAULT_QUALITY, resolveConfig } from "./config.js";
export type { EngineConfig, EngineConfigOverrides, QualityThresholds } from "./config.js";
export { KhmerOCRError } from "./errors.js";
export type { LocalErrorCode } from "./errors.js";
export type { LocalImageInput, RawImage } from "./image/input.js";
export { fieldsNeedingReview, parseWarnings, NATIONAL_ID_FIELDS, STUDENT_ID_FIELDS } from "@khmerocr/client";
export type {
  ActiveModel,
  BoundingBox,
  DocumentResult,
  ExtractedField,
  ModelSummary,
  NationalIdField,
  OcrResult,
  ParsedWarning,
  QualityReport,
  ResultStatus,
  StudentIdField,
  TextRegion,
} from "@khmerocr/client";
export { VERSION } from "./version.js";
