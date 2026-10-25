/**
 * Response types for the KhmerOCR `/v1` REST API. They mirror the JSON
 * Schemas in `schemas/` exactly, including their snake_case keys.
 */

export type ResultStatus = "success" | "review_required" | "failed";

export const NATIONAL_ID_FIELDS = [
  "id_number",
  "full_name_km",
  "full_name_latin",
  "date_of_birth",
  "gender",
] as const;
export type NationalIdField = (typeof NATIONAL_ID_FIELDS)[number];

export const STUDENT_ID_FIELDS = [
  "full_name_km",
  "full_name_latin",
  "gender",
  "date_of_birth",
  "department",
  "degree",
  "address_km",
  "student_id",
  "issue_date",
] as const;
export type StudentIdField = (typeof STUDENT_ID_FIELDS)[number];

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface TextRegion {
  text: string;
  /** Recognition confidence in [0, 1], or null when the backend reports none. */
  confidence: number | null;
  bbox: BoundingBox;
  label?: string | null;
}

export interface ExtractedField {
  /** Normalized value (ISO-8601 dates, uppercase Latin names, `M`/`F`), or null. */
  value: string | null;
  /**
   * Confidence in the text read for this field's region. It does not measure
   * whether the region was mapped to the right field.
   */
  confidence: number | null;
  source: string;
  /** Whether the value passed format validation for its field type. */
  valid?: boolean | null;
}

export interface DocumentResult<F extends string = string> {
  schema_version: "1.0";
  document_type: string;
  status: ResultStatus;
  fields: Partial<Record<F, ExtractedField>>;
  /** Document-level codes, or `<code>:<field>`. See `parseWarnings`. */
  warnings: string[];
  /** When true, a person must confirm the result before it is used. */
  review_required: boolean;
  metadata?: Record<string, unknown>;
  /** Present when `includeTextRegions` was requested. */
  text_regions?: TextRegion[];
}

export type NationalIdResult = DocumentResult<NationalIdField>;
export type StudentIdResult = DocumentResult<StudentIdField>;

export interface OcrResult {
  schema_version: "1.0";
  document_type: "general_text";
  status: ResultStatus;
  text: string;
  regions: TextRegion[];
  warnings: string[];
  review_required: boolean;
  metadata?: Record<string, unknown>;
}

export interface QualityReport {
  schema_version: "1.0";
  usable: boolean;
  width: number;
  height: number;
  blur_score: number;
  glare_ratio: number;
  brightness: number;
  contrast: number;
  issues: string[];
  blocking_issues: string[];
}

export interface HealthStatus {
  status: "ok";
  model_loaded: boolean;
  api_version: string;
  sdk_version: string;
  /** Error code when the model failed to load, e.g. `MODEL_NOT_AVAILABLE`. */
  model_error?: string;
}

export interface ModelSummary {
  model_id: string;
  model_version: string;
  task: string;
  runtime: string;
  recognition_model: string;
  schema_version: string;
  license: string;
  license_approved: boolean;
  recognizer_backend: string;
  weights_sha256: string;
}

export interface TemplateInfo {
  template_id: string;
  template_version: string;
  verified: boolean;
}

export interface ActiveModel {
  sdk_version: string;
  model: ModelSummary | null;
  recognizer: Record<string, string>;
  templates: Record<string, TemplateInfo>;
}

export interface ModelsInfo {
  /** The model serving requests, or null when none is loaded. */
  active: ActiveModel | null;
  installed: ModelSummary[];
}
