/** Versioned document layout templates. */
import { ocrError } from "../errors.js";
import type { RegionHint, Script } from "../recognition/base.js";
import type { FieldOptions } from "../validation/fields.js";
import { BUILTIN_TEMPLATES } from "./templates.js";

const SCRIPTS: readonly Script[] = ["khmer", "latin", "digits", "mixed"];

/** A template as stored in JSON (the Python engine's format). */
export interface TemplateData {
  template_id: string;
  template_version: string;
  document_type: string;
  verified: boolean;
  canonical_size: [number, number];
  fields: FieldData[];
  padding?: number | null;
  [key: string]: unknown;
}

export interface FieldData {
  name: string;
  type: string;
  box: number[];
  script?: string;
  critical?: boolean;
  source?: string;
  labels?: string[];
  languages?: string | null;
  char_whitelist?: string | null;
  pattern?: string | null;
  separator?: string | null;
  part?: number | null;
}

export interface FieldSpec {
  name: string;
  type: string;
  /** Normalized (x, y, width, height) relative to the canonical card size. */
  box: [number, number, number, number];
  script: Script;
  critical: boolean;
  source: string;
  labels: string[];
  languages: string | null;
  charWhitelist: string | null;
  pattern: string | null;
  /** For bilingual lines such as "ប្រុស / Male": split on `separator`, keep part `part`. */
  separator: string | null;
  part: number | null;
}

export interface DocumentTemplate {
  templateId: string;
  templateVersion: string;
  documentType: string;
  verified: boolean;
  /** Width and height of the rectified card. */
  canonicalSize: [number, number];
  fields: FieldSpec[];
  /** Region padding override (fraction of card size) for dense layouts. */
  padding: number | null;
}

export function fieldHint(spec: FieldSpec): RegionHint {
  return { name: spec.name, script: spec.script, languages: spec.languages,
           charWhitelist: spec.charWhitelist };
}

export function fieldOptions(spec: FieldSpec): FieldOptions {
  return spec.pattern ? { pattern: spec.pattern } : {};
}

export function criticalFields(template: DocumentTemplate): string[] {
  return template.fields.filter((f) => f.critical).map((f) => f.name);
}

export function parseTemplate(data: TemplateData): DocumentTemplate {
  try {
    const [width, height] = data.canonical_size;
    if (!Number.isFinite(width) || !Number.isFinite(height)) throw new Error("canonical_size");
    return {
      templateId: String(data.template_id),
      templateVersion: String(data.template_version),
      documentType: String(data.document_type),
      verified: Boolean(data.verified),
      canonicalSize: [Math.trunc(width), Math.trunc(height)],
      fields: data.fields.map(parseField),
      padding: data.padding == null ? null : Number(data.padding),
    };
  } catch (cause) {
    throw ocrError("UNSUPPORTED_DOCUMENT", "The document template is malformed.", undefined, cause);
  }
}

function parseField(data: FieldData): FieldSpec {
  const box = data.box.map(Number);
  if (box.length !== 4 || !box.every((v) => v >= 0 && v <= 1)) {
    throw new Error("box must contain four values in [0, 1]");
  }
  const script = (data.script ?? "mixed") as Script;
  if (!SCRIPTS.includes(script)) throw new Error(`unknown script ${script}`);
  return {
    name: String(data.name),
    type: String(data.type),
    box: [box[0]!, box[1]!, box[2]!, box[3]!],
    script,
    critical: Boolean(data.critical ?? false),
    source: String(data.source ?? "front_region"),
    labels: [...(data.labels ?? [])],
    languages: data.languages ?? null,
    charWhitelist: data.char_whitelist ?? null,
    pattern: data.pattern ?? null,
    separator: data.separator ?? null,
    part: data.part == null ? null : Math.trunc(data.part),
  };
}

/** Default layout template for each supported document type. */
export const DOCUMENT_TEMPLATES: Readonly<Record<string, string>> = {
  khmer_national_id: "khmer_national_id_front_v1",
  student_id_card: "student_id_card_front_v1",
};

export function templateForDocument(documentType: string): DocumentTemplate {
  const templateId = DOCUMENT_TEMPLATES[documentType];
  if (templateId === undefined) {
    throw ocrError("UNSUPPORTED_DOCUMENT", `Unsupported document type '${documentType}'.`,
      { supported: Object.keys(DOCUMENT_TEMPLATES).sort() });
  }
  return loadBuiltinTemplate(templateId);
}

export function loadBuiltinTemplate(templateId: string): DocumentTemplate {
  const data = BUILTIN_TEMPLATES[templateId];
  if (data === undefined) {
    throw ocrError("UNSUPPORTED_DOCUMENT", `Unknown document template '${templateId}'.`);
  }
  return parseTemplate(data);
}
