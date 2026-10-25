import type { DocumentResult } from "./types.js";

export interface ParsedWarning {
  code: string;
  /** The field a `<code>:<field>` warning refers to, or null for document-level warnings. */
  field: string | null;
}

/** Split warnings such as `low_confidence:address_km` into code and field. */
export function parseWarnings(result: Pick<DocumentResult, "warnings">): ParsedWarning[] {
  return result.warnings.map((warning) => {
    const separator = warning.indexOf(":");
    return separator === -1
      ? { code: warning, field: null }
      : { code: warning.slice(0, separator), field: warning.slice(separator + 1) };
  });
}

/**
 * Fields a person should look at first: those named in a warning, those that
 * failed validation, and those with no value. `review_required` remains the
 * signal for whether the whole result needs confirming.
 */
export function fieldsNeedingReview<F extends string>(result: DocumentResult<F>): F[] {
  const names = new Set<string>();
  for (const { field } of parseWarnings(result)) {
    if (field !== null) names.add(field);
  }
  for (const [name, field] of Object.entries(result.fields) as [string, DocumentResult["fields"][string]][]) {
    if (field !== undefined && (field.valid === false || field.value === null)) names.add(name);
  }
  return [...names] as F[];
}
