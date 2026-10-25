/**
 * Field-type normalizers.
 *
 * Each normalizer turns cleaned recognizer output into a canonical value and
 * reports whether it passed format validation. Normalizers never invent data:
 * if the text cannot be interpreted, the value is null or the result is
 * marked invalid so the field is sent to human review.
 */
import {
  isoDate,
  isPlausibleBirthDate,
  isPlausibleIssueDate,
  parseDayFirstDate,
  parseKhmerMonthDate,
} from "./dates.js";
import { containsKhmer, normalizeText, toAsciiDigits } from "./text.js";

// Characters commonly confused with digits by OCR engines.
const DIGIT_CONFUSIONS: Readonly<Record<string, string>> = {
  O: "0", o: "0", D: "0", I: "1", l: "1", "|": "1", S: "5", B: "8", Z: "2",
};
const LATIN_NAME = /^[A-Z][A-Z' -]*[A-Z]$/u;
const NON_KHMER_NAME = /[^ក-៿᧠-᧿ ]/gu;
// Never in a correctly read name: Khmer digits, punctuation or symbols, two
// vowel signs in a row, or a word that starts with a vowel sign or mark.
const IMPOSSIBLE_KHMER_NAME = /[។-៿᧠-᧿]|[ា-ៅ]{2}|(?:^| )[ា-៓]/u;

export interface Normalized {
  value: string | null;
  valid: boolean;
  /** Warning codes (without the field suffix), e.g. "ocr_corrected". */
  warnings: string[];
}

export interface FieldOptions {
  pattern?: string;
}

export type Normalizer = (text: string, options: FieldOptions) => Normalized;

const result = (value: string | null, valid: boolean, warnings: string[] = []): Normalized =>
  ({ value, valid, warnings });

function fullMatch(pattern: string, text: string): boolean {
  return new RegExp(`^(?:${pattern})$`, "u").test(text);
}

export function normalizeIdNumber(text: string, options: FieldOptions): Normalized {
  const digitsOnly = toAsciiDigits(text).replace(/[\s\-./]/gu, "");
  const warnings: string[] = [];
  const corrected = Array.from(digitsOnly).map((ch) => DIGIT_CONFUSIONS[ch] ?? ch).join("");
  if (corrected !== digitsOnly) warnings.push("ocr_corrected");
  if (!corrected) return result(null, false, warnings);
  const valid = fullMatch(options.pattern ?? "^[0-9]+$", corrected);
  if (!valid) warnings.push("invalid_format");
  return result(corrected, valid, warnings);
}

export function normalizeKhmerName(text: string): Normalized {
  if (!containsKhmer(text)) return result(null, false, text ? ["invalid_script"] : []);
  const cleaned = normalizeText(text.replace(NON_KHMER_NAME, " "));
  const warnings = cleaned === text ? [] : ["non_khmer_characters_removed"];
  if (cleaned && IMPOSSIBLE_KHMER_NAME.test(cleaned)) {
    return result(cleaned, false, [...warnings, "invalid_format"]);
  }
  return result(cleaned || null, Boolean(cleaned), warnings);
}

export function normalizeLatinName(text: string): Normalized {
  const cleaned = normalizeText(text.toUpperCase());
  if (!cleaned) return result(null, false);
  const valid = LATIN_NAME.test(cleaned);
  return result(cleaned, valid, valid ? [] : ["invalid_format"]);
}

export function normalizeBirthDate(text: string): Normalized {
  const parsed = parseDayFirstDate(text);
  if (parsed === null) return result(null, false, text ? ["invalid_format"] : []);
  if (!isPlausibleBirthDate(parsed)) return result(isoDate(parsed), false, ["implausible_date"]);
  return result(isoDate(parsed), true);
}

export function normalizeIssueDate(text: string): Normalized {
  const parsed = parseKhmerMonthDate(text) ?? parseDayFirstDate(text);
  if (parsed === null) return result(null, false, text ? ["invalid_format"] : []);
  if (!isPlausibleIssueDate(parsed)) return result(isoDate(parsed), false, ["implausible_date"]);
  return result(isoDate(parsed), true);
}

export function normalizeGender(text: string): Normalized {
  // Bilingual cards print both scripts ("ប្រុស / Male"); every reading found
  // must agree, otherwise the value is ambiguous.
  const compact = text.replaceAll(" ", "");
  const tokens = new Set(text.toUpperCase().match(/[A-Za-z]+/gu) ?? []);
  const readings = new Set<string>();
  if (compact.includes("ប្រុស") || tokens.has("M") || tokens.has("MALE")) readings.add("M");
  if (compact.includes("ស្រី") || tokens.has("F") || tokens.has("FEMALE")) readings.add("F");
  if (readings.size !== 1) return result(null, false, text ? ["ambiguous_value"] : []);
  return result([...readings][0]!, true);
}

export function normalizeTextField(text: string): Normalized {
  return result(text || null, Boolean(text));
}

export const NORMALIZERS: Readonly<Record<string, Normalizer>> = {
  id_number: normalizeIdNumber,
  khmer_name: normalizeKhmerName,
  latin_name: normalizeLatinName,
  birth_date: normalizeBirthDate,
  issue_date: normalizeIssueDate,
  gender: normalizeGender,
  text: normalizeTextField,
};
