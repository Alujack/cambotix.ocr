/** Unicode and script normalization for Khmer and Latin text. */

export const KHMER_DIGITS = "០១២៣៤៥៦៧៨៩";

// Khmer (U+1780-U+17FF) and Khmer Symbols (U+19E0-U+19FF).
const KHMER_CHAR = /[ក-៿᧠-᧿]/u;
const ZERO_WIDTH = /[​‌‍⁠﻿]/gu;
const WHITESPACE = /\s+/gu;

/** Convert Khmer digits to ASCII digits. */
export function toAsciiDigits(text: string): string {
  return text.replace(/[០-៩]/gu, (d) => String(KHMER_DIGITS.indexOf(d)));
}

/**
 * NFC-normalize, drop zero-width characters, and collapse whitespace.
 *
 * Zero-width spaces are commonly inserted between Khmer words by input
 * methods and OCR engines; they carry no meaning for field values.
 */
export function normalizeText(text: string): string {
  return text.normalize("NFC").replace(ZERO_WIDTH, "").replace(WHITESPACE, " ").trim();
}

export function containsKhmer(text: string): boolean {
  return KHMER_CHAR.test(text);
}

/** Python `str.rstrip(chars)` / `str.lstrip(chars)`. */
export function rstripChars(text: string, chars: string): string {
  const cps = Array.from(text);
  while (cps.length > 0 && chars.includes(cps[cps.length - 1]!)) cps.pop();
  return cps.join("");
}

export function lstripChars(text: string, chars: string): string {
  const cps = Array.from(text);
  let i = 0;
  while (i < cps.length && chars.includes(cps[i]!)) i++;
  return cps.slice(i).join("");
}

/**
 * Remove a leading printed label (and a following separator) from `text`.
 *
 * Labels are compared after removing spaces, since recognizers frequently
 * insert or drop spaces inside Khmer label text.
 */
export function stripLabels(text: string, labels: readonly string[]): string {
  const compact = text.replaceAll(" ", "");
  const sorted = [...labels].sort((a, b) => Array.from(b).length - Array.from(a).length);
  for (const label of sorted) {
    const labelCompact = normalizeText(label).replaceAll(" ", "");
    // Text that is only (part of) the printed label, e.g. when glare hides
    // the value, must not be returned as the value itself.
    if (compact && labelCompact.startsWith(rstripChars(compact, ":：-ៈ"))) return "";
    if (labelCompact && compact.startsWith(labelCompact)) {
      // Walk the original string until the label's characters are consumed.
      const chars = Array.from(text);
      const labelLength = Array.from(labelCompact).length;
      let consumed = 0;
      let i = 0;
      while (i < chars.length && consumed < labelLength) {
        if (chars[i] !== " ") consumed++;
        i++;
      }
      // "ៈ" (U+17C8) is how Khmer recognizers often read a printed ":".
      return lstripChars(chars.slice(i).join(""), " :：-ៈ").trim();
    }
  }
  return text;
}
