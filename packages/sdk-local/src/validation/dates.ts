/**
 * Date parsing for document fields.
 *
 * Only day-first numeric dates are accepted (`DD.MM.YYYY` with `.`, `/`,
 * `-` or space separators, in ASCII or Khmer digits). The date convention
 * printed on the card must be confirmed against authorized reference material;
 * until then, parsed dates are only as trustworthy as that assumption.
 */
import { toAsciiDigits } from "./text.js";

const DATE = /(\d{1,2})\s*[./\- ]\s*(\d{1,2})\s*[./\- ]\s*(\d{4})/u;
export const MAX_AGE_YEARS = 130;
export const EARLIEST_ISSUE_YEAR = 1950;

// Gregorian month names as printed on Cambodian documents. Lunar-calendar
// months (e.g. ខែផល្គុន) are deliberately not mapped.
export const KHMER_MONTHS: Readonly<Record<string, number>> = {
  "មករា": 1, "កុម្ភៈ": 2, "កុម្ភ": 2, "មីនា": 3, "មេសា": 4, "ឧសភា": 5, "មិថុនា": 6,
  "កក្កដា": 7, "សីហា": 8, "កញ្ញា": 9, "តុលា": 10, "វិច្ឆិកា": 11, "ធ្នូ": 12,
};
const KHMER_MONTH_DATE = new RegExp(
  "(\\d{1,2})ខែ(" + Object.keys(KHMER_MONTHS).sort((a, b) => b.length - a.length).join("|")
  + ")ឆ្នាំ(\\d{4})", "u");

/** A calendar date without a time zone. */
export interface CalendarDate {
  year: number;
  month: number;
  day: number;
}

/** Python `date(year, month, day)`: null when the date does not exist. */
export function makeDate(year: number, month: number, day: number): CalendarDate | null {
  if (year < 1 || year > 9999 || month < 1 || month > 12 || day < 1) return null;
  const daysInMonth = new Date(Date.UTC(2000, month, 0)).getUTCDate();
  const leap = (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
  const limit = month === 2 ? (leap ? 29 : 28) : daysInMonth;
  return day <= limit ? { year, month, day } : null;
}

export function isoDate(d: CalendarDate): string {
  const pad = (n: number, width: number) => String(n).padStart(width, "0");
  return `${pad(d.year, 4)}-${pad(d.month, 2)}-${pad(d.day, 2)}`;
}

export function today(): CalendarDate {
  const now = new Date();
  return { year: now.getFullYear(), month: now.getMonth() + 1, day: now.getDate() };
}

function compare(a: CalendarDate, b: CalendarDate): number {
  return a.year - b.year || a.month - b.month || a.day - b.day;
}

export function parseDayFirstDate(text: string): CalendarDate | null {
  const match = DATE.exec(toAsciiDigits(text));
  if (!match) return null;
  return makeDate(Number(match[3]), Number(match[2]), Number(match[1]));
}

/** Parse dates such as `ថ្ងៃទី០១ ខែមីនា ឆ្នាំ២០២៣` (1 March 2023). */
export function parseKhmerMonthDate(text: string): CalendarDate | null {
  const compact = toAsciiDigits(text).replace(/[\s​]/gu, "").replaceAll("ថ្ងៃទី", "");
  const match = KHMER_MONTH_DATE.exec(compact);
  if (!match) return null;
  return makeDate(Number(match[3]), KHMER_MONTHS[match[2]!]!, Number(match[1]));
}

export function isPlausibleIssueDate(value: CalendarDate, now: CalendarDate = today()): boolean {
  return EARLIEST_ISSUE_YEAR <= value.year && compare(value, now) <= 0;
}

export function isPlausibleBirthDate(value: CalendarDate, now: CalendarDate = today()): boolean {
  if (compare(value, now) > 0) return false;
  return now.year - value.year <= MAX_AGE_YEARS;
}
