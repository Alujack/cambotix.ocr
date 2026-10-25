// Mirrors packages/core-python/tests/test_validation.py and the date and
// gender cases in test_documents.py.
import { describe, expect, it } from "vitest";
import {
  isPlausibleBirthDate, makeDate, parseDayFirstDate, parseKhmerMonthDate,
} from "../src/validation/dates.js";
import {
  NORMALIZERS, normalizeBirthDate, normalizeGender, normalizeIdNumber, normalizeIssueDate,
  normalizeKhmerName, normalizeLatinName,
} from "../src/validation/fields.js";
import { normalizeText, stripLabels, toAsciiDigits } from "../src/validation/text.js";

const ID_SPEC = { pattern: "^[0-9]{9}$" };

describe("text", () => {
  it("converts Khmer digits", () => {
    expect(toAsciiDigits("០១២៣៤៥៦៧៨៩")).toBe("0123456789");
  });

  it("removes zero-width characters and collapses spaces", () => {
    expect(normalizeText("  គំរូ​  សាកល្បង \n")).toBe("គំរូ សាកល្បង");
  });

  it("strips labels with spacing inside them", () => {
    expect(stripLabels("គោត្តនាម និងនាម : គំរូ", ["គោត្តនាមនិងនាម"])).toBe("គំរូ");
    expect(stripLabels("NAME: SOK", ["NAME"])).toBe("SOK");
    expect(stripLabels("SOK", ["NAME"])).toBe("SOK");
  });

  it("never returns a truncated label as the value", () => {
    const labels = ["គោត្តនាមនិងនាម", "ឈ្មោះ"];
    expect(stripLabels("គោត្តនាមនិងន", labels)).toBe("");
    expect(stripLabels("គោត្តនាមនិងនាមៈ", labels)).toBe("");
    expect(stripLabels("គោត្តនាមនិងនាមៈគំរូ សាកល្បង", labels)).toBe("គំរូ សាកល្បង");
  });
});

describe("dates", () => {
  it("parses day-first dates", () => {
    expect(parseDayFirstDate("01.02.1990")).toEqual(makeDate(1990, 2, 1));
    expect(parseDayFirstDate("៣១/១២/២០០០")).toEqual(makeDate(2000, 12, 31));
    expect(parseDayFirstDate("31.02.1990")).toBeNull();
    expect(parseDayFirstDate("29.02.2000")).toEqual(makeDate(2000, 2, 29));
    expect(parseDayFirstDate("29.02.1900")).toBeNull();
    expect(parseDayFirstDate("no date")).toBeNull();
  });

  it("checks birth date plausibility", () => {
    const today = makeDate(2026, 1, 1)!;
    expect(isPlausibleBirthDate(makeDate(1990, 1, 1)!, today)).toBe(true);
    expect(isPlausibleBirthDate(makeDate(2027, 1, 1)!, today)).toBe(false);
    expect(isPlausibleBirthDate(makeDate(1800, 1, 1)!, today)).toBe(false);
  });

  it("parses Khmer month names", () => {
    expect(parseKhmerMonthDate("ថ្ងៃទី០១ ខែមីនា ឆ្នាំ២០២៣")).toEqual(makeDate(2023, 3, 1));
    expect(parseKhmerMonthDate("ថ្ងៃទី ៣១ ខែ ធ្នូ ឆ្នាំ ១៩៩៩")).toEqual(makeDate(1999, 12, 31));
    // Lunar-calendar months are not converted.
    expect(parseKhmerMonthDate("១០កើត ខែផល្គុន ឆ្នាំខាល")).toBeNull();
    expect(parseKhmerMonthDate("ថ្ងៃទី៣០ ខែកុម្ភៈ ឆ្នាំ២០២៣")).toBeNull();
  });
});

describe("field normalizers", () => {
  it("validates ID numbers", () => {
    expect(normalizeIdNumber("០០០ ០០០ ០០១", ID_SPEC)).toEqual({ value: "000000001", valid: true, warnings: [] });
    const short = normalizeIdNumber("12345", ID_SPEC);
    expect(short.valid).toBe(false);
    expect(short.warnings).toContain("invalid_format");
  });

  it("flags digits corrected from look-alike letters", () => {
    const result = normalizeIdNumber("O0000000I", ID_SPEC);
    expect(result.value).toBe("000000001");
    expect(result.warnings).toContain("ocr_corrected");
  });

  it("requires Khmer script in Khmer names", () => {
    expect(normalizeKhmerName("គំរូ សាកល្បង").valid).toBe(true);
    const latin = normalizeKhmerName("SOK");
    expect(latin.value).toBeNull();
    expect(latin.warnings).toContain("invalid_script");
  });

  it.each(["សៀុខ ៥ៅក", "សុខ ១២", "សុខ។", "ាសុខ"])("rejects impossible Khmer names (%s)", (text) => {
    // Shaped like a real misread of a bold printed name: a digit and two
    // vowel signs in a row.
    const result = normalizeKhmerName(text);
    expect(result.value).toBe(text);
    expect(result.valid).toBe(false);
    expect(result.warnings).toContain("invalid_format");
  });

  it("accepts Khmer names with a register shifter", () => {
    expect(normalizeKhmerName("សុខ ម៉ៅ").valid).toBe(true);
  });

  it("uppercases and validates Latin names", () => {
    expect(normalizeLatinName("kumru sakalbang").value).toBe("KUMRU SAKALBANG");
    expect(normalizeLatinName("SOK 123").valid).toBe(false);
  });

  it("normalizes birth dates to ISO", () => {
    expect(normalizeBirthDate("01.01.1990")).toMatchObject({ value: "1990-01-01", valid: true });
    const future = normalizeBirthDate("01.01.2999");
    expect(future.valid).toBe(false);
    expect(future.warnings).toContain("implausible_date");
  });

  it("normalizes issue dates", () => {
    expect(normalizeIssueDate("ថ្ងៃទី១៥ ខែមករា ឆ្នាំ២០២០").value).toBe("2020-01-15");
    expect(normalizeIssueDate("15.01.2020").value).toBe("2020-01-15");
    const future = normalizeIssueDate("ថ្ងៃទី១ ខែមករា ឆ្នាំ២៩៩៩");
    expect(future.valid).toBe(false);
    expect(future.warnings).toContain("implausible_date");
  });

  it("reads gender, and bilingual readings must agree", () => {
    expect(normalizeGender("ប្រុស").value).toBe("M");
    expect(normalizeGender("ស្រី").value).toBe("F");
    expect(normalizeGender("Sex: F").value).toBe("F");
    expect(normalizeGender("ប្រុស / Male").value).toBe("M");
    expect(normalizeGender("ស្រី / Female").value).toBe("F");
    for (const text of ["ប្រុស ស្រី", "ប្រុស / Female"]) {
      const result = normalizeGender(text);
      expect(result.value).toBeNull();
      expect(result.valid).toBe(false);
    }
  });

  it("never crash and never mark an empty value valid", () => {
    const samples = ["", " ", "្", "ៈ:", "O|lSBZ", "១២.០៣.១៩៩០", "​", "💳 ID", "a".repeat(200), "ä́"];
    for (const text of samples) {
      for (const [name, normalizer] of Object.entries(NORMALIZERS)) {
        const result = normalizer(text, name === "id_number" ? ID_SPEC : {});
        expect(result.value === null || typeof result.value === "string").toBe(true);
        if (result.valid) expect(result.value).toBeTruthy();
      }
    }
  });
});
