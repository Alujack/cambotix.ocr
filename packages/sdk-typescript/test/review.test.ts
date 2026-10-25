import { describe, expect, it } from "vitest";

import {
  NATIONAL_ID_FIELDS,
  STUDENT_ID_FIELDS,
  fieldsNeedingReview,
  parseWarnings,
  sniffContentType,
  type NationalIdResult,
} from "../src/index.js";
import { DOCUMENT, JPEG, PNG, WEBP } from "./helpers.js";

const result = DOCUMENT as unknown as NationalIdResult;

describe("parseWarnings", () => {
  it("splits field warnings and keeps document warnings whole", () => {
    expect(parseWarnings(result)).toEqual([
      { code: "layout_template_unverified", field: null },
      { code: "low_confidence", field: "date_of_birth" },
      { code: "field_not_detected", field: "full_name_km" },
    ]);
  });
});

describe("fieldsNeedingReview", () => {
  it("collects warned, invalid and empty fields once each", () => {
    expect(fieldsNeedingReview(result).sort()).toEqual(["date_of_birth", "full_name_km"]);
  });

  it("is empty for a clean result", () => {
    const clean: NationalIdResult = { ...result, warnings: [], fields: { gender: result.fields.gender! } };
    expect(fieldsNeedingReview(clean)).toEqual([]);
  });
});

describe("constants", () => {
  it("lists the template fields", () => {
    expect(NATIONAL_ID_FIELDS).toHaveLength(5);
    expect(STUDENT_ID_FIELDS).toContain("student_id");
  });
});

describe("sniffContentType", () => {
  it("recognizes JPEG, PNG and WebP only", () => {
    expect(sniffContentType(JPEG)).toBe("image/jpeg");
    expect(sniffContentType(PNG)).toBe("image/png");
    expect(sniffContentType(WEBP)).toBe("image/webp");
    expect(sniffContentType(new TextEncoder().encode("RIFF\0\0\0\0WAVE"))).toBeNull();
    expect(sniffContentType(new Uint8Array([0x47, 0x49, 0x46, 0x38]))).toBeNull();
    expect(sniffContentType(new Uint8Array())).toBeNull();
  });
});
