// Mirrors packages/core-python/tests/test_national_id.py, test_documents.py
// and the fallback cases of test_hybrid_recognition.py.
import { describe, expect, it } from "vitest";
import { resolveConfig } from "../src/config.js";
import { TemplateExtractor } from "../src/documents/extractor.js";
import { loadBuiltinTemplate, templateForDocument, type DocumentTemplate } from "../src/documents/layout.js";
import { BUILTIN_TEMPLATES } from "../src/documents/templates.js";
import { createRaster, rotateQuarterTurns, toGray, type Raster } from "../src/image/raster.js";
import { assessQuality } from "../src/preprocessing/quality.js";
import { Recognizer, type RecognizedText, type RegionHint } from "../src/recognition/base.js";
import { ScriptRouter } from "../src/recognition/router.js";
import { renderTemplates } from "../scripts/sync-templates.mjs";
import { readFileSync } from "node:fs";
import { ScriptedRecognizer } from "./fakes.js";
import { decode, fixtures } from "./helpers.js";

const GOOD: Record<string, [string, number | null]> = {
  id_number: ["000000001", 0.97],
  full_name_km: ["គោត្តនាមនិងនាម: គំរូ សាកល្បង", 0.93],
  full_name_latin: ["KUMRU SAKALBANG", 0.95],
  date_of_birth: ["ថ្ងៃខែឆ្នាំកំណើត: ០១.០១.១៩៩០", 0.91],
  gender: ["ភេទ: ប្រុស", 0.9],
};
const STUDENT: Record<string, [string, number | null]> = {
  full_name_km: ["គំរូ សាកល្បង / KUMRU SAKALBANG", 0.93],
  gender: ["ស្រី / Female", 0.94],
  date_of_birth: ["01/02/2003", 0.95],
  department: ["ដេប៉ាតឺម៉ង់ គំរូ / Department of Samples", 0.9],
  degree: ["បរិញ្ញាបត្រ / Bachelor", 0.9],
  address_km: ["ភូមិគំរូ", 0.9],
  student_id: ["000000002", 0.9],
  issue_date: ["ថ្ងៃទី១៥ ខែមករា ឆ្នាំ២០២០", 0.9],
};

const photo = await decode(fixtures["photo"]);
const glare = await decode(fixtures["glare"]);
const national = () => templateForDocument("khmer_national_id");
const verified = (): DocumentTemplate => ({ ...national(), verified: true });

async function extract(recognizer: Recognizer, image: Raster = photo, template = national(),
                       config = resolveConfig()) {
  return new TemplateExtractor(recognizer, config, template).extract(image, assessQuality(image, config.quality));
}

const values = (r: Awaited<ReturnType<typeof extract>>) =>
  Object.fromEntries(Object.entries(r.fields).map(([k, f]) => [k, f.value]));

describe("National ID", () => {
  it("maps and normalizes fields", async () => {
    const result = await extract(new ScriptedRecognizer(GOOD));
    expect(values(result)).toEqual({ id_number: "000000001", full_name_km: "គំរូ សាកល្បង",
      full_name_latin: "KUMRU SAKALBANG", date_of_birth: "1990-01-01", gender: "M" });
    expect(Object.values(result.fields).every((f) => f.valid)).toBe(true);
    expect(result.metadata["card_detected"]).toBe(true);
  });

  it("always requires review while the template is unverified", async () => {
    const result = await extract(new ScriptedRecognizer(GOOD));
    expect(result.reviewRequired).toBe(true);
    expect(result.status).toBe("review_required");
    expect(result.warnings).toContain("layout_template_unverified");
  });

  it("succeeds with a verified template and confident fields", async () => {
    const result = await extract(new ScriptedRecognizer(GOOD), photo, verified());
    expect(result.status).toBe("success");
    expect(result.reviewRequired).toBe(false);
    expect(result.warnings).toEqual([]);
  });

  it.each([
    ["missing critical field", { id_number: ["", null] }, "field_not_detected:id_number"],
    ["low confidence", { date_of_birth: ["01.01.1990", 0.4] }, "low_confidence:date_of_birth"],
    ["missing confidence", { full_name_latin: ["KUMRU SAKALBANG", null] }, "low_confidence:full_name_latin"],
    ["corrected ID number", { id_number: ["O0000000I", 0.99] }, "ocr_corrected:id_number"],
  ] as const)("requires review for a %s", async (_name, change, warning) => {
    const result = await extract(new ScriptedRecognizer({ ...GOOD, ...change } as unknown as typeof GOOD), photo, verified());
    expect(result.reviewRequired).toBe(true);
    expect(result.warnings).toContain(warning);
  });

  it("drops an ambiguous non-critical value with a warning", async () => {
    const result = await extract(new ScriptedRecognizer({ ...GOOD, gender: ["ប្រុស ស្រី", 0.99] }), photo, verified());
    expect(result.fields["gender"]!.value).toBeNull();
    expect(result.warnings).toContain("ambiguous_value:gender");
  });

  it("fails when nothing is recognized", async () => {
    const result = await extract(new ScriptedRecognizer({}), photo, verified());
    expect(result.status).toBe("failed");
    expect(result.reviewRequired).toBe(true);
    for (const name of Object.keys(GOOD)) expect(result.warnings).toContain(`field_not_detected:${name}`);
  });

  it("forces review on glare", async () => {
    const result = await extract(new ScriptedRecognizer(GOOD), glare, verified());
    expect(result.reviewRequired).toBe(true);
    expect(result.warnings).toContain("glare_detected");
  });

  it("warns when no card boundary is found", async () => {
    const normalized = await decode(fixtures["normalized"]);
    const result = await extract(new ScriptedRecognizer(GOOD), normalized, verified());
    expect(result.warnings).toContain("card_boundary_not_detected");
    expect(result.reviewRequired).toBe(true);
  });

  it("retries an upside-down card rotated", async () => {
    class UpsideDownAware extends Recognizer {
      readonly name = "upside-down-aware";
      async recognizeLine(image: Raster, hint?: RegionHint): Promise<RecognizedText> {
        // On an upside-down card the ID-number region covers blank background.
        const gray = toGray(image);
        let darkest = 255;
        for (const v of gray.data) darkest = Math.min(darkest, v);
        if (darkest > 80) return { text: "", confidence: null };
        const [text, confidence] = GOOD[hint!.name!]!;
        return { text, confidence };
      }
    }
    const result = await extract(new UpsideDownAware(), rotateQuarterTurns(photo, 2), verified());
    expect(result.fields["id_number"]!.value).toBe("000000001");
    expect(result.metadata["quarter_turns"]).toBe(2);
    expect(result.warnings).toContain("orientation_corrected");
  });

  it("can skip the second orientation", async () => {
    const recognizer = new ScriptedRecognizer(GOOD);
    await extract(recognizer, photo, national(), resolveConfig({ tryAlternateOrientations: false }));
    expect(recognizer.calls).toHaveLength(Object.keys(GOOD).length);
  });
});

describe("fallback recognizer", () => {
  function hybrid(khmer: Record<string, [string, number | null]>, fallback: Record<string, [string, number | null]>) {
    const primary = new ScriptedRecognizer(khmer);
    const second = new ScriptedRecognizer(fallback);
    const router = new ScriptRouter(new Map<string, Recognizer>([["khmer", primary], ["tesseract", second]]),
      { khmer: ["khmer", "tesseract"] }, "tesseract");
    return { router, primary, second, config: resolveConfig({ tryAlternateOrientations: false }) };
  }

  it("is not called when the first read validates", async () => {
    const { router, primary, second, config } = hybrid(
      { full_name_km: ["គោត្តនាមនិងនាមៈគំរូ សាកល្បង", 0.99] },
      { full_name_km: ["wrong", 0.99], id_number: ["000000001", 0.97] });
    const result = await extract(router, photo, national(), config);
    expect(result.fields["full_name_km"]!.value).toBe("គំរូ សាកល្បង");
    expect(second.calls).not.toContain("full_name_km");
    // Only Khmer-script regions go to the Khmer model.
    expect(primary.calls).toEqual(national().fields.filter((f) => f.script === "khmer").map((f) => f.name));
    expect(result.warnings).not.toContain("fallback_recognizer:full_name_km");
  });

  it("is used when the first read fails validation", async () => {
    const { router, config } = hybrid({ full_name_km: ["", null] }, { full_name_km: ["គំរូ សាកល្បង", 0.9] });
    const result = await extract(router, photo, national(), config);
    expect(result.fields["full_name_km"]!.value).toBe("គំរូ សាកល្បង");
    expect(result.warnings).toContain("fallback_recognizer:full_name_km");
    expect(result.regions.find((r) => r.label === "full_name_km")!.text).toBe("គំរូ សាកល្បង");
  });

  it("reports the first read's problem when every read fails", async () => {
    const { router, config } = hybrid({ full_name_km: ["SOK", 0.99] }, { full_name_km: ["123", 0.99] });
    const result = await extract(router, photo, national(), config);
    expect(result.fields["full_name_km"]!.value).toBeNull();
    expect(result.warnings).toContain("invalid_script:full_name_km");
    expect(result.reviewRequired).toBe(true);
  });
});

describe("student ID", () => {
  it("splits bilingual lines and reads each shared region once", async () => {
    const recognizer = new ScriptedRecognizer(STUDENT);
    const result = await extract(recognizer, photo, templateForDocument("student_id_card"));
    expect(values(result)).toMatchObject({ full_name_km: "គំរូ សាកល្បង", full_name_latin: "KUMRU SAKALBANG",
      gender: "F", date_of_birth: "2003-02-01", student_id: "000000002", issue_date: "2020-01-15" });
    expect(result.documentType).toBe("student_id_card");
    expect(result.regions.filter((r) => r.label === "full_name_km")).toHaveLength(1);
    expect(result.regions.some((r) => r.label === "full_name_latin")).toBe(false);
  });

  it("does not guess the language side without a separator", async () => {
    const result = await extract(new ScriptedRecognizer({ ...STUDENT, full_name_km: ["KUMRU SAKALBANG", 0.95] }),
      photo, templateForDocument("student_id_card"));
    expect(result.fields["full_name_km"]!.value).toBeNull();
    expect(result.fields["full_name_latin"]!.value).toBeNull();
    expect(result.warnings).toContain("field_not_detected:full_name_latin");
  });

  it("keeps every region inside the card", () => {
    for (const field of loadBuiltinTemplate("student_id_card_front_v1").fields) {
      const [x, y, w, h] = field.box;
      expect(x + w).toBeLessThanOrEqual(1);
      expect(y + h).toBeLessThanOrEqual(1);
    }
  });
});

describe("templates", () => {
  it("rejects unknown document types", () => {
    expect(() => templateForDocument("passport")).toThrow(expect.objectContaining({ code: "UNSUPPORTED_DOCUMENT" }));
  });

  it("are identical to the Python engine's (run scripts/sync-templates.mjs if not)", () => {
    const current = readFileSync(new URL("../src/documents/templates.ts", import.meta.url), "utf8");
    expect(current).toBe(renderTemplates());
    expect(Object.keys(BUILTIN_TEMPLATES).sort()).toEqual(["khmer_national_id_front_v1", "student_id_card_front_v1"]);
  });

  it("white card backgrounds are not glare", () => {
    const card = createRaster(1152, 730, 3);
    card.data.fill(255);
    const q = assessQuality(card, resolveConfig().quality);
    expect(q.glareRatio).toBe(0);
  });
});
