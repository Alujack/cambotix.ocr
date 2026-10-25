/**
 * Runs against a live KhmerOCR server. Skipped unless KHMEROCR_TEST_URL is set:
 *
 *   KHMEROCR_TEST_URL=http://127.0.0.1:8000 KHMEROCR_TEST_IMAGE=card.jpg \
 *   KHMEROCR_TEST_DOCUMENT=student KHMEROCR_TEST_API_KEY=... npm test
 */
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { KhmerOCRClient, KhmerOCRError, NATIONAL_ID_FIELDS, STUDENT_ID_FIELDS } from "../src/index.js";

// vitest collects a skipped describe body too, so the client needs a URL even then.
const url = process.env["KHMEROCR_TEST_URL"] ?? "";
const baseUrl = url || "http://127.0.0.1:8000";
const apiKey = process.env["KHMEROCR_TEST_API_KEY"];
const imagePath = process.env["KHMEROCR_TEST_IMAGE"];
const documentType = process.env["KHMEROCR_TEST_DOCUMENT"] === "student" ? "student" : "national";

describe.skipIf(!url)("live server", () => {
  const ocr = new KhmerOCRClient({ baseUrl, ...(apiKey ? { apiKey } : {}), timeoutMs: 120_000 });

  it("reports health", async () => {
    const health = await ocr.health();
    expect(health.status).toBe("ok");
    expect(health.model_loaded).toBe(true);
  });

  it("lists models", async () => {
    const models = await ocr.models();
    expect(models.active?.model?.model_id).toBeTypeOf("string");
    expect(Array.isArray(models.installed)).toBe(true);
  });

  it.skipIf(!apiKey)("rejects a wrong API key", async () => {
    const wrong = new KhmerOCRClient({ baseUrl, apiKey: "wrong-key-0123456789" });
    await expect(wrong.models()).rejects.toMatchObject({ code: "UNAUTHORIZED", status: 401 });
  });

  it.skipIf(!imagePath)("checks quality and extracts the card", async () => {
    const image = readFileSync(imagePath ?? "");
    const quality = await ocr.checkQuality(image);
    expect(quality.schema_version).toBe("1.0");
    expect(typeof quality.usable).toBe("boolean");

    const result = documentType === "student"
      ? await ocr.extractStudentId(image, { includeTextRegions: true })
      : await ocr.extractNationalId(image, { includeTextRegions: true });
    expect(result.schema_version).toBe("1.0");
    expect(["success", "review_required", "failed"]).toContain(result.status);
    const expected: readonly string[] = documentType === "student" ? STUDENT_ID_FIELDS : NATIONAL_ID_FIELDS;
    for (const name of Object.keys(result.fields)) expect(expected).toContain(name);
    for (const field of Object.values(result.fields)) {
      expect(field && (field.value === null || typeof field.value === "string")).toBe(true);
    }
    expect(Array.isArray(result.text_regions)).toBe(true);
  });

  it("maps a server-side rejection to KhmerOCRError", async () => {
    // A 1x1 PNG: valid format, but far too small to be a card.
    const tiny = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
      "base64");
    const error = await ocr.extractNationalId(tiny).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(KhmerOCRError);
    expect((error as KhmerOCRError).status).toBeGreaterThanOrEqual(400);
    expect((error as KhmerOCRError).requestId).toMatch(/^req_/);
  });
});
