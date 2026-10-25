import { readFileSync } from "node:fs";
import { Ajv2020 } from "ajv/dist/2020.js";
import sharp from "sharp";
import { describe, expect, it } from "vitest";
import { KhmerOCRLocal, KhmerOCRError } from "../src/index.js";
import { readImageSize, sniffFormat } from "../src/image/input.js";
import { FakeSession, manifestFor, memoryPlatform } from "./fakes.js";
import { decode, fixtures } from "./helpers.js";

const TOKENS = "0123456789";
// The fake model reads "000000001" in every region.
const ID_CLASSES = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2];

const ajv = new Ajv2020({ strict: false });
const schema = (name: string) => ajv.compile(JSON.parse(
  readFileSync(new URL(`../../../schemas/${name}.schema.json`, import.meta.url), "utf8")));
const validDocument = schema("document-result");
const validQuality = schema("quality-report");
const validOcr = schema("ocr-result");

function bundleFiles(overrides: Record<string, string | Uint8Array> = {}) {
  const files: Record<string, string | Uint8Array> = { "rec/tokens.txt": `${TOKENS}\n`, "rec/model.onnx": "weights" };
  const recognizer = { backend: "onnx-ctc", options: { model: "rec/model.onnx", tokens_file: "rec/tokens.txt", token_offset: 1 } };
  const all: Record<string, string | Uint8Array> = { ...files, "manifest.json": manifestFor(files, recognizer) };
  return Object.fromEntries(Object.entries({ ...all, ...overrides }).map(([k, v]) => [`/m/${k}`, v]));
}

async function engine(files = bundleFiles(), session = () => new FakeSession(ID_CLASSES, 11)) {
  return KhmerOCRLocal.create(memoryPlatform(files, session), { model: "/m" });
}

const photoPng = new Uint8Array(Buffer.from(fixtures["photo"].png, "base64"));

describe("model bundle", () => {
  it("loads, describes itself and closes", async () => {
    const session = new FakeSession(ID_CLASSES, 11);
    const ocr = await engine(bundleFiles(), () => session);
    const info = ocr.info();
    expect(info.model?.model_id).toBe("test-model");
    expect(info.recognizer).toEqual({ backend: "onnx-ctc", model: "model.onnx", vocabulary_size: "10" });
    expect(Object.keys(info.templates)).toEqual(["khmer_national_id", "student_id_card"]);
    await ocr.close();
    expect(session.released).toBe(true);
    await expect(ocr.extractNationalId(photoPng)).rejects.toMatchObject({ code: "MODEL_NOT_AVAILABLE" });
  });

  it("rejects a modified model file", async () => {
    await expect(engine(bundleFiles({ "rec/model.onnx": "tampered" })))
      .rejects.toMatchObject({ code: "MODEL_NOT_AVAILABLE", details: { path: "rec/model.onnx" } });
  });

  it("rejects a manifest whose file list does not match its digest", async () => {
    const manifest = JSON.parse(bundleFiles()["/m/manifest.json"] as string);
    manifest.files.pop();
    await expect(engine(bundleFiles({ "manifest.json": JSON.stringify(manifest) })))
      .rejects.toMatchObject({ code: "MODEL_NOT_AVAILABLE" });
  });

  it("refuses paths outside the bundle", async () => {
    const files = { "../secret": "x" };
    const manifest = manifestFor(files, { backend: "onnx-ctc", options: { model: "../secret" } });
    await expect(engine(bundleFiles({ "manifest.json": manifest })))
      .rejects.toMatchObject({ code: "MODEL_NOT_AVAILABLE" });
  });

  it("explains a missing manifest", async () => {
    await expect(KhmerOCRLocal.create(memoryPlatform({}, () => new FakeSession([])), { model: "/nowhere" }))
      .rejects.toMatchObject({ code: "MODEL_NOT_AVAILABLE" });
  });
});

describe("results", () => {
  it("follow the document-result schema", async () => {
    const ocr = await engine();
    const result = await ocr.extractNationalId(photoPng, { includeTextRegions: true });
    expect(validDocument(result), JSON.stringify(validDocument.errors)).toBe(true);
    expect(result.fields.id_number).toMatchObject({ value: "000000001", valid: true, source: "front_region" });
    expect(result.fields.full_name_km?.valid).toBe(false);
    expect(result.text_regions).toHaveLength(5);
    expect(result.metadata).toMatchObject({ model_id: "test-model", engine: "local-node", card_detected: true });
    const plain = await ocr.extractNationalId(photoPng);
    expect(plain.text_regions).toBeUndefined();
    const student = await ocr.extractStudentId(photoPng);
    expect(validDocument(student)).toBe(true);
    expect(student.document_type).toBe("student_id_card");
  });

  it("follow the quality-report and ocr-result schemas", async () => {
    const ocr = await engine();
    const quality = await ocr.checkQuality(photoPng);
    expect(validQuality(quality), JSON.stringify(validQuality.errors)).toBe(true);
    expect(quality).toMatchObject({ usable: true, width: 800, height: 600, issues: [] });
    const text = await ocr.recognizeText(photoPng);
    expect(validOcr(text), JSON.stringify(validOcr.errors)).toBe(true);
    expect(text.document_type).toBe("general_text");
  });

  it("accept raw pixels such as canvas ImageData", async () => {
    const ocr = await engine();
    const { data, info } = await sharp(photoPng).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
    const result = await ocr.extractNationalId({ data: new Uint8ClampedArray(data), width: info.width, height: info.height });
    expect(result.fields.id_number?.value).toBe("000000001");
  });
});

describe("errors", () => {
  it.each([
    ["empty bytes", new Uint8Array(0), "INVALID_IMAGE"],
    ["unknown format", new Uint8Array([1, 2, 3, 4]), "UNSUPPORTED_MEDIA_TYPE"],
    ["truncated PNG", photoPng.subarray(0, 100), "INVALID_IMAGE"],
  ])("%s", async (_name, input, code) => {
    const ocr = await engine();
    const error = await ocr.extractNationalId(input).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(KhmerOCRError);
    expect(error).toMatchObject({ code });
  });

  it("rejects oversized files before decoding", async () => {
    const ocr = await KhmerOCRLocal.create(memoryPlatform(bundleFiles(), () => new FakeSession([])),
      { model: "/m", config: { maxImageBytes: 1000 } });
    await expect(ocr.checkQuality(photoPng)).rejects.toMatchObject({ code: "PAYLOAD_TOO_LARGE" });
  });

  it("rejects images over the pixel budget from their header", async () => {
    const ocr = await KhmerOCRLocal.create(memoryPlatform(bundleFiles(), () => new FakeSession([])),
      { model: "/m", config: { quality: { maxPixels: 1000 } } });
    await expect(ocr.checkQuality(photoPng)).rejects.toMatchObject({ code: "INVALID_IMAGE", details: { max_pixels: 1000 } });
  });

  it("refuses unusable photos with the reasons", async () => {
    const ocr = await engine();
    const blank = { data: new Uint8Array(800 * 600 * 3).fill(128), width: 800, height: 600, channels: 3 as const };
    await expect(ocr.extractNationalId(blank))
      .rejects.toMatchObject({ code: "IMAGE_QUALITY_TOO_LOW", details: { issues: ["low_contrast", "too_blurry"] } });
  });

  it("rejects unknown document types", async () => {
    const ocr = await engine();
    await expect(ocr.extractDocument(photoPng, "passport")).rejects.toMatchObject({ code: "UNSUPPORTED_DOCUMENT" });
  });

  it("stops when aborted", async () => {
    const ocr = await engine();
    const controller = new AbortController();
    controller.abort();
    await expect(ocr.extractNationalId(photoPng, { signal: controller.signal })).rejects.toMatchObject({ code: "ABORTED" });
  });

  it("runs concurrent calls one at a time", async () => {
    const ocr = await engine();
    const results = await Promise.all([ocr.extractNationalId(photoPng), ocr.checkQuality(photoPng),
      ocr.extractNationalId(photoPng)]);
    expect(results[0]).toEqual({ ...results[2], metadata: { ...results[2]!.metadata, processing_ms: results[0]!.metadata!["processing_ms"] } });
  });
});

describe("image headers", () => {
  it("reads sizes without decoding", async () => {
    const png = await sharp({ create: { width: 37, height: 21, channels: 3, background: "#fff" } }).png().toBuffer();
    const jpeg = await sharp({ create: { width: 37, height: 21, channels: 3, background: "#fff" } }).jpeg().toBuffer();
    const webp = await sharp({ create: { width: 37, height: 21, channels: 3, background: "#fff" } }).webp().toBuffer();
    const lossless = await sharp({ create: { width: 37, height: 21, channels: 3, background: "#fff" } }).webp({ lossless: true }).toBuffer();
    for (const data of [png, jpeg, webp, lossless]) {
      const bytes = new Uint8Array(data);
      expect(readImageSize(bytes, sniffFormat(bytes)!)).toEqual({ width: 37, height: 21 });
    }
  });
});
