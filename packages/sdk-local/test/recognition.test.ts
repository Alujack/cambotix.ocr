// Mirrors packages/core-python/tests/test_hybrid_recognition.py.
import { describe, expect, it } from "vitest";
import { createRaster, type Raster } from "../src/image/raster.js";
import { insertWordSpaces, OnnxCtcRecognizer, trimToInk } from "../src/recognition/onnxCtc.js";
import { ScriptRouter } from "../src/recognition/router.js";
import { checkOrtVersion } from "../src/recognition/runtimes.js";
import { FakeSession, ScriptedRecognizer } from "./fakes.js";

const TOKENS = "កខគ"; // class i -> TOKENS[i - 3]; class 0 = blank

/** A dark text line on white, as RGB. */
function line(width = 120, height = 40): Raster {
  const image = createRaster(width, height, 3);
  image.data.fill(255);
  for (let y = 12; y < 28; y++) {
    for (let x = 10; x < width - 10; x++) {
      const i = (y * width + x) * 3;
      image.data[i] = image.data[i + 1] = image.data[i + 2] = 20;
    }
  }
  return image;
}

const CARD_TOKENS = "ក្រ .ិBាុ៥"; // class i -> CARD_TOKENS[i - 1]: ក ្ រ space . ិ B ា ុ ៥

/** A recognizer whose model "reads" `text`; a blank between classes keeps repeats apart. */
function cardRecognizer(text: string, calibration: unknown = null): OnnxCtcRecognizer {
  const tokens = Array.from(CARD_TOKENS);
  const classes = Array.from(text).flatMap((ch) => [tokens.indexOf(ch) + 1, 0]);
  return new OnnxCtcRecognizer(new FakeSession(classes, tokens.length + 1), {
    modelFile: "rec.onnx", tokens: CARD_TOKENS, tokenOffset: 1,
    confidenceCalibration: calibration as number[][] | null });
}

describe("ONNX CTC recognizer", () => {
  it("collapses repeats and blanks", async () => {
    const session = new FakeSession([3, 3, 0, 3, 4, 4, 5, 0]);
    const result = await new OnnxCtcRecognizer(session, { modelFile: "rec.onnx", tokens: TOKENS })
      .recognizeLine(line());
    expect(result.text).toBe("កកខគ");
    expect(result.confidence).toBeCloseTo(1, 3);
    expect(session.lastDims?.[2]).toBe(32); // fixed input height
  });

  it("drops unmapped classes instead of misdecoding them", async () => {
    const rec = new OnnxCtcRecognizer(new FakeSession([3, 1, 4, 7]), { modelFile: "rec.onnx", tokens: TOKENS });
    expect((await rec.recognizeLine(line())).text).toBe("កខ");
  });

  it("returns nothing for a blank region without running the model", async () => {
    const session = new FakeSession([3]);
    const blank = createRaster(120, 40, 3);
    blank.data.fill(255);
    const result = await new OnnxCtcRecognizer(session, { modelFile: "rec.onnx", tokens: TOKENS }).recognizeLine(blank);
    expect(result).toEqual({ text: "", confidence: null });
    expect(session.lastDims).toBeNull();
  });

  it("trims crops to their ink", () => {
    const image = createRaster(200, 80, 1);
    image.data.fill(255);
    for (let y = 30; y < 50; y++) for (let x = 50; x < 150; x++) image.data[y * 200 + x] = 0;
    const trimmed = trimToInk(image);
    expect(trimmed && [trimmed.height, trimmed.width]).toEqual([20 + 8, 100 + 12]);
  });

  it("restores word spaces at wide gaps only", () => {
    const wide = createRaster(100, 32, 1);
    wide.data.fill(255);
    for (let y = 8; y < 24; y++) {
      for (let x = 5; x < 40; x++) wide.data[y * 100 + x] = 0;
      for (let x = 60; x < 95; x++) wide.data[y * 100 + x] = 0;
    }
    expect(insertWordSpaces(["ក", "ខ", "គ"], [2, 8, 18], wide, 25).join("")).toBe("កខ គ");
    const narrow = createRaster(100, 32, 1);
    narrow.data.fill(255);
    for (let y = 8; y < 24; y++) for (let x = 5; x < 95; x++) narrow.data[y * 100 + x] = 0;
    expect(insertWordSpaces(["ក", "ខ", "គ"], [2, 8, 18], narrow, 25).join("")).toBe("កខគ");
  });

  it.each(["ក្រ B .្", ".្ ក្រ B", "ិ ក្រ B ្"])("drops edge noise and flags the read (%s)", async (text) => {
    const result = await cardRecognizer(text).recognizeLine(line());
    expect(result.text).toBe("ក្រ B");
    expect(result.confidence).toBeCloseTo(0.5); // below the 0.85 review threshold
  });

  it("keeps clean reads, punctuation words and single-word lines", async () => {
    expect((await cardRecognizer("ក្រ B").recognizeLine(line())).confidence).toBeCloseTo(1, 3);
    expect((await cardRecognizer(". ក្រ .").recognizeLine(line())).text).toBe(". ក្រ .");
    expect((await cardRecognizer(".្").recognizeLine(line())).text).toBe(".្");
  });

  it.each(["ក្ ក", "ក ិក", "ក្.ក", "ក៥ា", "កាុ"])("gives impossible Khmer sequences zero confidence (%s)", async (text) => {
    const result = await cardRecognizer(text).recognizeLine(line());
    expect(result.text).toBe(text); // kept as read, but never trusted
    expect(result.confidence).toBe(0);
  });

  it("maps the mean confidence through the calibration", async () => {
    const rec = cardRecognizer("ក្រ", [[0, 0], [0.9, 0.5], [1, 0.8]]);
    expect((await rec.recognizeLine(line())).confidence).toBeCloseTo(0.8, 3);
  });

  it.each([[[[0.5, 0.5]]], [[[0.2, 0.1], [0.1, 0.2]]], [[[0, 0.9], [1, 0.1]]], [[[0, 0], [1, 1.5]]], ["bad"]])(
    "rejects invalid calibration %j", (points) => {
      expect(() => cardRecognizer("ក", points)).toThrow(expect.objectContaining({ code: "MODEL_NOT_AVAILABLE" }));
    });
});

describe("script router", () => {
  it("sends each script to its backends and closes them all", async () => {
    const khmer = new ScriptedRecognizer({ name_km: ["គំរូ", 0.9] });
    const latin = new ScriptedRecognizer({ name_latin: ["SOK", 0.9] });
    const router = new ScriptRouter(new Map([["khmer", khmer], ["latin", latin]]),
      { khmer: "khmer" }, "latin");
    expect((await router.recognizeLine(line(), { name: "name_km", script: "khmer" })).text).toBe("គំរូ");
    for (const script of ["latin", "digits", "mixed"] as const) {
      await router.recognizeLine(line(), { name: "name_latin", script });
    }
    expect(khmer.calls).toEqual(["name_km"]);
    expect(latin.calls).toEqual(["name_latin", "name_latin", "name_latin"]);
    await router.close();
    expect(khmer.closed && latin.closed).toBe(true);
  });

  it("rejects routes to unknown backends", () => {
    expect(() => new ScriptRouter(new Map([["a", new ScriptedRecognizer()]]), { khmer: "missing" }, "a"))
      .toThrow(expect.objectContaining({ code: "MODEL_NOT_AVAILABLE" }));
  });
});

describe("ONNX Runtime version", () => {
  it.each([["1.27.0", true], ["1.28.1", true], ["1.29.0", false], ["2.0.0", false], ["dev", false]])(
    "%s allowed: %s", (version, allowed) => {
      if (allowed) expect(() => checkOrtVersion(version)).not.toThrow();
      else expect(() => checkOrtVersion(version)).toThrow(expect.objectContaining({ code: "MODEL_NOT_AVAILABLE" }));
    });
});
