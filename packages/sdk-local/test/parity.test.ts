// The TypeScript engine against the Python engine on the same photos, with
// the real model. Needs a set from test/parity/make_parity_set.py and the
// model bundle; skipped otherwise:
//   KHMEROCR_PARITY_DIR=/tmp/parity KHMEROCR_TEST_BUNDLE=/path/to/khmerocr-id-v1 npm test
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { loadKhmerOCR, type KhmerOCRLocal } from "../src/node.js";

const dir = process.env["KHMEROCR_PARITY_DIR"];
const bundle = process.env["KHMEROCR_TEST_BUNDLE"];

describe.skipIf(!dir || !bundle)("parity with the Python engine", () => {
  let engine: KhmerOCRLocal | undefined;
  afterAll(() => engine?.close());

  it("extracts the same fields", async () => {
    engine = await loadKhmerOCR({ model: bundle! });
    const python = JSON.parse(readFileSync(join(dir!, "python.json"), "utf8")) as Record<string, any>;
    const labels = readFileSync(join(dir!, "labels.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
    let fields = 0, sameFields = 0, sameStatus = 0, sameWarnings = 0, pyCorrect = 0, tsCorrect = 0;
    const differences: string[] = [];
    for (const label of labels) {
      const expected = python[label.image];
      let actual: any;
      try {
        actual = await engine.extractNationalId(readFileSync(join(dir!, label.image)));
      } catch (error: any) {
        actual = { error: error.code };
      }
      if (expected.error || actual.error) {
        expect(actual.error, label.image).toBe(expected.error);
        continue;
      }
      sameStatus += Number(actual.status === expected.status && actual.review_required === expected.review_required);
      sameWarnings += Number(JSON.stringify(actual.warnings) === JSON.stringify(expected.warnings));
      for (const [name, field] of Object.entries<any>(expected.fields)) {
        fields++;
        const mine = actual.fields[name];
        if (mine.value === field.value && mine.valid === field.valid) sameFields++;
        else differences.push(`${label.image} ${name}: ${field.value} != ${mine.value}`);
        pyCorrect += Number(field.value === label.fields[name]);
        tsCorrect += Number(mine.value === label.fields[name]);
      }
    }
    console.log(`fields identical ${sameFields}/${fields}, status ${sameStatus}/${labels.length}, `
      + `warnings ${sameWarnings}/${labels.length}; correct: python ${pyCorrect}, typescript ${tsCorrect}`);
    // Pixel rounding differs between OpenCV builds (x86 float32, arm64 float16), so a
    // borderline read can flip; anything beyond that is a porting bug.
    expect(sameFields / fields, differences.slice(0, 10).join("\n")).toBeGreaterThanOrEqual(0.99);
    expect(sameStatus / labels.length).toBeGreaterThanOrEqual(0.97);
    expect(Math.abs(pyCorrect - tsCorrect) / fields).toBeLessThanOrEqual(0.01);
  });
});
