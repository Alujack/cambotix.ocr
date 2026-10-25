// The TypeScript image code against OpenCV's own output (fixtures recorded by
// test/fixtures/make_opencv_fixtures.py). Exact where the algorithm is
// bit-exact in OpenCV; elsewhere within one grey level on a few pixels.
import { describe, expect, it } from "vitest";
import { canny } from "../src/image/canny.js";
import { connectedComponents } from "../src/image/components.js";
import { contourArea, findExternalContours } from "../src/image/contours.js";
import {
  bitwiseNot, closeRect, dilateRect, gaussianBlur5, laplacianVariance, otsuBinarize, otsuThreshold,
} from "../src/image/filters.js";
import { crop, toGray } from "../src/image/raster.js";
import { resize } from "../src/image/resize.js";
import { perspectiveTransform, warpPerspectiveCubic } from "../src/image/warp.js";
import { detectCardQuad, normalizeCard } from "../src/preprocessing/geometry.js";
import { assessQuality } from "../src/preprocessing/quality.js";
import { DEFAULT_QUALITY } from "../src/config.js";
import { trimToInk } from "../src/recognition/onnxCtc.js";
import { compare, decode, fixtures } from "./helpers.js";

const rgb = await decode(fixtures["rgb"]);
const gray = await decode(fixtures["gray"]);
const blurred = await decode(fixtures["blurred"]);
const otsu = await decode(fixtures["otsu"]);

describe("pixel operations match OpenCV", () => {
  it("converts RGB to grey exactly", () => {
    expect(compare(toGray(rgb), gray).mismatch).toBe(0);
  });

  it("blurs exactly (5x5 Gaussian)", () => {
    expect(compare(gaussianBlur5(gray), blurred).mismatch).toBe(0);
  });

  it.each([
    ["area_down", "rgb", "area"], ["area_int", "gray", "area"], ["area_up", "gray", "area"],
    ["area_mixed", "gray", "area"], ["cubic_up", "rgb", "cubic"], ["cubic_down", "gray", "cubic"],
  ] as const)("resizes (%s)", async (name, source, interpolation) => {
    const expected = await decode(fixtures["resize"][name]);
    const input = source === "rgb" ? rgb : gray;
    const { mismatch, maxDiff } = compare(resize(input, expected.width, expected.height, interpolation), expected);
    expect(maxDiff).toBeLessThanOrEqual(1);
    expect(mismatch).toBeLessThan(0.01);
  });

  it("computes the Laplacian variance", () => {
    expect(laplacianVariance(gray)).toBeCloseTo(fixtures["laplacian_var"], 6);
  });

  it("finds the Otsu threshold and binarizes", () => {
    expect(otsuThreshold(blurred)).toBe(fixtures["otsu_threshold"]);
    expect(compare(otsuBinarize(blurred, 255), otsu).mismatch).toBe(0);
  });

  it("detects edges like Canny", async () => {
    const { mismatch } = compare(canny(blurred, 40, 120), await decode(fixtures["edges"]));
    expect(mismatch).toBe(0);
  });

  it("dilates and closes", async () => {
    const closed = closeRect(dilateRect(canny(blurred, 40, 120), 3, 3, 2), 7);
    expect(compare(closed, await decode(fixtures["closed"])).mismatch).toBe(0);
  });

  it("finds the same external contours", () => {
    const areas = (mask: typeof otsu) =>
      findExternalContours(mask).map(contourArea).sort((a, b) => b - a);
    expect(areas(otsu)).toEqual(fixtures["contour_areas"]);
    expect(areas(bitwiseNot(otsu))).toEqual(fixtures["inverted_contour_areas"]);
  });

  it("labels connected components", () => {
    const mask = new Uint8Array(gray.data.length);
    gray.data.forEach((v, i) => { mask[i] = v >= 200 ? 1 : 0; });
    const { stats } = connectedComponents(mask, gray.width, gray.height);
    const rows = stats.map((s) => [s.x, s.y, s.width, s.height, s.area])
      .sort((a, b) => a[0]! - b[0]! || a[1]! - b[1]! || a[2]! - b[2]! || a[3]! - b[3]! || a[4]! - b[4]!);
    expect(rows).toEqual(fixtures["components"]);
  });

  it("warps perspective (bicubic)", async () => {
    const matrix = perspectiveTransform([[0, 0], [179, 0], [179, 119], [0, 119]],
      [[8, 5], [170, 12], [175, 110], [3, 116]]);
    matrix.forEach((v, i) => expect(v).toBeCloseTo(fixtures["matrix"][i], 9));
    const { mismatch, maxDiff } = compare(warpPerspectiveCubic(rgb, matrix, 180, 120), await decode(fixtures["warped"]));
    expect(maxDiff).toBeLessThanOrEqual(1);
    expect(mismatch).toBeLessThan(0.01);
  });
});

describe("pipeline steps match the Python engine", () => {
  it("finds the card corners", async () => {
    const quad = detectCardQuad(await decode(fixtures["photo"]));
    expect(quad).not.toBeNull();
    quad!.forEach(([x, y], i) => {
      expect(Math.abs(x - fixtures["quad"][i][0])).toBeLessThanOrEqual(1);
      expect(Math.abs(y - fixtures["quad"][i][1])).toBeLessThanOrEqual(1);
    });
  });

  it("rectifies the card", async () => {
    const card = normalizeCard(await decode(fixtures["photo"]), [1012, 638]);
    expect(card.detected).toBe(true);
    const { mismatch, maxDiff } = compare(card.image, await decode(fixtures["normalized"]));
    expect(maxDiff).toBeLessThanOrEqual(2);
    expect(mismatch).toBeLessThan(0.01);
  });

  it.each(["photo", "glare"])("measures quality (%s)", async (name) => {
    const expected = fixtures["quality"][name];
    const q = assessQuality(await decode(fixtures[name]), DEFAULT_QUALITY);
    expect(q.issues).toEqual(expected.issues);
    expect(q.blockingIssues).toEqual(expected.blocking_issues);
    expect(q.blurScore).toBeCloseTo(expected.blur_score, 3);
    expect(q.glareRatio).toBeCloseTo(expected.glare_ratio, 6);
    expect(q.brightness).toBeCloseTo(expected.brightness, 6);
    expect(q.contrast).toBeCloseTo(expected.contrast, 6);
  });

  it("trims a field crop to its ink", async () => {
    const normalized = await decode(fixtures["normalized"]);
    const trimmed = trimToInk(toGray(crop(normalized, 330, 200, 650, 90)));
    expect(trimmed && [trimmed.height, trimmed.width]).toEqual(fixtures["trimmed_shape"]);
  });
});
