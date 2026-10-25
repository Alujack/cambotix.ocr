import { readFileSync } from "node:fs";
import sharp from "sharp";
import type { Raster } from "../src/image/raster.js";

export interface Encoded {
  shape: number[];
  b64?: string;
  png?: string;
}

export const fixtures = JSON.parse(
  readFileSync(new URL("./fixtures/opencv.json", import.meta.url), "utf8")) as Record<string, any>;

export async function decode(e: Encoded): Promise<Raster> {
  const [height, width, channels = 1] = e.shape as [number, number, number?];
  let data: Uint8Array;
  if (e.png) {
    const { data: pixels } = await sharp(Buffer.from(e.png, "base64")).raw().toBuffer({ resolveWithObject: true });
    data = new Uint8Array(pixels);
  } else {
    data = new Uint8Array(Buffer.from(e.b64!, "base64"));
  }
  return { width, height, channels: channels as 1 | 3, data };
}

/** Share of differing values and the largest difference. */
export function compare(actual: Raster, expected: Raster): { mismatch: number; maxDiff: number } {
  if (actual.width !== expected.width || actual.height !== expected.height
      || actual.channels !== expected.channels) {
    throw new Error(`shape ${actual.width}x${actual.height}x${actual.channels} != `
      + `${expected.width}x${expected.height}x${expected.channels}`);
  }
  let mismatched = 0;
  let maxDiff = 0;
  for (let i = 0; i < actual.data.length; i++) {
    const d = Math.abs(actual.data[i]! - expected.data[i]!);
    if (d) mismatched++;
    maxDiff = Math.max(maxDiff, d);
  }
  return { mismatch: mismatched / actual.data.length, maxDiff };
}
