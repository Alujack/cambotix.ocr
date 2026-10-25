import { createRaster, type Raster } from "./raster.js";

/** OpenCV `BORDER_REFLECT_101` (the default border): `gfedcb|abcdefgh|gfedcba`. */
export function reflect101(i: number, size: number): number {
  if (size === 1) return 0;
  while (i < 0 || i >= size) {
    i = i < 0 ? -i : 2 * size - i - 2;
  }
  return i;
}

/**
 * OpenCV `GaussianBlur(gray, (5, 5), 0)` on an 8-bit image. With sigma 0 and
 * a 5-tap kernel OpenCV uses the fixed kernel [1, 4, 6, 4, 1] / 16 and its
 * bit-exact fixed-point path, which this reproduces exactly.
 */
export function gaussianBlur5(gray: Raster): Raster {
  const { width: w, height: h, data } = gray;
  // [1, 4, 6, 4, 1] / 16 with 8 fractional bits: 16, 64, 96, 64, 16.
  const xs = reflectTable(w, 2);
  const rows = new Uint32Array(w * h);
  for (let y = 0; y < h; y++) {
    const base = y * w;
    for (let x = 0; x < w; x++) {
      rows[base + x] = 16 * (data[base + xs[x]!]! + data[base + xs[x + 4]!]!)
        + 64 * (data[base + xs[x + 1]!]! + data[base + xs[x + 3]!]!) + 96 * data[base + xs[x + 2]!]!;
    }
  }
  const ys = reflectTable(h, 2);
  const out = createRaster(w, h, 1);
  for (let y = 0; y < h; y++) {
    const r0 = ys[y]! * w, r1 = ys[y + 1]! * w, r2 = ys[y + 2]! * w, r3 = ys[y + 3]! * w, r4 = ys[y + 4]! * w;
    const base = y * w;
    for (let x = 0; x < w; x++) {
      const acc = 16 * (rows[r0 + x]! + rows[r4 + x]!) + 64 * (rows[r1 + x]! + rows[r3 + x]!) + 96 * rows[r2 + x]!;
      out.data[base + x] = (acc + 32768) >>> 16;
    }
  }
  return out;
}

/** `table[i + radius]` is the reflect-101 index of `i` for i in [-radius, size + radius). */
function reflectTable(size: number, radius: number): Int32Array {
  const table = new Int32Array(size + 2 * radius);
  for (let i = -radius; i < size + radius; i++) table[i + radius] = reflect101(i, size);
  return table;
}

/** Variance of OpenCV `Laplacian(gray, CV_64F)` (3x3 kernel, reflect-101 border). */
export function laplacianVariance(gray: Raster): number {
  const { width: w, height: h, data } = gray;
  const n = w * h;
  if (n === 0) return 0;
  const values = new Float64Array(n);
  const xs = reflectTable(w, 1);
  let sum = 0;
  for (let y = 0; y < h; y++) {
    const up = reflect101(y - 1, h) * w;
    const down = reflect101(y + 1, h) * w;
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const v = data[up + x]! + data[down + x]! + data[row + xs[x]!]! + data[row + xs[x + 2]!]!
        - 4 * data[row + x]!;
      values[row + x] = v;
      sum += v;
    }
  }
  const mean = sum / n;
  let sq = 0;
  for (let i = 0; i < n; i++) {
    const d = values[i]! - mean;
    sq += d * d;
  }
  return sq / n;
}

/** OpenCV's Otsu threshold for an 8-bit image (`getThreshVal_Otsu_8u`). */
export function otsuThreshold(gray: Raster): number {
  const hist = new Float64Array(256);
  const { data } = gray;
  for (let i = 0; i < data.length; i++) hist[data[i]!]!++;
  const scale = 1 / data.length;
  let mu = 0;
  for (let i = 0; i < 256; i++) mu += i * hist[i]!;
  mu *= scale;
  const FLT_EPSILON = 1.1920928955078125e-7;
  let mu1 = 0;
  let q1 = 0;
  let maxSigma = 0;
  let maxVal = 0;
  for (let i = 0; i < 256; i++) {
    const p = hist[i]! * scale;
    mu1 *= q1;
    q1 += p;
    const q2 = 1 - q1;
    if (Math.min(q1, q2) < FLT_EPSILON || Math.max(q1, q2) > 1 - FLT_EPSILON) continue;
    mu1 = (mu1 + i * p) / q1;
    const mu2 = (mu - q1 * mu1) / q2;
    const sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
    if (sigma > maxSigma) {
      maxSigma = sigma;
      maxVal = i;
    }
  }
  return maxVal;
}

/**
 * OpenCV `threshold(gray, 0, maxValue, THRESH_BINARY[_INV] + THRESH_OTSU)`:
 * pixels above the Otsu threshold become `maxValue` (or 0 when inverted).
 */
export function otsuBinarize(gray: Raster, maxValue: number, invert = false): Raster {
  const t = otsuThreshold(gray);
  const out = createRaster(gray.width, gray.height, 1);
  const above = invert ? 0 : maxValue;
  const below = invert ? maxValue : 0;
  for (let i = 0; i < gray.data.length; i++) out.data[i] = gray.data[i]! > t ? above : below;
  return out;
}

export function bitwiseNot(gray: Raster): Raster {
  const out = createRaster(gray.width, gray.height, 1);
  for (let i = 0; i < gray.data.length; i++) out.data[i] = 255 - gray.data[i]!;
  return out;
}

/**
 * Max (dilate) or min (erode) over a `kw` x `kh` rectangle centred on each
 * pixel. Pixels outside the image are ignored, which is what OpenCV's default
 * morphology border value amounts to.
 */
function rectFilter(gray: Raster, kw: number, kh: number, max: boolean): Raster {
  const { width: w, height: h, data } = gray;
  const ax = Math.floor(kw / 2);
  const ay = Math.floor(kh / 2);
  // Comparing with XOR 255 turns the min filter into a max filter.
  const flip = max ? 0 : 255;
  const tmp = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let v = data[row + x]! ^ flip;
      const x1 = Math.min(w - 1, x - ax + kw - 1);
      for (let xx = Math.max(0, x - ax); xx <= x1; xx++) {
        const u = data[row + xx]! ^ flip;
        if (u > v) v = u;
      }
      tmp[row + x] = v;
    }
  }
  const out = createRaster(w, h, 1);
  const col = new Uint8Array(w);
  for (let y = 0; y < h; y++) {
    const y1 = Math.min(h - 1, y - ay + kh - 1);
    col.set(tmp.subarray(y * w, y * w + w));
    for (let yy = Math.max(0, y - ay); yy <= y1; yy++) {
      const row = yy * w;
      for (let x = 0; x < w; x++) {
        const u = tmp[row + x]!;
        if (u > col[x]!) col[x] = u;
      }
    }
    const base = y * w;
    for (let x = 0; x < w; x++) out.data[base + x] = col[x]! ^ flip;
  }
  return out;
}

export function dilateRect(gray: Raster, kw: number, kh = kw, iterations = 1): Raster {
  let out = gray;
  for (let i = 0; i < iterations; i++) out = rectFilter(out, kw, kh, true);
  return out;
}

export function erodeRect(gray: Raster, kw: number, kh = kw): Raster {
  return rectFilter(gray, kw, kh, false);
}

/** OpenCV `morphologyEx(gray, MORPH_CLOSE, ones((kh, kw)))`. */
export function closeRect(gray: Raster, kw: number, kh = kw): Raster {
  return erodeRect(dilateRect(gray, kw, kh), kw, kh);
}

/**
 * OpenCV `morphologyEx(gray, MORPH_GRADIENT, getStructuringElement(MORPH_ELLIPSE, (3, 3)))`;
 * the 3x3 "ellipse" is a cross.
 */
export function crossGradient(gray: Raster): Raster {
  const { width: w, height: h, data } = gray;
  const out = createRaster(w, h, 1);
  const offsets: Array<[number, number]> = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let hi = 0;
      let lo = 255;
      for (const [dx, dy] of offsets) {
        const xx = x + dx;
        const yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const v = data[yy * w + xx]!;
        if (v > hi) hi = v;
        if (v < lo) lo = v;
      }
      out.data[y * w + x] = hi - lo;
    }
  }
  return out;
}
