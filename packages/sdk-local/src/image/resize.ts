import { clampByte, createRaster, roundHalfEven, saturateByte, type Raster } from "./raster.js";

export type Interpolation = "area" | "cubic" | "linear";

// OpenCV resizes 8-bit images with 11-bit fixed-point coefficients.
const COEF_BITS = 11;
const COEF_SCALE = 1 << COEF_BITS;

/** OpenCV `cv2.resize(image, (width, height), interpolation=...)`. */
export function resize(image: Raster, width: number, height: number,
                       interpolation: Interpolation): Raster {
  if (width === image.width && height === image.height) return image;
  // OpenCV derives the scale from the inverse ratio; the rounding matters on exact pixel boundaries.
  const scaleX = 1 / (width / image.width);
  const scaleY = 1 / (height / image.height);
  if (interpolation === "area" && scaleX >= 1 && scaleY >= 1) {
    return resizeArea(image, width, height, scaleX, scaleY);
  }
  return resizeSeparable(image, width, height, interpolation);
}

interface AreaTap {
  dst: number;
  src: number;
  weight: number;
}

/** OpenCV `computeResizeAreaTab`: source pixels covered by each destination pixel. */
function areaTabs(srcSize: number, dstSize: number, scale: number): AreaTap[] {
  const taps: AreaTap[] = [];
  for (let dx = 0; dx < dstSize; dx++) {
    const fsx1 = dx * scale;
    const fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, srcSize - fsx1);
    let sx1 = Math.ceil(fsx1);
    let sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, srcSize - 1);
    sx1 = Math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) {
      taps.push({ dst: dx, src: sx1 - 1, weight: (sx1 - fsx1) / cellWidth });
    }
    for (let sx = sx1; sx < sx2; sx++) {
      taps.push({ dst: dx, src: sx, weight: 1 / cellWidth });
    }
    if (fsx2 - sx2 > 1e-3) {
      taps.push({ dst: dx, src: sx2, weight: Math.min(Math.min(fsx2 - sx2, 1), cellWidth) / cellWidth });
    }
  }
  return taps;
}

function resizeArea(image: Raster, width: number, height: number,
                    scaleX: number, scaleY: number): Raster {
  const cn = image.channels;
  const xTabs = areaTabs(image.width, width, scaleX);
  const yTabs = areaTabs(image.height, height, scaleY);
  const out = createRaster(width, height, cn);
  const rowSize = width * cn;
  const sum = new Float64Array(rowSize);
  const row = new Float64Array(rowSize);
  let tap = 0;
  for (let dy = 0; dy < height; dy++) {
    sum.fill(0);
    for (; tap < yTabs.length && yTabs[tap]!.dst === dy; tap++) {
      const { src: sy, weight: beta } = yTabs[tap]!;
      row.fill(0);
      const base = sy * image.width * cn;
      for (const { dst, src, weight } of xTabs) {
        for (let c = 0; c < cn; c++) {
          row[dst * cn + c]! += image.data[base + src * cn + c]! * weight;
        }
      }
      for (let i = 0; i < rowSize; i++) sum[i]! += row[i]! * beta;
    }
    const outBase = dy * rowSize;
    for (let i = 0; i < rowSize; i++) out.data[outBase + i] = saturateByte(sum[i]!);
  }
  return out;
}

/** OpenCV `interpolateCubic` (A = -0.75). */
export function cubicCoefficients(x: number): [number, number, number, number] {
  const A = -0.75;
  const c0 = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
  const c1 = ((A + 2) * x - (A + 3)) * x * x + 1;
  const c2 = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
  return [c0, c1, c2, 1 - c0 - c1 - c2];
}

interface AxisTaps {
  /** First source index for each destination index (before clamping). */
  start: Int32Array;
  /** Fixed-point coefficients, `ksize` per destination index. */
  coef: Int32Array;
  ksize: number;
}

/** Source offsets and coefficients along one axis, as OpenCV's `resize` computes them. */
function axisTaps(srcSize: number, dstSize: number, interpolation: Interpolation): AxisTaps {
  const inverse = dstSize / srcSize;
  const scale = 1 / inverse;
  const ksize = interpolation === "cubic" ? 4 : 2;
  const start = new Int32Array(dstSize);
  const coef = new Int32Array(dstSize * ksize);
  for (let d = 0; d < dstSize; d++) {
    let f: number;
    let s: number;
    if (interpolation === "area") {
      // INTER_AREA when enlarging: linear interpolation with "area" weights.
      s = Math.floor(d * scale);
      f = Math.fround((d + 1) - (s + 1) * inverse);
      f = f <= 0 ? 0 : f - Math.floor(f);
    } else {
      f = Math.fround((d + 0.5) * scale - 0.5);
      s = Math.floor(f);
      f -= s;
    }
    let weights: number[];
    if (interpolation === "cubic") {
      weights = cubicCoefficients(f);
      start[d] = s - 1;
    } else {
      if (s < 0) {
        f = 0;
        s = 0;
      }
      if (s >= srcSize - 1) {
        f = 0;
        s = srcSize - 1;
      }
      weights = [1 - f, f];
      start[d] = s;
    }
    for (let k = 0; k < ksize; k++) {
      coef[d * ksize + k] = roundHalfEven(weights[k]! * COEF_SCALE);
    }
  }
  return { start, coef, ksize };
}

function resizeSeparable(image: Raster, width: number, height: number,
                         interpolation: Interpolation): Raster {
  const cn = image.channels;
  const xt = axisTaps(image.width, width, interpolation);
  const yt = axisTaps(image.height, height, interpolation);
  const srcW = image.width;
  const srcH = image.height;

  // Horizontal pass over every source row (fixed point, scale 2^11).
  const rows = new Int32Array(srcH * width * cn);
  for (let y = 0; y < srcH; y++) {
    const base = y * srcW * cn;
    for (let dx = 0; dx < width; dx++) {
      for (let c = 0; c < cn; c++) {
        let acc = 0;
        for (let k = 0; k < xt.ksize; k++) {
          let sx = xt.start[dx]! + k;
          sx = sx < 0 ? 0 : sx >= srcW ? srcW - 1 : sx;
          acc += image.data[base + sx * cn + c]! * xt.coef[dx * xt.ksize + k]!;
        }
        rows[(y * width + dx) * cn + c] = acc;
      }
    }
  }

  // Vertical pass, then round away the 22 fractional bits.
  const out = createRaster(width, height, cn);
  const rowSize = width * cn;
  const shift = 2 * COEF_BITS;
  const delta = 1 << (shift - 1);
  for (let dy = 0; dy < height; dy++) {
    for (let i = 0; i < rowSize; i++) {
      let acc = 0;
      for (let k = 0; k < yt.ksize; k++) {
        let sy = yt.start[dy]! + k;
        sy = sy < 0 ? 0 : sy >= srcH ? srcH - 1 : sy;
        acc += rows[sy * rowSize + i]! * yt.coef[dy * yt.ksize + k]!;
      }
      out.data[dy * rowSize + i] = clampByte(Math.floor((acc + delta) / 2 ** shift));
    }
  }
  return out;
}
