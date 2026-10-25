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
    if (Number.isInteger(scaleX) && Number.isInteger(scaleY)) return resizeAreaFast(image, width, height, scaleX, scaleY);
    return resizeArea(image, width, height, scaleX, scaleY);
  }
  return resizeSeparable(image, width, height, interpolation);
}

/** OpenCV `computeResizeAreaTab`: source pixels covered by each destination pixel. */
interface AreaTabs {
  dst: Int32Array;
  src: Int32Array;
  weight: Float64Array;
}

function areaTabs(srcSize: number, dstSize: number, scale: number): AreaTabs {
  const dst: number[] = [];
  const src: number[] = [];
  const weight: number[] = [];
  const push = (d: number, s: number, w: number) => {
    dst.push(d);
    src.push(s);
    weight.push(w);
  };
  for (let dx = 0; dx < dstSize; dx++) {
    const fsx1 = dx * scale;
    const fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, srcSize - fsx1);
    let sx1 = Math.ceil(fsx1);
    let sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, srcSize - 1);
    sx1 = Math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) push(dx, sx1 - 1, (sx1 - fsx1) / cellWidth);
    for (let sx = sx1; sx < sx2; sx++) push(dx, sx, 1 / cellWidth);
    if (fsx2 - sx2 > 1e-3) push(dx, sx2, Math.min(Math.min(fsx2 - sx2, 1), cellWidth) / cellWidth);
  }
  return { dst: Int32Array.from(dst), src: Int32Array.from(src), weight: Float64Array.from(weight) };
}

function resizeArea(image: Raster, width: number, height: number,
                    scaleX: number, scaleY: number): Raster {
  const cn = image.channels;
  const x = areaTabs(image.width, width, scaleX);
  const y = areaTabs(image.height, height, scaleY);
  const out = createRaster(width, height, cn);
  const rowSize = width * cn;
  const sum = new Float64Array(rowSize);
  const row = new Float64Array(rowSize);
  const src = image.data;
  const xTaps = x.dst.length;
  let tap = 0;
  for (let dy = 0; dy < height; dy++) {
    sum.fill(0);
    for (; tap < y.dst.length && y.dst[tap] === dy; tap++) {
      const beta = y.weight[tap]!;
      row.fill(0);
      const base = y.src[tap]! * image.width * cn;
      if (cn === 1) {
        for (let k = 0; k < xTaps; k++) row[x.dst[k]!]! += src[base + x.src[k]!]! * x.weight[k]!;
      } else {
        for (let k = 0; k < xTaps; k++) {
          const d = x.dst[k]! * 3;
          const s = base + x.src[k]! * 3;
          const w = x.weight[k]!;
          row[d]! += src[s]! * w;
          row[d + 1]! += src[s + 1]! * w;
          row[d + 2]! += src[s + 2]! * w;
        }
      }
      for (let i = 0; i < rowSize; i++) sum[i]! += row[i]! * beta;
    }
    const outBase = dy * rowSize;
    for (let i = 0; i < rowSize; i++) out.data[outBase + i] = saturateByte(sum[i]!);
  }
  return out;
}

/** OpenCV `resizeAreaFast`: integer ratios average whole blocks. */
function resizeAreaFast(image: Raster, width: number, height: number, sx: number, sy: number): Raster {
  const cn = image.channels;
  const out = createRaster(width, height, cn);
  const scale = 1 / (sx * sy);
  const src = image.data;
  const stride = image.width * cn;
  for (let dy = 0; dy < height; dy++) {
    for (let dx = 0; dx < width; dx++) {
      for (let c = 0; c < cn; c++) {
        let sum = 0;
        for (let y = dy * sy; y < (dy + 1) * sy; y++) {
          const base = y * stride + c;
          for (let x = dx * sx; x < (dx + 1) * sx; x++) sum += src[base + x * cn]!;
        }
        out.data[(dy * width + dx) * cn + c] = saturateByte(sum * scale);
      }
    }
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

  const ks = xt.ksize;
  // Clamped (replicated-border) source offsets for every tap.
  const xOffsets = new Int32Array(width * ks);
  for (let dx = 0; dx < width; dx++) {
    for (let k = 0; k < ks; k++) {
      const sx = xt.start[dx]! + k;
      xOffsets[dx * ks + k] = (sx < 0 ? 0 : sx >= srcW ? srcW - 1 : sx) * cn;
    }
  }

  // Horizontal pass over every source row (fixed point, scale 2^11).
  const rowSize = width * cn;
  const rows = new Int32Array(srcH * rowSize);
  const src = image.data;
  for (let y = 0; y < srcH; y++) {
    const base = y * srcW * cn;
    const outBase = y * rowSize;
    for (let dx = 0; dx < width; dx++) {
      for (let c = 0; c < cn; c++) {
        let acc = 0;
        for (let k = 0; k < ks; k++) acc += src[base + xOffsets[dx * ks + k]! + c]! * xt.coef[dx * ks + k]!;
        rows[outBase + dx * cn + c] = acc;
      }
    }
  }

  // Vertical pass, then round away the 22 fractional bits.
  const out = createRaster(width, height, cn);
  const scale = 2 ** (2 * COEF_BITS);
  const delta = scale / 2;
  const sum = new Float64Array(rowSize);
  for (let dy = 0; dy < height; dy++) {
    sum.fill(0);
    for (let k = 0; k < yt.ksize; k++) {
      let sy = yt.start[dy]! + k;
      sy = sy < 0 ? 0 : sy >= srcH ? srcH - 1 : sy;
      const coef = yt.coef[dy * yt.ksize + k]!;
      const base = sy * rowSize;
      for (let i = 0; i < rowSize; i++) sum[i]! += rows[base + i]! * coef;
    }
    const outBase = dy * rowSize;
    for (let i = 0; i < rowSize; i++) out.data[outBase + i] = clampByte(Math.floor((sum[i]! + delta) / scale));
  }
  return out;
}
