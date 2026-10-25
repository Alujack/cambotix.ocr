import { clampByte, createRaster, roundHalfEven, type Raster } from "./raster.js";
import type { Point } from "./contours.js";

export type Matrix3 = [number, number, number, number, number, number, number, number, number];

/** OpenCV `getPerspectiveTransform(src, dst)` for four point pairs. */
export function perspectiveTransform(src: ReadonlyArray<Point>, dst: ReadonlyArray<Point>): Matrix3 {
  const a: number[][] = [];
  const b: number[] = [];
  for (let k = 0; k < 4; k++) {
    const [x, y] = src[k]!;
    const [u, v] = dst[k]!;
    a.push([x, y, 1, 0, 0, 0, -x * u, -y * u]);
    b.push(u);
    a.push([0, 0, 0, x, y, 1, -x * v, -y * v]);
    b.push(v);
  }
  const c = solve(a, b);
  return [c[0]!, c[1]!, c[2]!, c[3]!, c[4]!, c[5]!, c[6]!, c[7]!, 1];
}

/** Gaussian elimination with partial pivoting (OpenCV uses LU for this system). */
function solve(a: number[][], b: number[]): number[] {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]!]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(m[r]![col]!) > Math.abs(m[pivot]![col]!)) pivot = r;
    }
    if (Math.abs(m[pivot]![col]!) < 1e-12) throw new Error("singular perspective transform");
    [m[col], m[pivot]] = [m[pivot]!, m[col]!];
    const p = m[col]!;
    for (let r = col + 1; r < n; r++) {
      const row = m[r]!;
      const factor = row[col]! / p[col]!;
      for (let k = col; k <= n; k++) row[k]! -= factor * p[k]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    const row = m[r]!;
    let sum = row[n]!;
    for (let k = r + 1; k < n; k++) sum -= row[k]! * x[k]!;
    x[r] = sum / row[r]!;
  }
  return x;
}

export function invert3(m: Matrix3): Matrix3 {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (det === 0) throw new Error("singular matrix");
  const s = 1 / det;
  return [
    A * s, -(b * i - c * h) * s, (b * f - c * e) * s,
    B * s, (a * i - c * g) * s, -(a * f - c * d) * s,
    C * s, -(a * h - b * g) * s, (a * e - b * d) * s,
  ];
}

const f32 = Math.fround;
/** Fused multiply-add rounded to float32 (exact product, one rounding). */
const fma32 = (a: number, b: number, c: number) => f32(a * b + c);

/** OpenCV 5 `bicubicWeights` (float32, A = -0.75), as its float32 SIMD kernel computes them. */
function bicubicWeights(alpha: number, out: Float64Array): void {
  const A = -0.75;
  const a2 = f32(alpha * alpha);
  const b = f32(1 - alpha);
  const b2 = f32(b * b);
  const w0 = f32(A * f32(alpha * b2));
  const w3 = f32(A * f32(a2 * b));
  const w1 = fma32(a2, fma32(A + 2, alpha, -(A + 3)), 1);
  out[0] = w0;
  out[1] = w1;
  out[2] = f32(f32(f32(1 - w0) - w1) - w3);
  out[3] = w3;
}

/**
 * OpenCV `warpPerspective(image, matrix, (width, height), flags=INTER_CUBIC,
 * borderMode=BORDER_REPLICATE)`. `matrix` maps source to destination points.
 *
 * Follows OpenCV 5's table-free bicubic warp in its float32 form (the one
 * x86 builds run). ARM builds use float16 instead, so OpenCV itself differs
 * by a grey level on some pixels between those machines.
 */
export function warpPerspectiveCubic(image: Raster, matrix: Matrix3, width: number,
                                     height: number): Raster {
  const m = invert3(matrix).map(f32) as Matrix3;
  const cn = image.channels;
  const sw = image.width;
  const sh = image.height;
  const minX = -Math.max(sw, 16);
  const maxX = Math.max(sw, 16) * 2;
  const minY = -Math.max(sh, 16);
  const maxY = Math.max(sh, 16) * 2;
  const out = createRaster(width, height, cn);
  const src = image.data;
  const wx = new Float64Array(4);
  const wy = new Float64Array(4);
  const cols = new Int32Array(4);
  const rowStride = sw * cn;
  for (let y = 0; y < height; y++) {
    const mx = f32(f32(y * m[1]) + m[2]);
    const my = f32(f32(y * m[4]) + m[5]);
    const mz = f32(f32(y * m[7]) + m[8]);
    for (let x = 0; x < width; x++) {
      const invz = 1 / (mz + m[6] * x);
      const xs = Math.min(Math.max(f32((mx + m[0] * x) * invz), minX), maxX);
      const ys = Math.min(Math.max(f32((my + m[3] * x) * invz), minY), maxY);
      const ix = Math.floor(xs);
      const iy = Math.floor(ys);
      bicubicWeights(f32(xs - ix), wx);
      bicubicWeights(f32(ys - iy), wy);
      for (let k = 0; k < 4; k++) {
        const xx = ix - 1 + k;
        cols[k] = (xx < 0 ? 0 : xx >= sw ? sw - 1 : xx) * cn;
      }
      const dst = (y * width + x) * cn;
      const c0 = cols[0]!, c1 = cols[1]!, c2 = cols[2]!, c3 = cols[3]!;
      const w0 = wx[0]!, w1 = wx[1]!, w2 = wx[2]!, w3 = wx[3]!;
      const inside = iy >= 1 && iy + 2 < sh;
      for (let c = 0; c < cn; c++) {
        let acc = 0;
        for (let r = 0; r < 4; r++) {
          const yy = iy - 1 + r;
          const row = (inside ? yy : yy < 0 ? 0 : yy >= sh ? sh - 1 : yy) * rowStride + c;
          let sum = f32(src[row + c0]! * w0);
          sum = f32(src[row + c1]! * w1 + sum);
          sum = f32(src[row + c2]! * w2 + sum);
          sum = f32(src[row + c3]! * w3 + sum);
          acc = f32(sum * wy[r]! + acc);
        }
        out.data[dst + c] = clampByte(roundHalfEven(acc));
      }
    }
  }
  return out;
}
