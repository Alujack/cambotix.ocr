import { cubicCoefficients } from "./resize.js";
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

const TAB_BITS = 5;
const TAB_SIZE = 1 << TAB_BITS;
const REMAP_BITS = 15;
const REMAP_SCALE = 1 << REMAP_BITS;

let cubicTable: Int32Array | null = null;

/**
 * OpenCV's 2D bicubic weight table: for each of 32 x 32 sub-pixel positions,
 * 16 fixed-point weights (scale 2^15) that sum to exactly 2^15.
 */
function bicubicTable(): Int32Array {
  if (cubicTable) return cubicTable;
  const table = new Int32Array(TAB_SIZE * TAB_SIZE * 16);
  const oneD = Array.from({ length: TAB_SIZE }, (_, k) => cubicCoefficients(k / TAB_SIZE));
  for (let ay = 0; ay < TAB_SIZE; ay++) {
    for (let ax = 0; ax < TAB_SIZE; ax++) {
      const base = (ay * TAB_SIZE + ax) * 16;
      let sum = 0;
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          const v = roundHalfEven(oneD[ay]![r]! * oneD[ax]![c]! * REMAP_SCALE);
          table[base + r * 4 + c] = v;
          sum += v;
        }
      }
      if (sum !== REMAP_SCALE) {
        const diff = sum - REMAP_SCALE;
        let minR = 2, minC = 2, maxR = 2, maxC = 2;
        for (let r = 2; r < 4; r++) {
          for (let c = 2; c < 4; c++) {
            const v = table[base + r * 4 + c]!;
            if (v < table[base + minR * 4 + minC]!) {
              minR = r;
              minC = c;
            } else if (v > table[base + maxR * 4 + maxC]!) {
              maxR = r;
              maxC = c;
            }
          }
        }
        if (diff < 0) table[base + maxR * 4 + maxC]! -= diff;
        else table[base + minR * 4 + minC]! -= diff;
      }
    }
  }
  cubicTable = table;
  return table;
}

/**
 * OpenCV `warpPerspective(image, matrix, (width, height), flags=INTER_CUBIC,
 * borderMode=BORDER_REPLICATE)`. `matrix` maps source to destination points.
 */
export function warpPerspectiveCubic(image: Raster, matrix: Matrix3, width: number,
                                     height: number): Raster {
  const m = invert3(matrix);
  const table = bicubicTable();
  const cn = image.channels;
  const sw = image.width;
  const sh = image.height;
  const out = createRaster(width, height, cn);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let wgt = m[6] * x + m[7] * y + m[8];
      wgt = wgt ? TAB_SIZE / wgt : 0;
      const fx = Math.max(-2147483648, Math.min(2147483647, (m[0] * x + m[1] * y + m[2]) * wgt));
      const fy = Math.max(-2147483648, Math.min(2147483647, (m[3] * x + m[4] * y + m[5]) * wgt));
      const X = roundHalfEven(fx);
      const Y = roundHalfEven(fy);
      const sx = Math.floor(X / TAB_SIZE) - 1;
      const sy = Math.floor(Y / TAB_SIZE) - 1;
      const base = (((Y & (TAB_SIZE - 1)) * TAB_SIZE) + (X & (TAB_SIZE - 1))) * 16;
      const dst = (y * width + x) * cn;
      for (let c = 0; c < cn; c++) {
        let acc = 0;
        for (let r = 0; r < 4; r++) {
          let yy = sy + r;
          yy = yy < 0 ? 0 : yy >= sh ? sh - 1 : yy;
          for (let k = 0; k < 4; k++) {
            let xx = sx + k;
            xx = xx < 0 ? 0 : xx >= sw ? sw - 1 : xx;
            acc += image.data[(yy * sw + xx) * cn + c]! * table[base + r * 4 + k]!;
          }
        }
        out.data[dst + c] = clampByte(Math.floor((acc + (1 << (REMAP_BITS - 1))) / REMAP_SCALE));
      }
    }
  }
  return out;
}
