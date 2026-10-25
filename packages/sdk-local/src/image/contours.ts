import type { Raster } from "./raster.js";

export type Point = [number, number];

// Neighbour offsets [dx, dy], counter-clockwise on screen (y grows downwards)
// starting east.
const DIRS: ReadonlyArray<Point> = [
  [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1],
];

/**
 * OpenCV `findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_NONE)`: the outer
 * border of every connected component (8-connectivity) that is not inside a
 * hole of another component. Uses Suzuki and Abe's border following, the
 * algorithm OpenCV implements. Points are pixel coordinates in tracing order.
 *
 * The Python engine asks for CHAIN_APPROX_SIMPLE; that only drops points in
 * the middle of straight runs, which changes neither the area nor the convex
 * hull, the only things the engine uses.
 */
export function findExternalContours(mask: Raster): Point[][] {
  const w = mask.width + 2;
  const h = mask.height + 2;
  const f = new Int32Array(w * h);
  for (let y = 0; y < mask.height; y++) {
    for (let x = 0; x < mask.width; x++) {
      if (mask.data[y * mask.width + x] !== 0) f[(y + 1) * w + x + 1] = 1;
    }
  }
  // Border number -> [isHole, parent]; border 1 is the frame.
  const borders: Array<{ hole: boolean; parent: number }> = [];
  borders[1] = { hole: true, parent: 0 };
  const contours: Point[][] = [];
  let nbd = 1;

  for (let i = 1; i < h - 1; i++) {
    let lnbd = 1;
    for (let j = 1; j < w - 1; j++) {
      const p = i * w + j;
      const fij = f[p]!;
      if (fij === 0) continue;
      let start: Point | null = null;
      let hole = false;
      if (fij === 1 && f[p - 1] === 0) {
        start = [j - 1, i];
      } else if (fij >= 1 && f[p + 1] === 0) {
        start = [j + 1, i];
        hole = true;
        if (fij > 1) lnbd = fij;
      }
      if (start !== null) {
        nbd++;
        const previous = borders[lnbd]!;
        const parent = hole === previous.hole ? previous.parent : lnbd;
        borders[nbd] = { hole, parent };
        const points = traceBorder(f, w, j, i, start, nbd);
        if (!hole && parent === 1) {
          contours.push(points.map(([x, y]) => [x - 1, y - 1] as Point));
        }
      }
      const value = f[p]!;
      if (value !== 1) lnbd = Math.abs(value);
    }
  }
  return contours;
}

function dirIndex(fromX: number, fromY: number, toX: number, toY: number): number {
  const dx = toX - fromX;
  const dy = toY - fromY;
  for (let d = 0; d < 8; d++) {
    if (DIRS[d]![0] === dx && DIRS[d]![1] === dy) return d;
  }
  throw new Error("not a neighbour");
}

/** Steps 3.1 to 3.5 of Suzuki and Abe; returns the border pixels. */
function traceBorder(f: Int32Array, w: number, j: number, i: number,
                     start: Point, nbd: number): Point[] {
  const points: Point[] = [];
  // 3.1: clockwise from `start` around (j, i) for a non-zero pixel.
  const startDir = dirIndex(j, i, start[0], start[1]);
  let found = -1;
  for (let k = 0; k < 8; k++) {
    const d = (startDir - k + 8) % 8;
    const [ox, oy] = DIRS[d]!;
    if (f[(i + oy) * w + j + ox] !== 0) {
      found = d;
      break;
    }
  }
  if (found < 0) {
    f[i * w + j] = -nbd;
    points.push([j, i]);
    return points;
  }
  const j1 = j + DIRS[found]![0];
  const i1 = i + DIRS[found]![1];
  let j2 = j1;
  let i2 = i1;
  let j3 = j;
  let i3 = i;
  for (;;) {
    points.push([j3, i3]);
    // 3.3: counter-clockwise around (j3, i3), starting after (j2, i2).
    const from = dirIndex(j3, i3, j2, i2);
    let j4 = 0;
    let i4 = 0;
    let eastZeroExamined = false;
    for (let k = 1; k <= 8; k++) {
      const d = (from + k) % 8;
      const [ox, oy] = DIRS[d]!;
      const value = f[(i3 + oy) * w + j3 + ox]!;
      if (value !== 0) {
        j4 = j3 + ox;
        i4 = i3 + oy;
        break;
      }
      if (d === 0) eastZeroExamined = true;
    }
    // 3.4
    const p3 = i3 * w + j3;
    if (eastZeroExamined) f[p3] = -nbd;
    else if (f[p3] === 1) f[p3] = nbd;
    // 3.5
    if (j4 === j && i4 === i && j3 === j1 && i3 === i1) break;
    j2 = j3;
    i2 = i3;
    j3 = j4;
    i3 = i4;
  }
  return points;
}

/** OpenCV `contourArea` (shoelace formula, unsigned). */
export function contourArea(points: ReadonlyArray<Point>): number {
  let area = 0;
  for (let k = 0; k < points.length; k++) {
    const [x0, y0] = points[k]!;
    const [x1, y1] = points[(k + 1) % points.length]!;
    area += x0 * y1 - x1 * y0;
  }
  return Math.abs(area) / 2;
}

/** OpenCV `arcLength(points, closed=True)`. */
export function arcLength(points: ReadonlyArray<Point>): number {
  let length = 0;
  for (let k = 0; k < points.length; k++) {
    const [x0, y0] = points[k]!;
    const [x1, y1] = points[(k + 1) % points.length]!;
    length += Math.hypot(x1 - x0, y1 - y0);
  }
  return length;
}

/** Convex hull (Andrew's monotone chain), without collinear points. */
export function convexHull(points: ReadonlyArray<Point>): Point[] {
  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const unique = sorted.filter((p, k) => k === 0 || p[0] !== sorted[k - 1]![0] || p[1] !== sorted[k - 1]![1]);
  if (unique.length < 3) return unique;
  const cross = (o: Point, a: Point, b: Point) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Point[] = [];
  for (const p of unique) {
    while (lower.length >= 2 && cross(lower[lower.length - 2]!, lower[lower.length - 1]!, p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper: Point[] = [];
  for (let k = unique.length - 1; k >= 0; k--) {
    const p = unique[k]!;
    while (upper.length >= 2 && cross(upper[upper.length - 2]!, upper[upper.length - 1]!, p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/**
 * OpenCV `approxPolyDP(curve, epsilon, closed=True)`: Douglas-Peucker with
 * OpenCV's choice of starting points and its final pass that drops nearly
 * collinear vertices.
 */
export function approxPolyDPClosed(curve: ReadonlyArray<Point>, epsilon: number): Point[] {
  const count = curve.length;
  if (count === 0) return [];
  const eps = epsilon * epsilon;
  const out: Point[] = [];

  // 1. Starting points: repeatedly the point farthest from the last one.
  let pos = 0;
  let rightStart = 0;
  let startPt: Point = curve[0]!;
  let leEps = false;
  for (let iter = 0; iter < 3; iter++) {
    pos = (pos + rightStart) % count;
    startPt = curve[pos]!;
    let maxDist = 0;
    for (let k = 1; k < count; k++) {
      const pt = curve[(pos + k) % count]!;
      const dx = pt[0] - startPt[0];
      const dy = pt[1] - startPt[1];
      const dist = dx * dx + dy * dy;
      if (dist > maxDist) {
        maxDist = dist;
        rightStart = k;
      }
    }
    leEps = maxDist <= eps;
  }

  // 2. Two slices between the starting points; 3. split until within epsilon.
  const stack: Array<[number, number]> = [];
  if (!leEps) {
    const a = pos % count;
    const b = (rightStart + a) % count;
    stack.push([b, a], [a, b]);
  } else {
    out.push(startPt);
  }
  while (stack.length > 0) {
    const [start, end] = stack.pop()!;
    const s = curve[start]!;
    let split = start;
    let within: boolean;
    let next = (start + 1) % count;
    if (next !== end) {
      const e = curve[end]!;
      const dx = e[0] - s[0];
      const dy = e[1] - s[1];
      let maxDist = 0;
      while (next !== end) {
        const pt = curve[next]!;
        const dist = Math.abs((pt[1] - s[1]) * dx - (pt[0] - s[0]) * dy);
        if (dist > maxDist) {
          maxDist = dist;
          split = next;
        }
        next = (next + 1) % count;
      }
      within = maxDist * maxDist <= eps * (dx * dx + dy * dy);
    } else {
      within = true;
    }
    if (within) {
      out.push(s);
    } else {
      stack.push([split, end], [start, split]);
    }
  }

  // 4. Drop vertices that lie almost on the line between their neighbours.
  const dst = out.slice();
  let n = dst.length;
  if (n <= 2) return dst;
  let readPos = n - 1;
  const read = (): Point => {
    const pt = dst[readPos]!;
    readPos = readPos + 1 >= n ? 0 : readPos + 1;
    return pt;
  };
  let start = read();
  let writePos = readPos;
  let pt = read();
  const total = n;
  let newCount = n;
  for (let k = 0; k < total && newCount > 2; k++) {
    const end = read();
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const dist = Math.abs((pt[0] - start[0]) * dy - (pt[1] - start[1]) * dx);
    const inner = (pt[0] - start[0]) * (end[0] - pt[0]) + (pt[1] - start[1]) * (end[1] - pt[1]);
    if (dist * dist <= 0.5 * eps * (dx * dx + dy * dy) && dx !== 0 && dy !== 0 && inner >= 0) {
      newCount--;
      dst[writePos] = start = end;
      writePos = writePos + 1 >= total ? 0 : writePos + 1;
      pt = read();
      k++;
      continue;
    }
    dst[writePos] = start = pt;
    writePos = writePos + 1 >= total ? 0 : writePos + 1;
    pt = end;
  }
  n = newCount;
  return dst.slice(0, n);
}

export interface RotatedRect {
  width: number;
  height: number;
  /** Corners, in order around the rectangle. */
  corners: [Point, Point, Point, Point];
}

/** Minimum-area enclosing rectangle of a convex polygon (OpenCV `minAreaRect` + `boxPoints`). */
export function minAreaRect(hull: ReadonlyArray<Point>): RotatedRect {
  if (hull.length === 0) {
    return { width: 0, height: 0, corners: [[0, 0], [0, 0], [0, 0], [0, 0]] };
  }
  let best: RotatedRect | null = null;
  let bestArea = Infinity;
  const edges = hull.length === 1 ? 1 : hull.length;
  for (let k = 0; k < edges; k++) {
    const a = hull[k]!;
    const b = hull[(k + 1) % hull.length]!;
    let ux = b[0] - a[0];
    let uy = b[1] - a[1];
    const len = Math.hypot(ux, uy);
    if (len === 0) {
      ux = 1;
      uy = 0;
    } else {
      ux /= len;
      uy /= len;
    }
    let minU = Infinity;
    let maxU = -Infinity;
    let minV = Infinity;
    let maxV = -Infinity;
    for (const [x, y] of hull) {
      const u = x * ux + y * uy;
      const v = -x * uy + y * ux;
      minU = Math.min(minU, u);
      maxU = Math.max(maxU, u);
      minV = Math.min(minV, v);
      maxV = Math.max(maxV, v);
    }
    const area = (maxU - minU) * (maxV - minV);
    if (area < bestArea) {
      bestArea = area;
      const corner = (u: number, v: number): Point => [u * ux - v * uy, u * uy + v * ux];
      best = {
        width: maxU - minU,
        height: maxV - minV,
        corners: [corner(minU, minV), corner(maxU, minV), corner(maxU, maxV), corner(minU, maxV)],
      };
    }
  }
  return best!;
}

/** OpenCV `boundingRect` of integer points. */
export function boundingRect(points: ReadonlyArray<Point>): { x: number; y: number; width: number; height: number } {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const [x, y] of points) {
    x0 = Math.min(x0, x);
    y0 = Math.min(y0, y);
    x1 = Math.max(x1, x);
    y1 = Math.max(y1, y);
  }
  return { x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1 };
}
