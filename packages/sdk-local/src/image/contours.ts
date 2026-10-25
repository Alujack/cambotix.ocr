import type { Raster } from "./raster.js";

export type Point = [number, number];

// Neighbour offsets [dx, dy], counter-clockwise on screen (y grows downwards)
// starting east.
const DIRS: ReadonlyArray<Point> = [
  [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1],
];

/**
 * OpenCV `findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`: the outer
 * border of every connected component (8-connectivity) that is not inside a
 * hole of another component. Uses Suzuki and Abe's border following, the
 * algorithm OpenCV implements: each border starts at its first pixel in
 * raster order and runs counter-clockwise on screen. Points in the middle of
 * straight runs are dropped, as CHAIN_APPROX_SIMPLE does.
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
          contours.push(compressRuns(points).map(([x, y]) => [x - 1, y - 1] as Point));
        }
      }
      const value = f[p]!;
      if (value !== 1) lnbd = Math.abs(value);
    }
  }
  return contours;
}

/** Keep the first point and every point where the chain changes direction. */
function compressRuns(points: Point[]): Point[] {
  if (points.length < 3) return points;
  const kept: Point[] = [points[0]!];
  for (let k = 1; k < points.length; k++) {
    const [px, py] = points[k - 1]!;
    const [x, y] = points[k]!;
    const [nx, ny] = points[(k + 1) % points.length]!;
    if (x - px !== nx - x || y - py !== ny - y) kept.push(points[k]!);
  }
  return kept;
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

/** OpenCV's `Sklansky_` scan over sorted points from `start` to `end`. */
function sklansky(pts: ReadonlyArray<Point>, start: number, end: number, stack: number[],
                  nsign: number, sign2: number): number {
  const incr = end > start ? 1 : -1;
  let pprev = start;
  let pcur = pprev + incr;
  let pnext = pcur + incr;
  let size = 3;
  if (start === end || (pts[start]![0] === pts[end]![0] && pts[start]![1] === pts[end]![1])) {
    stack[0] = start;
    return 1;
  }
  stack[0] = pprev;
  stack[1] = pcur;
  stack[2] = pnext;
  const afterEnd = end + incr;
  while (pnext !== afterEnd) {
    const cury = pts[pcur]![1];
    const by = pts[pnext]![1] - cury;
    if (Math.sign(by) !== nsign) {
      const ax = pts[pcur]![0] - pts[pprev]![0];
      const ay = cury - pts[pprev]![1];
      const bx = pts[pnext]![0] - pts[pcur]![0];
      const convexity = ay * bx - ax * by;
      if (Math.sign(convexity) === sign2 && (ax !== 0 || ay !== 0)) {
        pprev = pcur;
        pcur = pnext;
        pnext += incr;
        stack[size++] = pnext;
      } else if (pprev === start) {
        pcur = pnext;
        stack[1] = pcur;
        pnext += incr;
        stack[2] = pnext;
      } else {
        stack[size - 2] = pnext;
        pcur = pprev;
        pprev = stack[size - 4]!;
        size--;
      }
    } else {
      pnext += incr;
      stack[size - 1] = pnext;
    }
  }
  return size - 1;
}

/**
 * OpenCV `convexHull(points)` (clockwise=False, returnPoints=True), including
 * its vertex order: OpenCV starts the hull where the vertices' positions in
 * the input form a monotonic sequence, and that order decides which corner
 * `approxPolyDP` starts from.
 */
export function convexHull(points: ReadonlyArray<Point>): Point[] {
  const total = points.length;
  if (total === 0) return [];
  const order = points.map((_, k) => k).sort((a, b) =>
    points[a]![0] - points[b]![0] || points[a]![1] - points[b]![1] || a - b);
  const pts = order.map((k) => points[k]!);
  let minY = 0;
  let maxY = 0;
  for (let k = 1; k < total; k++) {
    if (pts[minY]![1] > pts[k]![1]) minY = k;
    if (pts[maxY]![1] < pts[k]![1]) maxY = k;
  }
  const hull: number[] = [];
  if (pts[0]![0] === pts[total - 1]![0] && pts[0]![1] === pts[total - 1]![1]) {
    hull.push(0);
  } else {
    // Upper half; with clockwise=False OpenCV swaps the two chains.
    const tl: number[] = [];
    const tlCount = sklansky(pts, 0, maxY, tl, -1, 1);
    const tr: number[] = [];
    const trCount = sklansky(pts, total - 1, maxY, tr, -1, -1);
    const [upA, upACount, upB, upBCount] = [tr, trCount, tl, tlCount];
    for (let k = 0; k < upACount - 1; k++) hull.push(upA[k]!);
    for (let k = upBCount - 1; k > 0; k--) hull.push(upB[k]!);
    const stopIdx = upBCount > 2 ? upB[1]! : upACount > 2 ? upA[upACount - 2]! : -1;

    const bl: number[] = [];
    let blCount = sklansky(pts, 0, minY, bl, 1, -1);
    const br: number[] = [];
    let brCount = sklansky(pts, total - 1, minY, br, 1, 1);
    if (stopIdx >= 0) {
      const checkIdx = blCount > 2 ? bl[1]! : blCount + brCount > 2 ? br[2 - blCount]! : -1;
      if (checkIdx === stopIdx || (checkIdx >= 0 && pts[checkIdx]![0] === pts[stopIdx]![0]
          && pts[checkIdx]![1] === pts[stopIdx]![1])) {
        // All points on one line: the lower half mirrors the upper half.
        blCount = Math.min(blCount, 2);
        brCount = Math.min(brCount, 2);
      }
    }
    for (let k = 0; k < blCount - 1; k++) hull.push(bl[k]!);
    for (let k = brCount - 1; k > 0; k--) hull.push(br[k]!);
  }
  let indices = hull.map((k) => order[k]!);

  // Cyclic shift so the input positions ascend or descend, when they can.
  const n = indices.length;
  if (n >= 3) {
    let minIdx = 0;
    let maxIdx = 0;
    let lt = 0;
    for (let k = 1; k < n; k++) {
      const idx = indices[k]!;
      lt += indices[k - 1]! < idx ? 1 : 0;
      if (lt > 1 && lt <= k - 2) break;
      if (idx < indices[minIdx]!) minIdx = k;
      if (idx > indices[maxIdx]!) maxIdx = k;
    }
    const mmdist = Math.abs(maxIdx - minIdx);
    if ((mmdist === 1 || mmdist === n - 1) && (lt <= 1 || lt >= n - 2)) {
      const ascending = (maxIdx + 1) % n === minIdx;
      const i0 = ascending ? minIdx : maxIdx;
      if (i0 > 0) {
        const shifted: number[] = [];
        let j = i0;
        let k = 0;
        for (; k < n; k++) {
          const curr = indices[j]!;
          shifted.push(curr);
          const nextJ = j + 1 < n ? j + 1 : 0;
          if (k < n - 1 && ascending !== (curr < indices[nextJ]!)) break;
          j = nextJ;
        }
        if (k === n) indices = shifted;
      }
    }
  }
  return indices.map((k) => points[k]!);
}

/**
 * OpenCV `approxPolyDP(curve, epsilon, closed=True)`: Douglas-Peucker with
 * OpenCV 5's point-to-segment distance, its choice of starting points, and
 * its final pass that drops nearly collinear vertices.
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
      const segment2 = dx * dx + dy * dy;
      let maxDist2 = 0;
      while (next !== end) {
        const pt = curve[next]!;
        // Squared distance to the segment, times its squared length.
        const projection = (pt[0] - s[0]) * dx + (pt[1] - s[1]) * dy;
        let dist2: number;
        if (projection < 0) {
          dist2 = ((pt[0] - s[0]) ** 2 + (pt[1] - s[1]) ** 2) * segment2;
        } else if (projection > segment2) {
          dist2 = ((pt[0] - e[0]) ** 2 + (pt[1] - e[1]) ** 2) * segment2;
        } else {
          const cross = (pt[1] - s[1]) * dx - (pt[0] - s[0]) * dy;
          dist2 = cross * cross;
        }
        if (dist2 > maxDist2) {
          maxDist2 = dist2;
          split = next;
        }
        next = (next + 1) % count;
      }
      within = maxDist2 <= eps * segment2;
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
