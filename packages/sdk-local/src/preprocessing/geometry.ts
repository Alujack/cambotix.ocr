/** Card boundary detection, perspective correction, and rotation. */
import { canny } from "../image/canny.js";
import {
  approxPolyDPClosed,
  arcLength,
  contourArea,
  convexHull,
  findExternalContours,
  minAreaRect,
  type Point,
} from "../image/contours.js";
import { bitwiseNot, closeRect, dilateRect, gaussianBlur5, otsuBinarize } from "../image/filters.js";
import { rotateQuarterTurns, roundHalfEven, toGray, type Raster } from "../image/raster.js";
import { resize } from "../image/resize.js";
import { perspectiveTransform, warpPerspectiveCubic } from "../image/warp.js";

// ISO/IEC 7810 ID-1 card: 85.60 mm x 53.98 mm.
export const ID1_ASPECT = 85.6 / 53.98;
const DETECTION_LONG_SIDE = 800;
// The card must fill at least this share of the frame. Lower values let
// inner rectangles (such as the portrait photo on a tightly cropped card) be
// mistaken for the card itself.
const MIN_CARD_AREA_RATIO = 0.2;
const ASPECT_TOLERANCE = 0.25;

export type Quad = [Point, Point, Point, Point];

export interface CardNormalization {
  image: Raster;
  detected: boolean;
  /** Card corners in source-image coordinates (tl, tr, br, bl), if detected. */
  quad: Quad | null;
  /** Quarter turns (counter-clockwise) applied after warping. */
  quarterTurns: number;
}

/**
 * Locate the card, rectify it, and resize it to `size` (width, height). If no
 * card boundary is found the whole image is used, on the assumption that it
 * is already a tight crop of the card.
 */
export function normalizeCard(image: Raster, size: [number, number]): CardNormalization {
  const quad = detectCardQuad(image);
  let card = quad !== null ? warpQuad(image, quad) : image;
  let turns = 0;
  if (card.height > card.width) {
    card = rotateQuarterTurns(card, 1);
    turns = 1;
  }
  card = resize(card, size[0], size[1], card.width > size[0] ? "area" : "cubic");
  return { image: card, detected: quad !== null, quad, quarterTurns: turns };
}

export function detectCardQuad(image: Raster): Quad | null {
  const { width: w, height: h } = image;
  const scale = DETECTION_LONG_SIDE / Math.max(h, w);
  const small = resize(image, Math.max(1, roundHalfEven(w * scale)),
    Math.max(1, roundHalfEven(h * scale)), "area");
  const gray = gaussianBlur5(toGray(small));
  const area = gray.width * gray.height;

  for (const mask of candidateMasks(gray)) {
    const quad = largestCardQuad(mask, area);
    if (quad !== null) {
      // Reject quads that are simply the image border.
      if (coversWholeImage(quad, gray.width, gray.height)) return null;
      return quad.map(([x, y]) => [Math.fround(x / scale), Math.fround(y / scale)]) as Quad;
    }
  }
  return null;
}

const dist = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export function warpQuad(image: Raster, quad: Quad): Raster {
  const [tl, tr, br, bl] = quad;
  const width = roundHalfEven(Math.max(dist(tr, tl), dist(br, bl)));
  const height = roundHalfEven(Math.max(dist(bl, tl), dist(br, tr)));
  const dst: Quad = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]];
  return warpPerspectiveCubic(image, perspectiveTransform(quad, dst), width, height);
}

/** Order four points as top-left, top-right, bottom-right, bottom-left. */
export function orderQuad(points: ReadonlyArray<Point>): Quad {
  const pts = points.map(([x, y]) => [Math.fround(x), Math.fround(y)] as Point);
  const sums = pts.map(([x, y]) => x + y);
  const diffs = pts.map(([x, y]) => y - x);
  const argmin = (v: number[]) => v.indexOf(Math.min(...v));
  const argmax = (v: number[]) => v.indexOf(Math.max(...v));
  return [pts[argmin(sums)]!, pts[argmin(diffs)]!, pts[argmax(sums)]!, pts[argmax(diffs)]!];
}

function candidateMasks(gray: Raster): Raster[] {
  const edges = closeRect(dilateRect(canny(gray, 40, 120), 3, 3, 2), 7);
  const otsu = otsuBinarize(gray, 255);
  return [edges, otsu, bitwiseNot(otsu)];
}

function largestCardQuad(mask: Raster, imageArea: number): Quad | null {
  const contours = findExternalContours(mask)
    .map((points) => ({ points, area: contourArea(points) }))
    .sort((a, b) => b.area - a.area)
    .slice(0, 5);
  for (const { points, area } of contours) {
    if (area < MIN_CARD_AREA_RATIO * imageArea) break;
    const hull = convexHull(points);
    const approx = approxPolyDPClosed(hull, 0.02 * arcLength(hull));
    let quad: Quad;
    if (approx.length === 4) {
      quad = orderQuad(approx);
    } else {
      const rect = minAreaRect(hull);
      if (area / Math.max(rect.width * rect.height, 1) < 0.85) continue;
      quad = orderQuad(rect.corners);
    }
    if (plausibleCard(quad)) return quad;
  }
  return null;
}

function plausibleCard(quad: Quad): boolean {
  const [tl, tr, br, bl] = quad;
  const width = Math.max(dist(tr, tl), dist(br, bl));
  const height = Math.max(dist(bl, tl), dist(br, tr));
  if (Math.min(width, height) < 1) return false;
  const aspect = Math.max(width, height) / Math.min(width, height);
  return Math.abs(aspect - ID1_ASPECT) <= ASPECT_TOLERANCE;
}

function coversWholeImage(quad: Quad, width: number, height: number): boolean {
  const margin = 3;
  const xs = quad.map((p) => p[0]);
  const ys = quad.map((p) => p[1]);
  return Math.min(...xs) <= margin && Math.min(...ys) <= margin
    && Math.max(...xs) >= width - 1 - margin && Math.max(...ys) >= height - 1 - margin;
}
