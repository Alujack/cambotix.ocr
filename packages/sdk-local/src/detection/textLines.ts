/**
 * Classical (model-free) text-line detection, a fallback for recognizers
 * that only read single lines. Khmer has stacked vowels and subscript
 * consonants above and below the baseline, so the vertical closing kernel is
 * deliberately taller than one used for Latin text.
 */
import { boundingRect, findExternalContours } from "../image/contours.js";
import { closeRect, crossGradient, otsuBinarize } from "../image/filters.js";
import { toGray, type Raster } from "../image/raster.js";
import type { BoundingBox } from "../schemas.js";

export function detectTextLines(image: Raster, minHeight = 8): BoundingBox[] {
  const gray = toGray(image);
  const { width: w, height: h } = gray;
  const binary = otsuBinarize(crossGradient(gray), 255);
  const kernelW = Math.max(9, Math.floor(w / 60));
  const kernelH = Math.max(3, Math.floor(h / 200));
  const closed = closeRect(binary, kernelW, kernelH);

  const boxes: BoundingBox[] = [];
  for (const contour of findExternalContours(closed)) {
    const { x, y, width: bw, height: bh } = boundingRect(contour);
    if (bh < minHeight || bw < bh || bw * bh < 0.0002 * w * h) continue;
    if (bw > 0.98 * w && bh > 0.5 * h) continue;
    let filled = 0;
    for (let yy = y; yy < y + bh; yy++) {
      for (let xx = x; xx < x + bw; xx++) if (binary.data[yy * w + xx] !== 0) filled++;
    }
    if (filled / (bw * bh) < 0.08) continue;
    const pad = Math.max(2, Math.floor(bh / 6));
    const x0 = Math.max(0, x - pad);
    const y0 = Math.max(0, y - pad);
    const x1 = Math.min(w, x + bw + pad);
    const y1 = Math.min(h, y + bh + pad);
    boxes.push({ x: x0, y: y0, width: x1 - x0, height: y1 - y0 });
  }

  // Reading order: top-to-bottom by line, then left-to-right.
  const key = (b: BoundingBox) =>
    Math.floor((b.y + Math.floor(b.height / 2)) / Math.max(1, minHeight * 2)) * 10_000 + b.x;
  return boxes.sort((a, b) => key(a) - key(b));
}
